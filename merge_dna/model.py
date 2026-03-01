"""
Full MergeDNA model.

Assembles the four hierarchical components described in Sec 3.2:
  1. Local Encoder  (learnable tokenizer)
  2. Latent Encoder  (global context modelling)
  3. Latent Decoder  (token-level reconstruction, pre-training only)
  4. Local Decoder   (base-level reconstruction / detokenizer)

The model supports three operating modes:
  - ``encoder_only``: Local Encoder + Latent Encoder (for classification)
  - ``full``: all four components (for pre-training and generative tasks)
  - ``pretrain``: three forward-pass pre-training loop (returns all
    intermediate quantities needed to compute the combined loss)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LocalAttentionBlock, LocalToMeAttentionBlock, TransformerBlock
from .config import MergeDNAConfig
from .layers import RMSNorm, precompute_rope_freqs
from .token_merge import GlobalTokenMergeModule, token_unmerge


@dataclass
class MergeDNAOutput:
    """Container for model outputs across the three pre-training passes."""

    # Pass 1: full reconstruction (MTR)
    logits_mtr: Optional[torch.Tensor] = None  # (B, N, vocab)

    # Pass 2: latent selection + reconstruction (latent MTR)
    logits_latent_mtr: Optional[torch.Tensor] = None  # (B, N, vocab)
    latent_source: Optional[torch.Tensor] = None  # S' from global merge

    # Pass 3: adaptive masked token modelling (AMTM)
    logits_amtm: Optional[torch.Tensor] = None  # (B, N, vocab)
    mask_n: Optional[torch.Tensor] = None  # (B, N) bool mask

    # Encoder-only output (for downstream classification)
    latent_embeds: Optional[torch.Tensor] = None  # (B, L, D)


class MergeDNA(nn.Module):
    def __init__(self, config: MergeDNAConfig):
        super().__init__()
        self.config = config
        D = config.embed_dim

        # --- Shared embedding & output head ---
        self.token_embed = nn.Embedding(config.vocab_size, D, padding_idx=config.pad_token_id)
        # DESIGN DECISION: Output head for base prediction. Not described
        # explicitly but implied by the cross-entropy loss formulation.
        self.output_head = nn.Linear(D, config.vocab_size, bias=False)
        self.final_norm = RMSNorm(D)

        # --- RoPE frequencies ---
        self.rope_freqs = precompute_rope_freqs(config.head_dim, config.max_seq_len)

        # --- 1. Local Encoder (learnable tokenizer) ---
        self.local_encoder = nn.ModuleList([
            LocalToMeAttentionBlock(
                dim=D,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                window_size=config.local_window_size,
                merge_group_dim=config.merge_group_dim,
                dropout=config.dropout,
            )
            for _ in range(config.local_encoder_layers)
        ])
        self.local_encoder_norm = RMSNorm(D)

        # --- 2. Latent Encoder (global context) ---
        self.latent_encoder = nn.ModuleList([
            TransformerBlock(
                dim=D,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            )
            for _ in range(config.latent_encoder_layers)
        ])
        self.latent_encoder_norm = RMSNorm(D)

        # Global merge module for latent selection (pre-training pass 2)
        self.global_merge = GlobalTokenMergeModule(D, config.merge_group_dim)

        # --- 3. Latent Decoder (pre-training only) ---
        self.latent_decoder = nn.ModuleList([
            TransformerBlock(
                dim=D,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            )
            for _ in range(config.latent_decoder_layers)
        ])
        self.latent_decoder_norm = RMSNorm(D)

        # --- 4. Local Decoder (detokenizer) ---
        self.local_decoder = nn.ModuleList([
            LocalAttentionBlock(
                dim=D,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                window_size=config.local_window_size,
                dropout=config.dropout,
            )
            for _ in range(config.local_decoder_layers)
        ])
        self.local_decoder_norm = RMSNorm(D)

    # ------------------------------------------------------------------
    # Sub-component forward methods
    # ------------------------------------------------------------------

    def _run_local_encoder(
        self,
        x_embed: torch.Tensor,
        r_per_layer: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the Local Encoder stack.

        Returns:
            z_l: (B, L, D) merged token embeddings
            source: (B, L, N) source matrix
            pos_ids: (B, L) position ids of keeper tokens
        """
        B, N, D = x_embed.shape
        source = (
            torch.eye(N, device=x_embed.device, dtype=x_embed.dtype)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        pos_ids = (
            torch.arange(N, device=x_embed.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        x = x_embed
        for layer_idx, block in enumerate(self.local_encoder):
            r = r_per_layer[layer_idx] if layer_idx < len(r_per_layer) else 0
            x, source, pos_ids = block(
                x, source, pos_ids, r=r, rope_freqs=self.rope_freqs,
            )
        x = self.local_encoder_norm(x)
        return x, source, pos_ids

    def _run_latent_encoder(
        self,
        z: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the Latent Encoder (full attention)."""
        for block in self.latent_encoder:
            z = block(z, self.rope_freqs, pos_ids)
        return self.latent_encoder_norm(z)

    def _run_latent_decoder(
        self,
        z: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the Latent Decoder."""
        for block in self.latent_decoder:
            z = block(z, self.rope_freqs, pos_ids)
        return self.latent_decoder_norm(z)

    def _run_local_decoder(
        self,
        z_n: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the Local Decoder (local attention, no merging)."""
        for block in self.local_decoder:
            z_n = block(z_n, self.rope_freqs, pos_ids)
        return self.local_decoder_norm(z_n)

    def _decode_to_logits(
        self,
        z_hat_l: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Unmerge + Local Decoder + output head -> logits (B, N, vocab)."""
        z_n = token_unmerge(z_hat_l, source)
        N = source.shape[2]
        base_pos = torch.arange(N, device=z_n.device).unsqueeze(0).expand(z_n.shape[0], -1)
        z_n = self._run_local_decoder(z_n, base_pos)
        return self.output_head(self.final_norm(z_n))

    # ------------------------------------------------------------------
    # Compression ratio sampling (Sec 3.3)
    # ------------------------------------------------------------------

    def _sample_r_per_layer(
        self, N: int, device: torch.device
    ) -> List[int]:
        """Sample per-layer merge counts achieving a sampled compression ratio.

        DESIGN DECISION: The paper samples L from a Gaussian centered at N/2
        with variance ensuring L in [0.4N, 0.6N].  We sample the compression
        ratio, then distribute merges evenly across the 4 Local Encoder
        layers.  Each layer merges r tokens per window of (current) effective
        window size. We keep the original window partitioning fixed.
        """
        cfg = self.config
        if self.training:
            ratio = torch.empty(1, device=device).normal_(
                cfg.compression_ratio_mean,
                (cfg.compression_ratio_max - cfg.compression_ratio_min) / 4,
            ).clamp(cfg.compression_ratio_min, cfg.compression_ratio_max).item()
        else:
            ratio = cfg.target_local_compression

        L_target = max(1, int(N * ratio))
        total_to_remove = N - L_target
        n_layers = cfg.local_encoder_layers
        W = cfg.local_window_size

        r_per_layer = []
        current_len = N
        for i in range(n_layers):
            n_windows = max(1, current_len // W)
            remaining_layers = n_layers - i
            remove_this_layer = total_to_remove // remaining_layers
            r = max(0, min(remove_this_layer // n_windows, W - 2))
            actual_removed = r * n_windows
            total_to_remove -= actual_removed
            current_len -= actual_removed
            r_per_layer.append(r)

        return r_per_layer

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------

    def forward_encoder_only(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode-only path for classification / downstream tasks.

        DESIGN DECISION: Pooling not specified in paper. We return the
        full (B, L, D) latent embeddings; mean-pooling is applied externally.
        """
        B, N = input_ids.shape
        x_embed = self.token_embed(input_ids)
        r_per_layer = self._sample_r_per_layer(N, input_ids.device)
        z_l, source, pos_ids = self._run_local_encoder(x_embed, r_per_layer)
        z_prime = self._run_latent_encoder(z_l, pos_ids)
        return z_prime

    def forward_full(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Full autoencoder forward -> logits (B, N, vocab)."""
        B, N = input_ids.shape
        x_embed = self.token_embed(input_ids)
        r_per_layer = self._sample_r_per_layer(N, input_ids.device)
        z_l, source, pos_ids = self._run_local_encoder(x_embed, r_per_layer)
        z_prime = self._run_latent_encoder(z_l, pos_ids)
        z_hat_l = self._run_latent_decoder(z_prime, pos_ids)
        logits = self._decode_to_logits(z_hat_l, source)
        return logits

    def forward_pretrain(
        self, input_ids: torch.Tensor
    ) -> MergeDNAOutput:
        """Three-pass pre-training forward (Sec 3.4, Eq. 8).

        Returns a ``MergeDNAOutput`` containing logits and masks for all
        three loss terms.
        """
        B, N = input_ids.shape
        cfg = self.config
        output = MergeDNAOutput()

        # ---- Pass 1: full MTR (Eq. 6) ----
        x_embed = self.token_embed(input_ids)
        r_per_layer = self._sample_r_per_layer(N, input_ids.device)
        z_l, source, pos_ids = self._run_local_encoder(x_embed, r_per_layer)
        z_prime = self._run_latent_encoder(z_l, pos_ids)
        z_hat_l = self._run_latent_decoder(z_prime, pos_ids)
        output.logits_mtr = self._decode_to_logits(z_hat_l, source)

        # ---- Pass 2: latent selection + reconstruction ----
        # Freeze local encoder (phi not updated in this pass).
        # We use z_l from pass 1 (detached from local encoder graph).
        z_l_detached = z_l.detach()
        z_prime_2 = self._run_latent_encoder(z_l_detached, pos_ids)

        K = max(1, int(z_l_detached.shape[1] * cfg.target_latent_compression))
        z_k, source_prime = self.global_merge(z_prime_2, source.detach(), K)

        z_hat_l_2 = token_unmerge(z_k, source_prime)
        # source_prime is (B, K, N_orig_from_local) but we need to map
        # back to base space.  Since source maps local->base and
        # source_prime maps latent->local, the combined mapping is
        # source_prime already tracks original positions because
        # GlobalTokenMergeModule receives the local source matrix.
        # So z_hat_l_2 is already (B, L, D).

        # Run latent decoder on the upsampled tokens, then local decoder
        z_hat_l_2 = self._run_latent_decoder(z_hat_l_2, pos_ids)
        output.logits_latent_mtr = self._decode_to_logits(z_hat_l_2, source)
        output.latent_source = source_prime

        # ---- Pass 3: Adaptive Masked Token Modeling (Eq. 7) ----
        mask_l = self._compute_adaptive_mask(source_prime, z_l.shape[1], K)
        mask_n = self._map_mask_to_base(mask_l, source)  # (B, N)
        output.mask_n = mask_n

        masked_ids = input_ids.clone()
        masked_ids[mask_n] = cfg.pad_token_id

        x_embed_masked = self.token_embed(masked_ids)
        z_l_m, source_m, pos_ids_m = self._run_local_encoder(
            x_embed_masked, r_per_layer,
        )
        z_prime_m = self._run_latent_encoder(z_l_m, pos_ids_m)
        z_hat_l_m = self._run_latent_decoder(z_prime_m, pos_ids_m)
        output.logits_amtm = self._decode_to_logits(z_hat_l_m, source_m)

        return output

    # ------------------------------------------------------------------
    # AMTM mask computation (Sec 3.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_adaptive_mask(
        source_prime: torch.Tensor,
        L: int,
        K: int,
    ) -> torch.Tensor:
        """Compute per-local-token mask using importance weighting from
        the global merge outcome S'.

        Paper (Sec 3.4): g_i = sum_j S'_{i,j}, w_i = 1/g_i.
        For each local token j in group i: P_L(j) proportional to w_i / g_i.
        Then sample K tokens without replacement according to P_L.

        Returns:
            mask_l: (B, L) boolean mask over local tokens.
        """
        B = source_prime.shape[0]
        # source_prime: (B, K, L_local) -- maps K latent tokens -> L local tokens
        # But after global merge the second dim tracks original N_orig positions.
        # We need to find, for each local-token index, which latent group it
        # belongs to.  source_prime rows sum the original base positions.
        # DESIGN DECISION: We approximate the importance weighting. Since
        # source_prime encodes which original positions each latent token
        # covers, we derive per-local-token importance from how many local
        # tokens were aggregated into its latent group.

        # Compute the number of original positions per latent token
        group_sizes = source_prime.sum(dim=-1)  # (B, K)
        # Weight inversely proportional to group size
        weights = 1.0 / (group_sizes + 1e-8)  # (B, K)

        # Build per-local-token probability.  Each local token's probability
        # is the weight of its parent latent group divided by group size.
        # Since we may not have a clean L-dim mapping, we spread evenly.
        # Assign probability proportional to 1/g_i^2 for tokens in group i.
        # We construct this via matrix multiply.
        # P(base_pos j) = sum_i [ S'_{i,j} * w_i / g_i ] then we sum over
        # base positions that belong to each local token.  For simplicity
        # we use a uniform importance-based selection at the local level.

        # Simpler approach: each latent token "owns" some base positions.
        # Tokens in small groups are important. We sample K local token
        # indices weighted by how "important" their region is.
        # prob per base position:
        w_per_base = torch.bmm(
            weights.unsqueeze(1) / (group_sizes.unsqueeze(1) + 1e-8),
            source_prime.float(),
        ).squeeze(1)  # (B, N_orig)

        # We need to select K positions, but these are base-level.
        # The paper selects K *local* tokens.  We'll select K base positions
        # and later map to local via source.  For local-level selection we
        # just need a (B, L) mask.  Since we work at base level:
        w_per_base = w_per_base.clamp(min=0)
        total = w_per_base.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs = w_per_base / total  # (B, N_orig)

        n_mask = min(K, probs.shape[1])
        # Sample without replacement
        indices = torch.multinomial(probs, n_mask, replacement=False)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        return mask  # (B, N_orig) -- this is base-level mask

    @staticmethod
    def _map_mask_to_base(
        mask_l: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Map a local-token mask to base-level positions using source matrix.

        If _compute_adaptive_mask already returns a base-level mask, this is
        a passthrough.
        """
        if mask_l.shape[1] == source.shape[2]:
            return mask_l
        # mask_l: (B, L), source: (B, L, N)
        # A base position is masked if its merged token is masked.
        expanded = mask_l.unsqueeze(-1).float()  # (B, L, 1)
        base_mask = torch.bmm(source.transpose(1, 2).float(), expanded)
        return base_mask.squeeze(-1) > 0.5

    def forward(
        self,
        input_ids: torch.Tensor,
        mode: str = "pretrain",
    ):
        """Dispatch to the appropriate forward method.

        Args:
            input_ids: (B, N) integer token ids
            mode: one of "pretrain", "full", "encoder_only"
        """
        if mode == "pretrain":
            return self.forward_pretrain(input_ids)
        elif mode == "full":
            return self.forward_full(input_ids)
        elif mode == "encoder_only":
            return self.forward_encoder_only(input_ids)
        else:
            raise ValueError(f"Unknown mode: {mode}")
