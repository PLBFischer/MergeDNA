"""
Local Encoder -- learnable tokenizer for MergeDNA.

Embeds raw nucleotide token IDs and progressively merges tokens
through a stack of LocalToMeAttentionBlock layers, producing a
compressed latent representation with an associated source matrix.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model.layers import RMSNorm, precompute_rope_freqs
from model.local_blocks import LocalToMeAttentionBlock


class LocalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 6,
        pad_token_id: int = 0,
        embed_dim: int = 1024,
        num_heads: int = 16,
        ffn_dim: int = 2752,
        num_layers: int = 4,
        local_window_size: int = 16,
        merge_group_dim: int = 64,
        max_seq_len: int = 4096,
        compression_ratio_mean: float = 0.5,
        compression_ratio_min: float = 0.4,
        compression_ratio_max: float = 0.6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.local_window_size = local_window_size
        self.compression_ratio_mean = compression_ratio_mean
        self.compression_ratio_min = compression_ratio_min
        self.compression_ratio_max = compression_ratio_max

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(embed_dim // num_heads, max_seq_len),
            persistent=False,
        )

        self.blocks = nn.ModuleList([
            LocalToMeAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                window_size=local_window_size,
                merge_group_dim=merge_group_dim,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

    def _sample_r_per_layer(
        self, content_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Sample per-layer merge counts relative to *content* (non-padding) length.

        Compression is applied only to real bases, so the ratio always means
        what it says regardless of how much padding a sequence has.

        During training the ratio is sampled from a Gaussian; at eval time the
        mean ratio is used deterministically.

        Args:
            content_lengths: (B,) int tensor of non-padding base counts.

        Returns:
            r_schedule: (B, num_layers) int tensor — per-sequence per-layer
                merge counts.
        """
        B = content_lengths.shape[0]
        device = content_lengths.device

        if self.training:
            ratio = torch.empty(1, device=device).normal_(
                self.compression_ratio_mean,
                (self.compression_ratio_max - self.compression_ratio_min) / 4,
            ).clamp(self.compression_ratio_min, self.compression_ratio_max).item()
        else:
            ratio = self.compression_ratio_mean

        L_targets = (content_lengths.float() * ratio).long().clamp(min=1)   # (B,)
        total_to_remove = content_lengths - L_targets                         # (B,)

        base = total_to_remove // self.num_layers                             # (B,)
        remainder = total_to_remove - base * self.num_layers                  # (B,)

        r_schedule = base.unsqueeze(1).expand(B, self.num_layers).clone()
        r_schedule[:, -1] += remainder

        return r_schedule

    def forward(
        self,
        input_ids: torch.Tensor,
        r_per_layer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode token IDs into merged latent tokens.

        Args:
            input_ids: (B, N) integer token ids.
            r_per_layer: optional (B, num_layers) int tensor of predetermined
                merge counts (used in the AMTM pass to reuse pass-1 schedule).
                When ``None`` a new schedule is sampled via
                ``_sample_r_per_layer`` based on each sequence's content length.

        Returns:
            z_l: (B, L, D) merged token embeddings.
            source: (B, L, N) source matrix mapping merged -> original positions.
            pos_ids: (B, L) position ids of keeper tokens.
            span_ids: (B, L) number of base tokens in each merged token.
            r_per_layer: (B, num_layers) the merge schedule used (passed back
                for the AMTM pass).
        """
        B, N = input_ids.shape
        x = self.token_embed(input_ids)

        # Real-base mask: 1 for content positions, 0 for padding.
        # Kept at original length N; source accumulates the mapping as tokens merge.
        pad_mask = (input_ids != self.token_embed.padding_idx).float()  # (B, N)

        if r_per_layer is None:
            content_lengths = pad_mask.sum(dim=-1).long()  # (B,)
            r_per_layer = self._sample_r_per_layer(content_lengths)

        source = (
            torch.eye(N, device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        pos_ids = (
            torch.arange(N, device=x.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        span_ids = torch.ones(B, N, dtype=torch.long, device=x.device)

        for layer_idx, block in enumerate(self.blocks):
            r = r_per_layer[:, layer_idx]  # (B,) per-sequence merge count
            x, source, pos_ids, span_ids = block(
                x, source, pos_ids, span_ids, r=r, rope_freqs=self.rope_freqs,
                pad_mask=pad_mask,
            )

        x = self.norm(x)
        return x, source, pos_ids, span_ids, r_per_layer
