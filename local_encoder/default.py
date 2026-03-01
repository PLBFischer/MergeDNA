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
        head_dim: int = 64,
        ffn_dim: int = 2752,
        num_layers: int = 4,
        local_window_size: int = 16,
        merge_group_dim: int = 64,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
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
            precompute_rope_freqs(head_dim, max_seq_len),
            persistent=False,
        )

        self.blocks = nn.ModuleList([
            LocalToMeAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                ffn_dim=ffn_dim,
                window_size=local_window_size,
                merge_group_dim=merge_group_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

    def _sample_r_per_layer(self, N: int, device: torch.device) -> List[int]:
        """Sample per-layer merge counts achieving a sampled compression ratio.

        During training the ratio is sampled from a Gaussian; at eval time the
        mean ratio is used deterministically.
        """
        if self.training:
            ratio = torch.empty(1, device=device).normal_(
                self.compression_ratio_mean,
                (self.compression_ratio_max - self.compression_ratio_min) / 4,
            ).clamp(self.compression_ratio_min, self.compression_ratio_max).item()
        else:
            ratio = self.compression_ratio_mean

        L_target = max(1, int(N * ratio))
        total_to_remove = N - L_target
        W = self.local_window_size

        r_per_layer: List[int] = []
        current_len = N
        for i in range(self.num_layers):
            n_windows = max(1, current_len // W)
            remaining_layers = self.num_layers - i
            remove_this_layer = total_to_remove // remaining_layers
            r = max(0, min(remove_this_layer // n_windows, W - 2))
            actual_removed = r * n_windows
            total_to_remove -= actual_removed
            current_len -= actual_removed
            r_per_layer.append(r)

        return r_per_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        r_per_layer: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Encode token IDs into merged latent tokens.

        Args:
            input_ids: (B, N) integer token ids.
            r_per_layer: optional predetermined merge schedule. When ``None``
                a new schedule is sampled via ``_sample_r_per_layer``.

        Returns:
            z_l: (B, L, D) merged token embeddings.
            source: (B, L, N) source matrix mapping merged -> original positions.
            pos_ids: (B, L) position ids of keeper tokens.
            r_per_layer: the merge schedule that was used (useful for AMTM pass).
        """
        B, N = input_ids.shape
        x = self.token_embed(input_ids)

        if r_per_layer is None:
            r_per_layer = self._sample_r_per_layer(N, input_ids.device)

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

        for layer_idx, block in enumerate(self.blocks):
            r = r_per_layer[layer_idx] if layer_idx < len(r_per_layer) else 0
            x, source, pos_ids = block(
                x, source, pos_ids, r=r, rope_freqs=self.rope_freqs,
            )

        x = self.norm(x)
        return x, source, pos_ids, r_per_layer
