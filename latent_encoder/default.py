"""
Latent Encoder -- global context modelling for MergeDNA.

Processes the merged local tokens (L) through a deep stack of
FullToMeAttentionBlocks, progressively merging them down to K tokens —
analogous to how the Local Encoder merges N base tokens down to L.

Each block performs full self-attention followed by adjacent-pair token
merging.  The encoder outputs a source_prime matrix (B, K, N_orig) that
tracks which original base positions feed into each latent token, used
by the AMTM loss in pre-training pass 3.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model.layers import RMSNorm, precompute_rope_freqs
from model.local_blocks import FullToMeAttentionBlock


class LatentEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        ffn_dim: int = 2752,
        num_layers: int = 20,
        merge_group_dim: int = 64,
        max_seq_len: int = 4096,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.compression_ratio = compression_ratio
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(embed_dim // num_heads, max_seq_len),
            persistent=False,
        )

        self.blocks = nn.ModuleList([
            FullToMeAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                merge_group_dim=merge_group_dim,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

    def _compute_r_per_layer(self, L: int) -> List[int]:
        """Compute a uniform per-layer merge schedule for L → K tokens."""
        K = max(1, int(L * self.compression_ratio))
        total_to_remove = L - K
        base = total_to_remove // self.num_layers
        remainder = total_to_remove - base * self.num_layers
        r_per_layer = [base] * self.num_layers
        r_per_layer[-1] += remainder
        return r_per_layer

    def forward(
        self,
        z: torch.Tensor,
        source: torch.Tensor,
        pos_ids: torch.Tensor,
        span_ids: torch.Tensor,
        r_per_layer: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Progressively merge local tokens into latent tokens.

        Args:
            z: (B, L, D) merged embeddings from the local encoder.
            source: (B, L, N_orig) source matrix from the local encoder.
            pos_ids: (B, L) position ids.
            span_ids: (B, L) span lengths.
            r_per_layer: optional predetermined merge schedule. When ``None``
                a new schedule is computed via ``_compute_r_per_layer``.

        Returns:
            z_prime: (B, K, D) latent token embeddings.
            source_prime: (B, K, N_orig) source matrix mapping latent -> original positions.
            pos_ids_k: (B, K) position ids of kept tokens.
            span_ids_k: (B, K) span lengths of kept tokens.
            r_per_layer: the merge schedule that was used.
        """
        _, L, _ = z.shape
        if r_per_layer is None:
            r_per_layer = self._compute_r_per_layer(L)

        for layer_idx, block in enumerate(self.blocks):
            r = r_per_layer[layer_idx] if layer_idx < len(r_per_layer) else 0
            z, source, pos_ids, span_ids = block(
                z, source, pos_ids, span_ids, r=r, rope_freqs=self.rope_freqs,
            )

        z = self.norm(z)
        return z, source, pos_ids, span_ids, r_per_layer
