"""
Latent Encoder -- global context modelling for MergeDNA.

``forward`` runs the full-attention stack without any token merging (used
in pre-training passes 1 and 3).

``forward_merged`` additionally applies a per-layer TokenMergeModule after
each TransformerBlock, progressively compressing L tokens down to K using
the same uniform adjacent-pair merge schedule as the Local Encoder (used
in pre-training pass 2).  It returns the source_prime matrix that tracks
which original base positions belong to each latent token; this matrix
drives the AMTM mask in pass 3.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model.layers import RMSNorm, precompute_rope_freqs
from model.token_merge import TokenMergeModule
from model.transformer_block import TransformerBlock


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
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
            )
            for _ in range(num_layers)
        ])
        # One TokenMergeModule per layer; only used in forward_merged (pass 2).
        self.merge_modules = nn.ModuleList([
            TokenMergeModule(embed_dim, merge_group_dim)
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
        pos_ids: Optional[torch.Tensor] = None,
        span_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full-attention stack without token merging (passes 1 and 3).

        Args:
            z: (B, L, D) merged embeddings from the local encoder.
            pos_ids: (B, L) position ids.
            span_ids: (B, L) span lengths.

        Returns:
            z_prime: (B, L, D) contextualised latent embeddings.
        """
        for block in self.blocks:
            z = block(z, self.rope_freqs, pos_ids, span_ids)
        return self.norm(z)

    def forward_merged(
        self,
        z: torch.Tensor,
        source: torch.Tensor,
        pos_ids: torch.Tensor,
        span_ids: torch.Tensor,
        r_per_layer: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Full-attention stack with progressive adjacent-pair merging (pass 2).

        Each layer runs its TransformerBlock then reduces the sequence by r
        adjacent-pair merges, distributing the total compression budget
        uniformly — identical in spirit to how the Local Encoder merges N→L.

        Args:
            z: (B, L, D) merged embeddings from the local encoder.
            source: (B, L, N_orig) source matrix from the local encoder.
            pos_ids: (B, L) position ids.
            span_ids: (B, L) span lengths.
            r_per_layer: optional predetermined merge schedule. When ``None``
                a new schedule is computed via ``_compute_r_per_layer``.

        Returns:
            z_k: (B, K, D) compressed latent embeddings.
            source_prime: (B, K, N_orig) source matrix mapping latent -> original positions.
            pos_ids_k: (B, K) position ids of kept tokens.
            span_ids_k: (B, K) span lengths of kept tokens.
            r_per_layer: the merge schedule that was used.
        """
        _, L, _ = z.shape
        if r_per_layer is None:
            r_per_layer = self._compute_r_per_layer(L)

        for layer_idx, (block, merge) in enumerate(zip(self.blocks, self.merge_modules)):
            z = block(z, self.rope_freqs, pos_ids, span_ids)
            r = r_per_layer[layer_idx] if layer_idx < len(r_per_layer) else 0
            z, source, pos_ids, span_ids = merge(z, source, pos_ids, span_ids, r)

        z = self.norm(z)
        return z, source, pos_ids, span_ids, r_per_layer
