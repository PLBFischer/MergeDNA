"""
Latent Decoder -- token-level reconstruction for MergeDNA.

Processes latent embeddings through a stack of full-attention
TransformerBlocks to reconstruct merged-token representations
(used during pre-training only).
"""

from typing import Optional

import torch
import torch.nn as nn

from model.layers import RMSNorm, precompute_rope_freqs
from model.transformer_block import TransformerBlock


class LatentDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        ffn_dim: int = 2752,
        num_layers: int = 4,
        max_seq_len: int = 4096,
    ):
        super().__init__()
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
        self.norm = RMSNorm(embed_dim)

    def forward(
        self,
        z: torch.Tensor,
        pos_ids: torch.Tensor,
        span_ids: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the latent decoder stack.

        Args:
            z: (B, L, D) latent embeddings.
            pos_ids: (B, L) position ids.
            span_ids: (B, L) number of base tokens per merged token.
            key_padding_mask: (B, L) bool — ``True`` = real token.

        Returns:
            z_hat: (B, L, D) reconstructed latent embeddings.
        """
        for block in self.blocks:
            z = block(z, self.rope_freqs, pos_ids, span_ids, key_padding_mask=key_padding_mask)
        return self.norm(z)
