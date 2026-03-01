"""
Latent Encoder -- global context modelling for MergeDNA.

Processes the merged local tokens through a deep stack of full-attention
TransformerBlocks and optionally performs global token selection via
the GlobalTokenMergeModule (used in pre-training pass 2).
"""

from typing import Optional, Tuple

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
        self.global_merge = TokenMergeModule(embed_dim, merge_group_dim)

    def forward(
        self,
        z: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full-attention latent encoder stack.

        Args:
            z: (B, L, D) merged token embeddings from the local encoder.
            pos_ids: (B, L) position ids.

        Returns:
            z_prime: (B, L, D) contextualised latent embeddings.
        """
        for block in self.blocks:
            z = block(z, self.rope_freqs, pos_ids)
        return self.norm(z)

    def merge(
        self,
        z_prime: torch.Tensor,
        source: torch.Tensor,
        K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Global token selection (pre-training pass 2).

        Args:
            z_prime: (B, L, D) latent embeddings.
            source: (B, L, N_orig) source matrix.
            K: target number of salient tokens.

        Returns:
            z_k: (B, K, D) selected latent tokens.
            source_prime: (B, K, N_orig) updated source matrix (S').
        """
        return self.global_merge(z_prime, source, K)
