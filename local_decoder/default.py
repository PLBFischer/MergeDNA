"""
Local Decoder -- base-level detokenizer for MergeDNA.

Unmerges latent tokens back to original sequence length using the
source matrix, refines with local-window attention, and projects
to vocabulary logits.  Provides a ``loss`` method for computing
cross-entropy reconstruction loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import RMSNorm, precompute_rope_freqs
from model.local_blocks import LocalAttentionBlock
from model.token_merge import token_unmerge


class LocalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 6,
        embed_dim: int = 1024,
        num_heads: int = 16,
        ffn_dim: int = 2752,
        num_layers: int = 2,
        local_window_size: int = 16,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(embed_dim // num_heads, max_seq_len),
            persistent=False,
        )

        self.blocks = nn.ModuleList([
            LocalAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                window_size=local_window_size,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        z_hat_l: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Unmerge, refine, and project to logits.

        Args:
            z_hat_l: (B, L, D) latent-decoded merged embeddings.
            source: (B, L, N) source matrix (merged -> original positions).

        Returns:
            logits: (B, N, vocab_size) per-base vocabulary logits.
        """
        z_n = token_unmerge(z_hat_l, source)
        N = source.shape[2]
        base_pos = (
            torch.arange(N, device=z_n.device)
            .unsqueeze(0)
            .expand(z_n.shape[0], -1)
        )
        for block in self.blocks:
            z_n = block(z_n, self.rope_freqs, base_pos)
        return self.output_head(self.final_norm(z_n))

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Compute cross-entropy reconstruction loss.

        When *mask* is provided (AMTM), only masked positions contribute.
        Otherwise the full sequence is used (MTR / latent MTR).

        Args:
            logits: (B, N, vocab_size).
            targets: (B, N) ground-truth token ids.
            mask: optional (B, N) boolean mask (True = predict).
            pad_id: padding token id to ignore.

        Returns:
            Scalar loss.
        """
        B, N, V = logits.shape
        if mask is not None and mask.sum() > 0:
            return F.cross_entropy(
                logits[mask],
                targets[mask],
                ignore_index=pad_id,
                reduction="mean",
            )
        return F.cross_entropy(
            logits.reshape(B * N, V),
            targets.reshape(B * N),
            ignore_index=pad_id,
            reduction="mean",
        )
