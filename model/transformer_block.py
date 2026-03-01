"""TransformerBlock -- pre-norm Transformer block with full self-attention.

Used as the backbone in the Latent Encoder and Latent Decoder.
"""

from typing import Optional

import torch
import torch.nn as nn

from model.attention import FullAttention
from model.layers import RMSNorm, SwiGLUFFN


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with full self-attention (LLaMA-style)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FullAttention(dim, num_heads, head_dim, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs, position_ids)
        x = x + self.ffn(self.norm2(x))
        return x
