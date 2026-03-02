"""TransformerBlock -- pre-norm Transformer block with full self-attention.

Used as the backbone in the Latent Encoder and Latent Decoder.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.attention import Attention
from src.utils import RMSNorm, SpanEncoding, SwiGLUFFN


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with full self-attention (LLaMA-style)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.span_enc = SpanEncoding(dim)
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        span_ids: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if span_ids is not None:
            x = x + self.span_enc(span_ids, padding_mask=key_padding_mask)
        x = x + self.attn(self.norm1(x), rope_freqs, position_ids, key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x
