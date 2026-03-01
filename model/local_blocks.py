"""Local attention blocks for MergeDNA.

* ``LocalToMeAttentionBlock`` -- local-window attention followed by
  differentiable token merging (used in Local Encoder).
* ``LocalAttentionBlock`` -- local-window attention without merging
  (used in Local Decoder).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.attention import LocalWindowAttention
from model.layers import RMSNorm, SwiGLUFFN
from model.token_merge import TokenMergeModule


class LocalToMeAttentionBlock(nn.Module):
    """Local-window attention + differentiable token merging.

    Each forward pass:
      1. Pre-norm local-window self-attention
      2. Pre-norm SwiGLU FFN
      3. Token merging (reduces sequence length by *r* pairs per window)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        window_size: int = 16,
        merge_group_dim: int = 64,
    ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = RMSNorm(dim)
        self.attn = LocalWindowAttention(dim, num_heads, window_size)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)
        self.merge = TokenMergeModule(dim, merge_group_dim)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        position_ids: torch.Tensor,
        r: int,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), rope_freqs, position_ids)
        x = x + self.ffn(self.norm2(x))
        x, source, position_ids = self.merge(
            x, source, position_ids, r, self.window_size,
        )
        return x, source, position_ids


class LocalAttentionBlock(nn.Module):
    """Local-window attention block *without* token merging.

    Used in the Local Decoder to refine base-level details after unmerging.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        window_size: int = 16,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = LocalWindowAttention(dim, num_heads, window_size)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs, position_ids)
        x = x + self.ffn(self.norm2(x))
        return x
