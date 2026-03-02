from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from model.attention import Attention
from model.layers import RMSNorm, SpanEncoding, SwiGLUFFN
from model.token_merge import TokenMergeModule


class LocalToMeAttentionBlock(nn.Module):
    """Local-window attention + differentiable token merging.

    Each forward pass:
      1. Add span encoding (log-scale merged-token length)
      2. Pre-norm local-window self-attention
      3. Pre-norm SwiGLU FFN
      4. Token merging (reduces sequence length by *r* adjacent pairs)
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
        self.span_enc = SpanEncoding(dim)
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, window_size)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)
        self.merge = TokenMergeModule(dim, merge_group_dim)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        position_ids: torch.Tensor,
        span_ids: torch.Tensor,
        r: Union[int, torch.Tensor],
        rope_freqs: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        seq_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_pad_mask: (B, S) bool — ``True`` = real token at current
                (possibly merged) positions.  Used for attention masking and
                to zero-out SpanEncoding at padding positions.
            pad_mask: (B, N_orig) float — 1 for real bases, 0 for padding.
                Forwarded to the merge module to deprioritise padding pairs.
        """
        x = x + self.span_enc(span_ids, padding_mask=seq_pad_mask)
        x = x + self.attn(self.norm1(x), rope_freqs, position_ids, key_padding_mask=seq_pad_mask)
        x = x + self.ffn(self.norm2(x))
        x, source, position_ids, span_ids = self.merge(
            x, source, position_ids, span_ids, r, pad_mask=pad_mask,
        )
        return x, source, position_ids, span_ids


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
        self.attn = Attention(dim, num_heads, window_size)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs, position_ids, key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x
