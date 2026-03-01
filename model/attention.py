"""
Attention implementations for MergeDNA.

* ``FullAttention``  -- standard multi-head self-attention using Flash
  Attention (flash_attn library).
* ``LocalWindowAttention`` -- non-overlapping window attention where each
  window is processed independently.  Window size is fixed at 16 as
  stated in the paper.

Both modules follow LLaMA conventions (no bias, RoPE applied externally).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func

from model.layers import apply_rope


class FullAttention(nn.Module):
    """Multi-head self-attention over the full sequence using flash_attn_func.

    flash_attn_func expects tensors in (B, S, H, D) layout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.num_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.num_heads, self.head_dim)

        if rope_freqs is not None:
            if position_ids is None:
                freqs = rope_freqs[:S]
            elif position_ids.dim() == 1:
                freqs = rope_freqs[position_ids]
            else:
                freqs = rope_freqs[position_ids]

            q_r = q.permute(0, 2, 1, 3)
            k_r = k.permute(0, 2, 1, 3)
            q_r = apply_rope(q_r, freqs if freqs.dim() == 2 else freqs)
            k_r = apply_rope(k_r, freqs if freqs.dim() == 2 else freqs)
            q = q_r.permute(0, 2, 1, 3)
            k = k_r.permute(0, 2, 1, 3)

        attn_out = flash_attn_func(q, k, v, causal=False)
        out = attn_out.reshape(B, S, self.num_heads * self.head_dim)
        return self.wo(out)


class LocalWindowAttention(nn.Module):
    """Non-overlapping window attention.

    The sequence is partitioned into windows of ``window_size`` tokens.
    Attention is computed independently within each window via Flash Attention.
    If the sequence length is not divisible by window_size, the last window
    is simply shorter (padded internally, then unpadded).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 16,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        W = self.window_size

        pad = (W - S % W) % W
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
            if position_ids is not None and position_ids.dim() == 2:
                position_ids = F.pad(position_ids, (0, pad), value=0)
        S_padded = S + pad
        n_windows = S_padded // W

        x_win = x.reshape(B * n_windows, W, D)
        q = self.wq(x_win).view(B * n_windows, W, self.num_heads, self.head_dim)
        k = self.wk(x_win).view(B * n_windows, W, self.num_heads, self.head_dim)
        v = self.wv(x_win).view(B * n_windows, W, self.num_heads, self.head_dim)

        if rope_freqs is not None:
            if position_ids is None:
                pos_flat = (
                    torch.arange(W, device=x.device)
                    .unsqueeze(0)
                    .expand(n_windows, -1)
                )
                pos_flat = pos_flat.unsqueeze(0).expand(B, -1, -1)
                pos_flat = pos_flat.reshape(B * n_windows, W)
            elif position_ids.dim() == 1:
                pos_flat = position_ids[:S_padded].reshape(n_windows, W)
                pos_flat = pos_flat.unsqueeze(0).expand(B, -1, -1)
                pos_flat = pos_flat.reshape(B * n_windows, W)
            else:
                pos_flat = position_ids[:, :S_padded].reshape(
                    B * n_windows, W
                )

            freqs = rope_freqs[pos_flat]  # (B*nw, W, hd//2)
            q_r = q.permute(0, 2, 1, 3)
            k_r = k.permute(0, 2, 1, 3)
            q_r = apply_rope(q_r, freqs)
            k_r = apply_rope(k_r, freqs)
            q = q_r.permute(0, 2, 1, 3)
            k = k_r.permute(0, 2, 1, 3)

        attn_out = flash_attn_func(q, k, v, causal=False)  # (B*n_win, W, H, hd)

        attn_out = attn_out.reshape(B, S_padded, self.num_heads * self.head_dim)
        out = self.wo(attn_out)

        if pad > 0:
            out = out[:, :S, :]
        return out
