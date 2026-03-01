from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import apply_rope


class FullAttention(nn.Module):
    """Multi-head self-attention over the full sequence."""

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

        # F.scaled_dot_product_attention expects (B, H, S, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.num_heads * self.head_dim)
        return self.wo(out)


class LocalWindowAttention(nn.Module):
    """Sliding window (local) attention.

    Each token attends to a window of ``window_size`` tokens centered around
    it: ``window_size // 2`` tokens to the left and the remainder to the right.
    Tokens near the sequence boundaries attend to a smaller effective window;
    artificial padding tokens are never attended to.
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
        H = self.num_heads
        hd = self.head_dim
        left_pad = W // 2
        right_pad = W - left_pad - 1

        q = self.wq(x).view(B, S, H, hd)
        k = self.wk(x).view(B, S, H, hd)
        v = self.wv(x).view(B, S, H, hd)

        # Apply RoPE at absolute positions before any padding so that the
        # relative rotation between q_i and k_j correctly encodes |i - j|.
        if rope_freqs is not None:
            if position_ids is None:
                freqs = rope_freqs[:S]
            elif position_ids.dim() == 1:
                freqs = rope_freqs[position_ids]
            else:
                freqs = rope_freqs[position_ids]

            q_r = q.permute(0, 2, 1, 3)
            k_r = k.permute(0, 2, 1, 3)
            q_r = apply_rope(q_r, freqs)
            k_r = apply_rope(k_r, freqs)
            q = q_r.permute(0, 2, 1, 3)
            k = k_r.permute(0, 2, 1, 3)

        # Pad k and v along the sequence dimension to create the sliding windows.
        # After padding: (B, S + W - 1, H, hd)
        k_padded = F.pad(k, (0, 0, 0, 0, left_pad, right_pad))
        v_padded = F.pad(v, (0, 0, 0, 0, left_pad, right_pad))

        # unfold requires the sliding dimension to be last, so permute first.
        # (B, S+W-1, H, hd) -> (B, H, hd, S+W-1) -> unfold -> (B, H, hd, S, W)
        #                    -> permute -> (B, H, S, W, hd)
        k_windows = k_padded.permute(0, 2, 3, 1).unfold(-1, W, 1).permute(0, 1, 3, 4, 2)
        v_windows = v_padded.permute(0, 2, 3, 1).unfold(-1, W, 1).permute(0, 1, 3, 4, 2)

        # Flatten (B, S) into a single batch axis for sdpa.
        # q: (B, S, H, hd)      -> (B*S, H, 1, hd)
        # k/v: (B, H, S, W, hd) -> (B*S, H, W, hd)
        q_flat = q.reshape(B * S, H, hd).unsqueeze(2)
        k_flat = k_windows.permute(0, 2, 1, 3, 4).reshape(B * S, H, W, hd)
        v_flat = v_windows.permute(0, 2, 1, 3, 4).reshape(B * S, H, W, hd)

        # Boolean mask (True = attend) that excludes the zero-padded slots at
        # the two ends of the sequence.  Shape: (B*S, 1, 1, W).
        seq_idx = torch.arange(S, device=x.device)
        win_idx = torch.arange(W, device=x.device)
        real_pos = seq_idx.unsqueeze(1) + win_idx.unsqueeze(0) - left_pad  # (S, W)
        attn_mask = (real_pos >= 0) & (real_pos < S)  # (S, W)
        attn_mask = (
            attn_mask.unsqueeze(0).unsqueeze(2)  # (1, S, 1, W)
            .expand(B, -1, -1, -1)               # (B, S, 1, W)
            .reshape(B * S, 1, 1, W)
        )

        attn_out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, attn_mask=attn_mask)
        # (B*S, H, 1, hd) -> (B*S, H, hd) -> (B, S, H*hd)
        out = attn_out.squeeze(2).reshape(B, S, H * hd)
        return self.wo(out)
