from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import apply_rope


class Attention(nn.Module):
    """Multi-head self-attention with optional sliding-window locality.

    When ``window_size`` is ``None`` (default) every token attends to the full
    sequence.  When a positive integer is given, each token attends only to a
    window of ``window_size`` tokens centered around it (``window_size // 2``
    to the left, the remainder to the right).  Tokens near the sequence
    boundaries attend to a smaller effective window; padding slots are masked.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_freqs: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        S: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to q and k (both in (B,S,H,hd))."""
        freqs = rope_freqs[:S] if position_ids is None else rope_freqs[position_ids]
        q = apply_rope(q.permute(0, 2, 1, 3), freqs).permute(0, 2, 1, 3)
        k = apply_rope(k.permute(0, 2, 1, 3), freqs).permute(0, 2, 1, 3)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) input embeddings.
            rope_freqs: precomputed RoPE angle table.
            position_ids: (B, S) absolute position indices.
            key_padding_mask: (B, S) bool — ``True`` = real token,
                ``False`` = padding (will be masked out of attention).
        """
        B, S, _ = x.shape
        H, hd = self.num_heads, self.head_dim

        q = self.wq(x).view(B, S, H, hd)
        k = self.wk(x).view(B, S, H, hd)
        v = self.wv(x).view(B, S, H, hd)

        if rope_freqs is not None:
            q, k = self._apply_rope(q, k, rope_freqs, position_ids, S)

        if self.window_size is None:
            # Full attention: (B, H, S, hd)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            attn_mask: Optional[torch.Tensor] = None
            if key_padding_mask is not None:
                attn_mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, S)

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out = attn_out.permute(0, 2, 1, 3).reshape(B, S, H * hd)
        else:
            W = self.window_size
            left_pad = W // 2
            right_pad = W - left_pad - 1

            # Pad k and v to create sliding windows. After: (B, S+W-1, H, hd)
            k_padded = F.pad(k, (0, 0, 0, 0, left_pad, right_pad))
            v_padded = F.pad(v, (0, 0, 0, 0, left_pad, right_pad))

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

            # Boolean mask (True = attend) excluding zero-padded boundary slots.
            seq_idx = torch.arange(S, device=x.device)
            win_idx = torch.arange(W, device=x.device)
            real_pos = seq_idx.unsqueeze(1) + win_idx.unsqueeze(0) - left_pad  # (S, W)
            attn_mask = (real_pos >= 0) & (real_pos < S)  # (S, W)

            if key_padding_mask is not None:
                # Expand per-position padding mask into window coordinates.
                # Pad with False (= masked) to match the k/v padding scheme.
                kpm_padded = F.pad(key_padding_mask, (left_pad, right_pad), value=False)  # (B, S+W-1)
                kpm_windows = kpm_padded.unfold(-1, W, 1)  # (B, S, W)
                attn_mask = attn_mask.unsqueeze(0) & kpm_windows  # (B, S, W)
                attn_mask = attn_mask.unsqueeze(2).reshape(B * S, 1, 1, W)
            else:
                attn_mask = (
                    attn_mask.unsqueeze(0).unsqueeze(2)  # (1, S, 1, W)
                    .expand(B, -1, -1, -1)               # (B, S, 1, W)
                    .reshape(B * S, 1, 1, W)
                )

            attn_out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, attn_mask=attn_mask)
            # (B*S, H, 1, hd) -> (B*S, H, hd) -> (B, S, H*hd)
            out = attn_out.squeeze(2).reshape(B, S, H * hd)

        return self.wo(out)
