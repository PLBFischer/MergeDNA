"""
Core building-block layers for MergeDNA: RMSNorm, Rotary Positional
Embeddings (RoPE), and SwiGLU feed-forward network.

All follow the LLaMA-style conventions referenced in the paper
("Following the Transformer architecture as LLaMA").
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm  (LLaMA-style, no learnable bias)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------
def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Return complex-valued frequency tensor of shape (max_len, dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(
    x: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to *x* (B, n_heads, S, head_dim).

    *freqs* has shape (S, head_dim//2) and is gathered / sliced by the
    caller to match the sequence positions present in *x*.
    """
    # reshape x: view pairs of consecutive dims as complex numbers
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D//2)
    out = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return out.type_as(x)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network  (LLaMA-style, no bias)
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: out = W_down( SiLU(W_gate(x)) * W_up(x) ).

    DESIGN DECISION (bias=False): Following LLaMA convention — paper does
    not state this explicitly.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
