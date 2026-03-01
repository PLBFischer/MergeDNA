"""
Foundational layers for MergeDNA.

* ``RMSNorm``  -- LLaMA-style RMS normalisation.
* ``precompute_rope_freqs`` / ``apply_rope`` -- Rotary Position Embeddings.
* ``SwiGLUFFN`` -- SwiGLU feed-forward network.
* ``SpanEncoding`` -- learned log-span conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (LLaMA-style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Precompute RoPE complex-exponential frequencies.

    Returns a tensor of shape ``(max_seq_len, head_dim // 2)`` containing
    the angle for each (position, dimension-pair).
    """
    dim_pairs = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, dim_pairs).float() / dim_pairs))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (max_seq_len, dim_pairs)
    return angles


def apply_rope(
    x: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings.

    Args:
        x: (B, H, S, D)  -- query or key tensor
        angles: (S, D//2) or (B, S, D//2)

    Returns:
        Tensor of the same shape with RoPE applied.
    """
    *_, S, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    if angles.dim() == 2:
        cos = angles[:S].cos()
        sin = angles[:S].sin()
    elif angles.dim() == 3:
        cos = angles[:, :S].cos()
        sin = angles[:, :S].sin()
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        cos = angles.cos()
        sin = angles.sin()
        if cos.dim() < x1.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out


class SpanEncoding(nn.Module):
    """Encode merged-token span length as log(1 + span) projected to model dim.

    A linear projection of log(1 + span_length) is added to the residual
    stream at the start of each block that operates on merged tokens.  Using
    log-scale handles arbitrarily large spans without requiring a cap, and
    reflects the natural exponential growth of span sizes across merge layers.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(1, dim, bias=True)

    def forward(self, span_ids: torch.Tensor) -> torch.Tensor:
        """Project log span lengths into model dimension.

        Args:
            span_ids: (B, L) integer number of base tokens per merged token.

        Returns:
            (B, L, dim) span encoding to add to the residual stream.
        """
        log_span = torch.log1p(span_ids.float()).unsqueeze(-1)  # (B, L, 1)
        return self.proj(log_span)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (LLaMA-style).

    Uses three linear projections: gate, up, down.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
