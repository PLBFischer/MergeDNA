"""Tests for core building-block layers: RMSNorm, SwiGLU, RoPE."""

import math

import pytest
import torch

from merge_dna.layers import RMSNorm, SwiGLUFFN, apply_rope, precompute_rope_freqs


class TestRMSNorm:
    def test_output_shape(self, device):
        norm = RMSNorm(128).to(device)
        x = torch.randn(2, 16, 128, device=device)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_rms(self, device):
        """After RMSNorm (with identity weight), each vector should have
        RMS close to 1."""
        norm = RMSNorm(64).to(device)
        x = torch.randn(4, 32, 64, device=device)
        out = norm(x)
        rms = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.05)

    def test_learnable_weight(self, device):
        norm = RMSNorm(64).to(device)
        assert norm.weight.shape == (64,)
        assert norm.weight.requires_grad


class TestSwiGLUFFN:
    def test_output_shape(self, device):
        ffn = SwiGLUFFN(128, 256).to(device)
        x = torch.randn(2, 16, 128, device=device)
        out = ffn(x)
        assert out.shape == (2, 16, 128)

    def test_no_bias(self, device):
        ffn = SwiGLUFFN(64, 128).to(device)
        for name, param in ffn.named_parameters():
            assert "bias" not in name, f"Found bias in {name}"

    def test_parameter_count(self):
        """SwiGLU has 3 weight matrices: gate, up (both dim->hidden),
        down (hidden->dim)."""
        ffn = SwiGLUFFN(128, 256)
        n_params = sum(p.numel() for p in ffn.parameters())
        expected = 128 * 256 + 128 * 256 + 256 * 128  # gate + up + down
        assert n_params == expected


class TestRoPE:
    def test_freq_shape(self):
        freqs = precompute_rope_freqs(64, 512)
        assert freqs.shape == (512, 32)
        assert freqs.is_complex()

    def test_apply_rope_shape(self, device):
        freqs = precompute_rope_freqs(64, 128).to(device)
        x = torch.randn(2, 4, 32, 64, device=device)  # (B, H, S, D)
        out = apply_rope(x, freqs[:32])
        assert out.shape == x.shape

    def test_rope_preserves_norm(self, device):
        """RoPE is a rotation, so it should roughly preserve L2 norms."""
        freqs = precompute_rope_freqs(64, 128).to(device)
        x = torch.randn(2, 4, 32, 64, device=device)
        out = apply_rope(x, freqs[:32])
        x_norms = x.float().norm(dim=-1)
        out_norms = out.float().norm(dim=-1)
        assert torch.allclose(x_norms, out_norms, atol=1e-4)
