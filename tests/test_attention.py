"""Tests for FullAttention and LocalWindowAttention."""

import pytest
import torch

from merge_dna.attention import FullAttention, LocalWindowAttention
from merge_dna.layers import precompute_rope_freqs


@pytest.fixture
def rope_freqs(device):
    return precompute_rope_freqs(64, 256).to(device)


class TestFullAttention:
    def test_output_shape(self, device, rope_freqs):
        attn = FullAttention(128, 2, 64).to(device)
        x = torch.randn(2, 32, 128, device=device)
        out = attn(x, rope_freqs)
        assert out.shape == (2, 32, 128)

    def test_no_bias(self, device):
        attn = FullAttention(128, 2, 64).to(device)
        for name, param in attn.named_parameters():
            assert "bias" not in name

    def test_with_explicit_positions(self, device, rope_freqs):
        attn = FullAttention(128, 2, 64).to(device)
        x = torch.randn(2, 16, 128, device=device)
        pos = torch.arange(16, device=device).unsqueeze(0).expand(2, -1)
        out = attn(x, rope_freqs, pos)
        assert out.shape == (2, 16, 128)

    def test_deterministic_eval(self, device, rope_freqs):
        attn = FullAttention(128, 2, 64).to(device).eval()
        x = torch.randn(2, 16, 128, device=device)
        out1 = attn(x, rope_freqs)
        out2 = attn(x, rope_freqs)
        assert torch.allclose(out1, out2, atol=1e-6)


class TestLocalWindowAttention:
    def test_output_shape_divisible(self, device, rope_freqs):
        """Sequence length is a multiple of window size."""
        attn = LocalWindowAttention(128, 2, 64, window_size=8).to(device)
        x = torch.randn(2, 32, 128, device=device)
        out = attn(x, rope_freqs)
        assert out.shape == (2, 32, 128)

    def test_output_shape_non_divisible(self, device, rope_freqs):
        """Sequence length is NOT a multiple of window size (tests padding)."""
        attn = LocalWindowAttention(128, 2, 64, window_size=8).to(device)
        x = torch.randn(2, 30, 128, device=device)
        out = attn(x, rope_freqs)
        assert out.shape == (2, 30, 128)

    def test_window_locality(self, device, rope_freqs):
        """Changing tokens in one window should not affect another window's
        output (verifying windows are truly independent)."""
        attn = LocalWindowAttention(128, 2, 64, window_size=8).to(device).eval()
        x = torch.randn(1, 16, 128, device=device)
        out1 = attn(x, rope_freqs)

        x_modified = x.clone()
        x_modified[0, 8:, :] += 10.0  # perturb second window
        out2 = attn(x_modified, rope_freqs)

        # First window should be unchanged
        assert torch.allclose(out1[0, :8], out2[0, :8], atol=1e-5)
        # Second window should differ
        assert not torch.allclose(out1[0, 8:], out2[0, 8:], atol=1e-3)
