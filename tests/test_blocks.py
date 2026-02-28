"""Tests for TransformerBlock, LocalToMeAttentionBlock, LocalAttentionBlock."""

import pytest
import torch

from merge_dna.blocks import (
    LocalAttentionBlock,
    LocalToMeAttentionBlock,
    TransformerBlock,
)
from merge_dna.layers import precompute_rope_freqs


@pytest.fixture
def rope_freqs(device):
    return precompute_rope_freqs(64, 256).to(device)


class TestTransformerBlock:
    def test_output_shape(self, device, rope_freqs):
        block = TransformerBlock(128, 2, 64, 256).to(device)
        x = torch.randn(2, 16, 128, device=device)
        out = block(x, rope_freqs)
        assert out.shape == (2, 16, 128)

    def test_residual_connection(self, device, rope_freqs):
        """With zero-initialized attention and FFN weights, output should
        equal input (residual stream only)."""
        block = TransformerBlock(128, 2, 64, 256).to(device)
        for p in block.parameters():
            p.data.zero_()
        x = torch.randn(2, 8, 128, device=device)
        out = block(x, rope_freqs)
        assert torch.allclose(out, x, atol=1e-5)


class TestLocalToMeAttentionBlock:
    def test_output_shape_and_merge(self, device, rope_freqs):
        block = LocalToMeAttentionBlock(
            128, 2, 64, 256, window_size=8, merge_group_dim=32,
        ).to(device)
        B, S, D = 2, 16, 128
        x = torch.randn(B, S, D, device=device)
        source = torch.eye(S, device=device).unsqueeze(0).expand(B, -1, -1)
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        x_m, s_m, p_m = block(x, source, pos, r=2, rope_freqs=rope_freqs)
        expected_len = S - 2 * (S // 8)  # 2 merges per window
        assert x_m.shape == (B, expected_len, D)
        assert s_m.shape[0] == B
        assert s_m.shape[2] == S  # still maps to original positions


class TestLocalAttentionBlock:
    def test_output_shape(self, device, rope_freqs):
        block = LocalAttentionBlock(128, 2, 64, 256, window_size=8).to(device)
        x = torch.randn(2, 32, 128, device=device)
        out = block(x, rope_freqs)
        assert out.shape == (2, 32, 128)
