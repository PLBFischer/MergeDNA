"""Tests for token merging / unmerging modules."""

import pytest
import torch

from merge_dna.token_merge import (
    GlobalTokenMergeModule,
    TokenMergeModule,
    token_unmerge,
)


class TestTokenMergeModule:
    def test_output_lengths(self, device):
        """After merging r=2 pairs in a window of 8, length should decrease."""
        merge = TokenMergeModule(128, group_dim=32).to(device)
        B, S, D, N = 2, 16, 128, 16
        x = torch.randn(B, S, D, device=device)
        source = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        x_m, s_m, p_m = merge(x, source, pos, r=2, window_size=8)
        # 2 windows of 8, each losing 2 tokens -> 16 - 4 = 12
        assert x_m.shape == (B, 12, D)
        assert s_m.shape == (B, 12, N)
        assert p_m.shape == (B, 12)

    def test_source_matrix_row_sums(self, device):
        """Each row of the source matrix should sum to 1 (unmerged) or 2
        (merged pair), and columns should sum to exactly 1 (each original
        position assigned once)."""
        merge = TokenMergeModule(64, group_dim=16).to(device)
        B, S, D = 1, 8, 64
        x = torch.randn(B, S, D, device=device)
        source = torch.eye(S, device=device).unsqueeze(0)
        pos = torch.arange(S, device=device).unsqueeze(0)

        _, s_m, _ = merge(x, source, pos, r=2, window_size=8)
        # Column sums: every original position covered exactly once
        col_sums = s_m[0].sum(dim=0)
        assert torch.allclose(col_sums, torch.ones(S, device=device))

        # Row sums: each merged token covers 1 or 2 original positions
        row_sums = s_m[0].sum(dim=1)
        assert (row_sums >= 1).all()
        assert (row_sums <= 2).all()

    def test_r_zero_is_identity(self, device):
        """Merging with r=0 should not change anything."""
        merge = TokenMergeModule(64, group_dim=16).to(device)
        x = torch.randn(1, 8, 64, device=device)
        source = torch.eye(8, device=device).unsqueeze(0)
        pos = torch.arange(8, device=device).unsqueeze(0)

        x_m, s_m, p_m = merge(x, source, pos, r=0, window_size=8)
        assert torch.equal(x, x_m)
        assert torch.equal(source, s_m)

    def test_position_ids_monotonic(self, device):
        """Merged position ids should remain monotonically non-decreasing
        within each window (keepers preserve order)."""
        merge = TokenMergeModule(64, group_dim=16).to(device)
        x = torch.randn(1, 16, 64, device=device)
        source = torch.eye(16, device=device).unsqueeze(0)
        pos = torch.arange(16, device=device).unsqueeze(0)

        _, _, p_m = merge(x, source, pos, r=2, window_size=8)
        diffs = p_m[0, 1:] - p_m[0, :-1]
        assert (diffs >= 0).all(), "Position IDs should be non-decreasing"


class TestGlobalTokenMergeModule:
    def test_reaches_target_length(self, device):
        gmerge = GlobalTokenMergeModule(64, group_dim=16).to(device)
        B, L, D, N = 2, 32, 64, 64
        x = torch.randn(B, L, D, device=device)
        source = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        source = source[:, :L, :]

        x_m, s_m = gmerge(x, source, target_len=16)
        assert x_m.shape == (B, 16, D)
        assert s_m.shape == (B, 16, N)

    def test_source_column_sums(self, device):
        """After global merge, every original position should still be
        assigned to exactly one latent token."""
        gmerge = GlobalTokenMergeModule(64, group_dim=16).to(device)
        L, D = 16, 64
        x = torch.randn(1, L, D, device=device)
        source = torch.eye(L, device=device).unsqueeze(0)

        _, s_m = gmerge(x, source, target_len=8)
        col_sums = s_m[0].sum(dim=0)
        assert torch.allclose(col_sums, torch.ones(L, device=device))

    def test_noop_when_already_short(self, device):
        gmerge = GlobalTokenMergeModule(64, group_dim=16).to(device)
        x = torch.randn(1, 8, 64, device=device)
        source = torch.eye(8, device=device).unsqueeze(0)
        x_m, s_m = gmerge(x, source, target_len=10)
        assert torch.equal(x, x_m)


class TestTokenUnmerge:
    def test_unmerge_identity(self, device):
        """Unmerging with an identity source matrix should be a passthrough."""
        z = torch.randn(2, 8, 64, device=device)
        source = torch.eye(8, device=device).unsqueeze(0).expand(2, -1, -1)
        out = token_unmerge(z, source)
        assert torch.allclose(out, z, atol=1e-6)

    def test_unmerge_expansion(self, device):
        """Unmerging should expand from L to N using the source matrix."""
        B, L, N, D = 1, 4, 8, 32
        z = torch.randn(B, L, D, device=device)
        # Each merged token covers 2 positions
        source = torch.zeros(B, L, N, device=device)
        for i in range(L):
            source[0, i, 2 * i] = 1.0
            source[0, i, 2 * i + 1] = 1.0

        out = token_unmerge(z, source)
        assert out.shape == (B, N, D)
        # Positions 0 and 1 should both equal z[0,0]
        assert torch.allclose(out[0, 0], z[0, 0], atol=1e-6)
        assert torch.allclose(out[0, 1], z[0, 0], atol=1e-6)
