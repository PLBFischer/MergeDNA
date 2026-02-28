"""Tests for loss functions."""

import pytest
import torch

from merge_dna.losses import (
    adaptive_masked_token_modeling_loss,
    compute_pretrain_loss,
    merged_token_reconstruction_loss,
)
from merge_dna.config import MergeDNAConfig
from merge_dna.model import MergeDNA, MergeDNAOutput


class TestMTRLoss:
    def test_perfect_prediction_zero_loss(self, device):
        V = 6
        targets = torch.tensor([[1, 2, 3, 4]], device=device)
        logits = torch.zeros(1, 4, V, device=device)
        logits[0, 0, 1] = 100.0
        logits[0, 1, 2] = 100.0
        logits[0, 2, 3] = 100.0
        logits[0, 3, 4] = 100.0
        loss = merged_token_reconstruction_loss(logits, targets, pad_id=0)
        assert loss.item() < 0.01

    def test_ignores_pad(self, device):
        V = 6
        targets = torch.tensor([[1, 0, 3, 0]], device=device)
        logits = torch.randn(1, 4, V, device=device)
        # With pad_id=0, positions 1 and 3 should be ignored
        loss = merged_token_reconstruction_loss(logits, targets, pad_id=0)
        assert loss.isfinite()

    def test_random_loss_positive(self, device):
        logits = torch.randn(2, 16, 6, device=device)
        targets = torch.randint(1, 6, (2, 16), device=device)
        loss = merged_token_reconstruction_loss(logits, targets, pad_id=0)
        assert loss.item() > 0


class TestAMTMLoss:
    def test_mask_selects_positions(self, device):
        V = 6
        targets = torch.tensor([[1, 2, 3, 4]], device=device)
        logits = torch.zeros(1, 4, V, device=device)
        logits[0, 0, 1] = 100.0  # correct
        logits[0, 1, 2] = 100.0  # correct
        logits[0, 2, 1] = 100.0  # wrong (should be 3)
        logits[0, 3, 4] = 100.0  # correct

        mask_all = torch.ones(1, 4, dtype=torch.bool, device=device)
        loss_all = adaptive_masked_token_modeling_loss(
            logits, targets, mask_all, pad_id=0,
        )

        mask_correct = torch.tensor([[True, True, False, True]], device=device)
        loss_correct = adaptive_masked_token_modeling_loss(
            logits, targets, mask_correct, pad_id=0,
        )
        # Loss on only correct positions should be lower
        assert loss_correct.item() < loss_all.item()

    def test_empty_mask_returns_zero(self, device):
        logits = torch.randn(1, 4, 6, device=device)
        targets = torch.randint(1, 6, (1, 4), device=device)
        mask = torch.zeros(1, 4, dtype=torch.bool, device=device)
        loss = adaptive_masked_token_modeling_loss(logits, targets, mask)
        assert loss.item() == 0.0


class TestCombinedLoss:
    def test_combined_loss_structure(self, small_config, device):
        model = MergeDNA(small_config).to(device).train()
        ids = torch.randint(1, 6, (2, 64), device=device)
        output = model(ids, mode="pretrain")
        loss_dict = compute_pretrain_loss(output, ids, small_config)

        assert "loss" in loss_dict
        assert "loss_mtr" in loss_dict
        assert "loss_latent_mtr" in loss_dict
        assert "loss_amtm" in loss_dict
        assert loss_dict["loss"].isfinite()
        assert loss_dict["loss"].requires_grad

    def test_lambda_weighting(self, small_config, device):
        """Verify lambda=0.25 weighting: total = mtr + 0.25*latent_mtr + amtm."""
        model = MergeDNA(small_config).to(device).train()
        ids = torch.randint(1, 6, (2, 64), device=device)
        output = model(ids, mode="pretrain")
        loss_dict = compute_pretrain_loss(output, ids, small_config)

        expected_total = (
            loss_dict["loss_mtr"] +
            small_config.lambda_latent_mtr * loss_dict["loss_latent_mtr"] +
            loss_dict["loss_amtm"]
        )
        assert torch.allclose(
            loss_dict["loss"].detach(), expected_total, atol=1e-4
        )
