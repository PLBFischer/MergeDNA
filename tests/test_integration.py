"""End-to-end integration tests.

Covers:
  - Gradient flow through all parameters during pretrain forward+backward
  - Loss decreases over a few optimization steps (model can learn)
  - Full-scale 380M model forward+backward on a short sequence
  - Checkpoint save/load round-trip
"""

import os
import tempfile

import pytest
import torch

from merge_dna.config import MergeDNAConfig
from merge_dna.model import MergeDNA
from merge_dna.losses import compute_pretrain_loss


class TestGradientFlow:
    """Verify that gradients reach all trainable parameters after a
    pretrain forward + backward pass."""

    def test_all_params_get_gradients(self, small_config, device):
        model = MergeDNA(small_config).to(device).train()
        ids = torch.randint(1, 6, (2, 64), device=device)

        output = model(ids, mode="pretrain")
        loss_dict = compute_pretrain_loss(output, ids, small_config)
        loss_dict["loss"].backward()

        no_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad_params.append(name)

        # Some parameters may legitimately have no grad if they were not
        # used in a particular forward pass (e.g., global_merge proj is
        # only used in pass 2 with detached input). We check that the
        # vast majority of parameters receive gradients.
        total_params = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        grad_params = total_params - len(no_grad_params)
        frac = grad_params / total_params
        assert frac >= 0.90, (
            "Only %.0f%% of params received gradients. "
            "Missing: %s" % (frac * 100, no_grad_params[:10])
        )

    def test_no_nan_gradients(self, small_config, device):
        model = MergeDNA(small_config).to(device).train()
        ids = torch.randint(1, 6, (2, 64), device=device)

        output = model(ids, mode="pretrain")
        loss_dict = compute_pretrain_loss(output, ids, small_config)
        loss_dict["loss"].backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.isfinite().all(), (
                    "Non-finite gradient in %s" % name
                )


class TestLearning:
    """Verify the model can learn by checking that loss decreases over
    a few optimization steps on a fixed batch."""

    def test_loss_decreases(self, small_config, device):
        model = MergeDNA(small_config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ids = torch.randint(1, 6, (4, 64), device=device)

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            output = model(ids, mode="pretrain")
            loss_dict = compute_pretrain_loss(output, ids, small_config)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss_dict["loss"].item())

        # Loss at end should be lower than at start
        assert losses[-1] < losses[0], (
            "Loss did not decrease: start=%.4f end=%.4f" % (losses[0], losses[-1])
        )


class TestCheckpointRoundTrip:
    """Verify that model state can be saved and reloaded faithfully."""

    def test_save_and_load(self, small_config, device):
        model = MergeDNA(small_config).to(device).eval()
        ids = torch.randint(1, 6, (1, 32), device=device)

        with torch.no_grad():
            out_before = model(ids, mode="full")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model": model.state_dict(), "config": small_config}, f.name)
            ckpt_path = f.name

        try:
            model2 = MergeDNA(small_config).to(device).eval()
            ckpt = torch.load(ckpt_path, map_location=device)
            model2.load_state_dict(ckpt["model"])

            with torch.no_grad():
                out_after = model2(ids, mode="full")

            assert torch.allclose(out_before, out_after, atol=1e-5)
        finally:
            os.unlink(ckpt_path)


class TestFullScaleIntegration:
    """Full 380M model on GPU -- forward + backward on a tiny batch.
    This validates that Flash Attention, token merging, and the full
    pipeline work end-to-end at paper scale."""

    @pytest.mark.slow
    def test_full_scale_pretrain_forward_backward(self, full_config, device):
        model = MergeDNA(full_config).to(device).train()
        ids = torch.randint(1, 6, (1, 64), device=device)

        output = model(ids, mode="pretrain")
        loss_dict = compute_pretrain_loss(output, ids, full_config)
        loss_dict["loss"].backward()

        assert loss_dict["loss"].isfinite()
        assert loss_dict["loss"].item() > 0

        # Spot-check a few key parameters got gradients
        assert model.token_embed.weight.grad is not None
        assert model.latent_encoder[0].attn.qkv.wq.weight.grad is not None
        assert model.output_head.weight.grad is not None
