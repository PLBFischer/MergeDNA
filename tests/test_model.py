"""Tests for the full MergeDNA model."""

import pytest
import torch

from merge_dna.config import MergeDNAConfig
from merge_dna.model import MergeDNA, MergeDNAOutput


class TestModelForwardSmall:

    def test_encoder_only_shape(self, small_config, device):
        model = MergeDNA(small_config).to(device).eval()
        ids = torch.randint(1, 6, (2, 64), device=device)
        out = model(ids, mode="encoder_only")
        B, L, D = out.shape
        assert B == 2
        assert D == small_config.embed_dim
        assert L <= 64
        assert L >= int(64 * small_config.compression_ratio_min)

    def test_full_shape(self, small_config, device):
        model = MergeDNA(small_config).to(device).eval()
        ids = torch.randint(1, 6, (2, 64), device=device)
        logits = model(ids, mode="full")
        assert logits.shape == (2, 64, small_config.vocab_size)

    def test_pretrain_output_fields(self, small_config, device):
        model = MergeDNA(small_config).to(device).train()
        ids = torch.randint(1, 6, (2, 64), device=device)
        out = model(ids, mode="pretrain")
        assert isinstance(out, MergeDNAOutput)
        assert out.logits_mtr is not None
        assert out.logits_latent_mtr is not None
        assert out.logits_amtm is not None
        assert out.mask_n is not None
        assert out.logits_mtr.shape == (2, 64, small_config.vocab_size)
        assert out.logits_latent_mtr.shape == (2, 64, small_config.vocab_size)
        assert out.logits_amtm.shape == (2, 64, small_config.vocab_size)
        assert out.mask_n.shape == (2, 64)
        assert out.mask_n.dtype == torch.bool

    def test_invalid_mode_raises(self, small_config, device):
        model = MergeDNA(small_config).to(device)
        ids = torch.randint(1, 6, (1, 16), device=device)
        with pytest.raises(ValueError, match="Unknown mode"):
            model(ids, mode="invalid")


class TestCompressionRatio:

    def test_eval_compression_approx_half(self, small_config, device):
        small_config.target_local_compression = 0.5
        model = MergeDNA(small_config).to(device).eval()
        N = 64
        ids = torch.randint(1, 6, (1, N), device=device)
        out = model(ids, mode="encoder_only")
        L = out.shape[1]
        expected = int(N * 0.5)
        assert abs(L - expected) <= N * 0.15

    def test_train_compression_sampled(self, small_config, device):
        small_config.compression_ratio_min = 0.4
        small_config.compression_ratio_max = 0.6
        model = MergeDNA(small_config).to(device).train()
        N = 64
        ids = torch.randint(1, 6, (1, N), device=device)
        ratios = []
        for _ in range(20):
            out = model(ids, mode="encoder_only")
            ratios.append(out.shape[1] / N)
        assert min(ratios) >= 0.25
        assert max(ratios) <= 0.75


class TestParameterCount:

    def test_total_params_380m(self, full_config):
        model = MergeDNA(full_config)
        total = sum(p.numel() for p in model.parameters())
        lower = int(380_000_000 * 0.95)
        upper = int(380_000_000 * 1.05)
        assert lower <= total <= upper, (
            "Total params %.1fM outside [361M, 399M]" % (total / 1e6)
        )

    def test_component_param_breakdown(self, full_config):
        model = MergeDNA(full_config)

        def count(module_list, norm):
            return (sum(p.numel() for p in module_list.parameters()) +
                    sum(p.numel() for p in norm.parameters()))

        local_enc = count(model.local_encoder, model.local_encoder_norm)
        latent_enc = count(model.latent_encoder, model.latent_encoder_norm)
        latent_dec = count(model.latent_decoder, model.latent_decoder_norm)
        local_dec = count(model.local_decoder, model.local_decoder_norm)

        tol = 0.10
        assert abs(local_enc - 51e6) / 51e6 < tol
        assert abs(latent_enc - 253e6) / 253e6 < tol
        assert abs(latent_dec - 51e6) / 51e6 < tol
        assert abs(local_dec - 25e6) / 25e6 < tol


class TestFullModelForward:

    @pytest.mark.slow
    def test_full_scale_encoder_only(self, full_config, device):
        model = MergeDNA(full_config).to(device).eval()
        ids = torch.randint(1, 6, (1, 128), device=device)
        with torch.no_grad():
            out = model(ids, mode="encoder_only")
        assert out.shape[0] == 1
        assert out.shape[2] == 1024

    @pytest.mark.slow
    def test_full_scale_full_forward(self, full_config, device):
        model = MergeDNA(full_config).to(device).eval()
        ids = torch.randint(1, 6, (1, 128), device=device)
        with torch.no_grad():
            logits = model(ids, mode="full")
        assert logits.shape == (1, 128, 6)
