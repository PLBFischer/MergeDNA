"""
Shared fixtures for MergeDNA tests.

Most tests use a ``small_config`` (D=128, 2 heads, 1-2 layers) so they
finish in seconds.  A separate ``full_config`` fixture uses the paper's
380M-parameter settings for integration and parameter-count tests.
"""

import pytest
import torch

from merge_dna.config import MergeDNAConfig


@pytest.fixture
def small_config():
    """Tiny model config for fast unit tests."""
    return MergeDNAConfig(
        vocab_size=6,
        embed_dim=128,
        num_heads=2,
        head_dim=64,
        ffn_dim=256,
        local_encoder_layers=2,
        latent_encoder_layers=2,
        latent_decoder_layers=2,
        local_decoder_layers=1,
        local_window_size=8,
        merges_per_window_per_layer=2,
        merge_group_dim=32,
        max_seq_len=128,
        dropout=0.0,
        init_std=0.02,
    )


@pytest.fixture
def full_config():
    """Paper-scale 380M config."""
    return MergeDNAConfig()


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def random_input_ids(small_config, device):
    """Random batch of nucleotide sequences for the small config."""
    B, N = 2, 64
    return torch.randint(1, small_config.vocab_size, (B, N), device=device)
