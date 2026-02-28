"""
MergeDNA model configuration.

Hyperparameters sourced from the paper (Table A1) and supplementary details.
Where the paper is underspecified, independent design decisions are documented.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MergeDNAConfig:
    # --- Vocabulary ---
    # DESIGN DECISION: Paper defines alphabet {A,T,C,G}. We add PAD (index 0)
    # and N (index 5) for padding and ambiguous/unknown bases.
    vocab_size: int = 6  # PAD=0, A=1, T=2, C=3, G=4, N=5
    pad_token_id: int = 0

    # --- Dimensions (from paper) ---
    embed_dim: int = 1024
    # DESIGN DECISION: Not stated in paper. Inferred from LLaMA-1 convention
    # at D=1024 (16 heads x 64 head_dim).
    num_heads: int = 16
    head_dim: int = 64
    # DESIGN DECISION: Not stated in paper. Reverse-engineered from the
    # 253M parameter count for the 20-layer Latent Encoder using the SwiGLU
    # per-block formula: 4*D^2 (attn) + 3*D*ffn_dim (SwiGLU) + 2*D (norms).
    # 2752 yields ~12.65M per block x 20 = ~253M.
    ffn_dim: int = 2752

    # --- Architecture depths (from paper Table A1) ---
    local_encoder_layers: int = 4
    latent_encoder_layers: int = 20
    latent_decoder_layers: int = 4
    local_decoder_layers: int = 2

    # --- Token merging (from paper) ---
    local_window_size: int = 16
    # DESIGN DECISION: Paper says r_l pairs merged per window per layer.
    # With 4 layers and window=16, r_l=2 gives 16->14->12->10->8 = 50%.
    merges_per_window_per_layer: int = 2
    target_local_compression: float = 0.5   # L = N * ratio
    target_latent_compression: float = 0.5  # K = L * ratio

    # DESIGN DECISION: Grouping embedding dimension for DTEM-style merge
    # scoring. Paper says "lightweight grouping embedding" but doesn't specify
    # the dimension. We use head_dim (64) as a reasonable small projection.
    merge_group_dim: int = 64

    # --- Sequence ---
    max_seq_len: int = 4096

    # --- Regularization ---
    # DESIGN DECISION: Dropout not mentioned in paper. Included as
    # configurable, defaulting to 0.0.
    dropout: float = 0.0

    # --- Pre-training (from paper Table A1) ---
    lambda_latent_mtr: float = 0.25

    # --- Training (from paper Table A1) ---
    learning_rate: float = 1e-4
    weight_decay: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0
    total_iterations: int = 100_000
    warmup_iterations: int = 10_000
    batch_size: int = 256
    per_gpu_batch_size: int = 8
    gradient_accumulation_steps: int = 16

    # --- Compression ratio sampling (from paper Sec 3.3) ---
    compression_ratio_mean: float = 0.5
    compression_ratio_min: float = 0.4
    compression_ratio_max: float = 0.6

    # --- Initialization ---
    # DESIGN DECISION: Not specified in paper. Standard Transformer init.
    init_std: float = 0.02

    @property
    def num_kv_heads(self) -> int:
        return self.num_heads  # no GQA; paper doesn't mention it
