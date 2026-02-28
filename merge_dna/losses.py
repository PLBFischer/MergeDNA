"""
Pre-training loss functions for MergeDNA (Sec 3.3 -- 3.4, Eq. 8).

Three losses computed across three forward passes:

  L_total = L_MTR(theta) + lambda * L_MTR(theta \\ {phi}) + L_AMTM(theta)

where lambda = 0.25 by default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MergeDNAConfig
from .model import MergeDNA, MergeDNAOutput


def merged_token_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Merged Token Reconstruction loss (Eq. 6).

    Cross-entropy between reconstructed sequence logits and original input,
    ignoring PAD positions.

    Args:
        logits: (B, N, vocab_size)
        targets: (B, N) ground-truth token ids
        pad_id: padding token id to ignore
    """
    B, N, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * N, V),
        targets.reshape(B * N),
        ignore_index=pad_id,
        reduction="mean",
    )
    return loss


def adaptive_masked_token_modeling_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Adaptive Masked Token Modeling loss (Eq. 7).

    Cross-entropy computed only on masked (high-importance) positions.

    Args:
        logits: (B, N, vocab_size)
        targets: (B, N) ground-truth token ids
        mask: (B, N) boolean mask (True = masked = predict)
        pad_id: padding token id to ignore
    """
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    masked_logits = logits[mask]   # (n_masked, V)
    masked_targets = targets[mask]  # (n_masked,)
    loss = F.cross_entropy(
        masked_logits,
        masked_targets,
        ignore_index=pad_id,
        reduction="mean",
    )
    return loss


def compute_pretrain_loss(
    output: MergeDNAOutput,
    targets: torch.Tensor,
    config: MergeDNAConfig,
) -> dict:
    """Compute the combined pre-training loss (Eq. 8).

    Returns a dict with individual and total losses for logging.
    """
    pad = config.pad_token_id
    lam = config.lambda_latent_mtr

    l_mtr = merged_token_reconstruction_loss(output.logits_mtr, targets, pad)
    l_latent_mtr = merged_token_reconstruction_loss(
        output.logits_latent_mtr, targets, pad,
    )
    l_amtm = adaptive_masked_token_modeling_loss(
        output.logits_amtm, targets, output.mask_n, pad,
    )

    total = l_mtr + lam * l_latent_mtr + l_amtm

    return {
        "loss": total,
        "loss_mtr": l_mtr.detach(),
        "loss_latent_mtr": l_latent_mtr.detach(),
        "loss_amtm": l_amtm.detach(),
    }
