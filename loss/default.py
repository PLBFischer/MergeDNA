"""
Loss manager for MergeDNA pre-training.

Orchestrates the three-pass forward (Sec 3.4, Eq. 8) and combines the
three loss terms:

  L_total = L_MTR(theta) + lambda * L_MTR(theta \\ {phi}) + L_AMTM(theta)

Pass 1 (MTR):        local N→L, latent L→L (no merge), reconstruct N.
Pass 2 (Latent MTR): local encoder frozen; latent L→K via progressive
                     adjacent-pair merging spread across all layers;
                     unmerge K→L, reconstruct N.  Produces source_prime.
Pass 3 (AMTM):       mask base tokens guided by source_prime; local N→L,
                     latent L→L (no merge), reconstruct masked N.
"""

import torch

from model.token_merge import token_unmerge


class LossManager:
    def __init__(
        self,
        lambda_latent_mtr: float = 0.25,
        pad_token_id: int = 0,
    ):
        self.lambda_latent_mtr = lambda_latent_mtr
        self.pad_token_id = pad_token_id

    def _unwrap(self, module):
        """Access the underlying module when DDP-wrapped."""
        return module.module if hasattr(module, "module") else module

    def loss(self, local_encoder, latent_encoder, latent_decoder, local_decoder, batch, device):
        """Run the three-pass pre-training forward and compute combined loss.

        Args:
            local_encoder: LocalEncoder (possibly DDP-wrapped).
            latent_encoder: LatentEncoder (possibly DDP-wrapped).
            latent_decoder: LatentDecoder (possibly DDP-wrapped).
            local_decoder: LocalDecoder (possibly DDP-wrapped).
            batch: tensor of token ids from the dataloader.
            device: torch device to place the batch on.

        Returns:
            total_loss: scalar tensor (with grad).
            losses: dict of detached per-component losses for logging.
        """
        input_ids = batch.to(device)

        # ---- Pass 1: Merged Token Reconstruction (MTR, Eq. 6) ----
        z_l, source, pos_ids, span_ids, r_per_layer = local_encoder(input_ids)
        z_prime = latent_encoder(z_l, pos_ids, span_ids)
        z_hat_l = latent_decoder(z_prime, pos_ids, span_ids)
        logits_mtr = local_decoder(z_hat_l, source)
        l_mtr = self._unwrap(local_decoder).loss(
            logits_mtr, input_ids, pad_id=self.pad_token_id,
        )

        # ---- Pass 2: Latent MTR (phi frozen, Eq. 6 with detached z_l) ----
        z_l_detached = z_l.detach()
        source_detached = source.detach()

        # Progressive L→K merging with phi frozen; produces source_prime used
        # in pass 3 to compute the AMTM mask.
        z_k, source_prime, _, _, _ = self._unwrap(latent_encoder).forward_merged(
            z_l_detached, source_detached, pos_ids, span_ids,
        )

        # Unmerge K → L: source_prime (B, K, N) @ source_detached^T (B, N, L) = (B, K, L)
        overlap = torch.bmm(source_prime.float(), source_detached.float().transpose(1, 2))
        source_kl = (overlap > 0.5).float()   # (B, K, L)
        z_l_2 = token_unmerge(z_k, source_kl)  # (B, L, D)

        z_hat_l_2 = latent_decoder(z_l_2, pos_ids, span_ids)
        logits_latent_mtr = local_decoder(z_hat_l_2, source_detached)
        l_latent_mtr = self._unwrap(local_decoder).loss(
            logits_latent_mtr, input_ids, pad_id=self.pad_token_id,
        )

        # ---- Pass 3: Adaptive Masked Token Modeling (AMTM, Eq. 7) ----
        K = source_prime.shape[1]
        mask_n = self._compute_amtm_mask(source_prime, source, K)

        masked_ids = input_ids.clone()
        masked_ids[mask_n] = self.pad_token_id

        z_l_m, source_m, pos_ids_m, span_ids_m, _ = local_encoder(
            masked_ids, r_per_layer=r_per_layer,
        )
        z_prime_m = latent_encoder(z_l_m, pos_ids_m, span_ids_m)
        z_hat_l_m = latent_decoder(z_prime_m, pos_ids_m, span_ids_m)
        logits_amtm = local_decoder(z_hat_l_m, source_m)
        l_amtm = self._unwrap(local_decoder).loss(
            logits_amtm, input_ids, mask=mask_n, pad_id=self.pad_token_id,
        )

        # ---- Combined loss (Eq. 8) ----
        total = l_mtr + self.lambda_latent_mtr * l_latent_mtr + l_amtm

        losses = {
            "loss_mtr": l_mtr.detach(),
            "loss_latent_mtr": l_latent_mtr.detach(),
            "loss_amtm": l_amtm.detach(),
        }
        return total, losses

    @staticmethod
    def _compute_amtm_mask(
        source_prime: torch.Tensor,
        source: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Compute the AMTM mask via importance-weighted sampling (Sec 3.4).

        Tokens in smaller latent groups are considered more important and
        are more likely to be masked, forcing the model to reconstruct
        fine-grained regions.

        Args:
            source_prime: (B, K, N_orig) source matrix from pass 2 latent merge.
            source: (B, L, N_orig) source matrix from local encoder.
            K: number of latent tokens selected.

        Returns:
            mask_n: (B, N) boolean base-level mask.
        """
        group_sizes = source_prime.sum(dim=-1)  # (B, K)
        weights = 1.0 / (group_sizes + 1e-8)  # (B, K)

        w_per_base = torch.bmm(
            weights.unsqueeze(1) / (group_sizes.unsqueeze(1) + 1e-8),
            source_prime.float(),
        ).squeeze(1)  # (B, N_orig)

        w_per_base = w_per_base.clamp(min=0)
        total = w_per_base.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs = w_per_base / total

        n_mask = min(K, probs.shape[1])
        indices = torch.multinomial(probs, n_mask, replacement=False)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, indices, True)

        # The mask is already at base level (N_orig) since source_prime
        # maps K latent tokens -> N_orig base positions.
        if mask.shape[1] == source.shape[2]:
            return mask

        expanded = mask.unsqueeze(-1).float()
        base_mask = torch.bmm(source.transpose(1, 2).float(), expanded)
        return base_mask.squeeze(-1) > 0.5
