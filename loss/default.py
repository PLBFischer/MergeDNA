"""
Loss manager for MergeDNA pre-training.

Orchestrates the three-pass forward (Sec 3.4, Eq. 8) and combines the
three loss terms:

  L_total = L_MTR(theta) + lambda * L_MTR(theta \\ {phi}) + lambda_amtm * L_AMTM(theta)

Pass 1 (MTR):        local N→L, latent L→L (no merge), reconstruct N.
Pass 2 (Latent MTR): local encoder frozen; latent L→K via progressive
                     adjacent-pair merging spread across all layers;
                     unmerge K→L, reconstruct N.  Produces source_prime.
Pass 3 (AMTM):       importance-weighted mask over L local tokens (derived
                     from the K→L source matrix source_kl); expand to base
                     level, then local N→L, latent L→L (no merge),
                     reconstruct masked positions.
"""

import torch

from src.token_merge import token_unmerge


class LossManager:
    def __init__(
        self,
        lambda_latent_mtr: float = 0.25,
        lambda_amtm: float = 1.0,
        pad_token_id: int = 0,
        mask_token_id: int = 6,
    ):
        self.lambda_latent_mtr = lambda_latent_mtr
        self.lambda_amtm = lambda_amtm
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

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

        # Base-level padding mask (True = real base, False = pad) — constant
        # across all three passes; used for attention masking and AMTM.
        base_pad_mask = input_ids != self.pad_token_id          # (B, N) bool
        pad_mask_float = base_pad_mask.float()                  # (B, N) float for merge

        # ---- Pass 1: Merged Token Reconstruction (MTR, Eq. 6) ----
        z_l, source, pos_ids, span_ids, r_per_layer, seq_pad_mask = local_encoder(input_ids)
        z_prime = latent_encoder(z_l, pos_ids, span_ids, key_padding_mask=seq_pad_mask)
        z_hat_l = latent_decoder(z_prime, pos_ids, span_ids, key_padding_mask=seq_pad_mask)
        logits_mtr = local_decoder(z_hat_l, source, base_pad_mask=base_pad_mask)
        l_mtr = local_decoder.loss(
            logits_mtr, input_ids, pad_id=self.pad_token_id,
        )

        # ---- Pass 2: Latent MTR (phi frozen, Eq. 6 with detached z_l) ----
        z_l_detached = z_l.detach()
        source_detached = source.detach()
        seq_pad_mask_det = seq_pad_mask.detach()

        # Progressive L→K merging with phi frozen; produces source_prime used
        # in pass 3 to compute the AMTM mask.
        z_k, source_prime, _, _, _ = latent_encoder.forward_merged(
            z_l_detached, source_detached, pos_ids, span_ids,
            key_padding_mask=seq_pad_mask_det,
            pad_mask=pad_mask_float,
        )

        # Unmerge K → L: source_prime (B, K, N) @ source_detached^T (B, N, L) = (B, K, L)
        overlap = torch.bmm(source_prime.float(), source_detached.float().transpose(1, 2))
        source_kl = (overlap > 0.5).float()   # (B, K, L)
        z_l_2 = token_unmerge(z_k, source_kl)  # (B, L, D)

        z_hat_l_2 = latent_decoder(z_l_2, pos_ids, span_ids, key_padding_mask=seq_pad_mask_det)
        logits_latent_mtr = local_decoder(z_hat_l_2, source_detached, base_pad_mask=base_pad_mask)
        l_latent_mtr = local_decoder.loss(
            logits_latent_mtr, input_ids, pad_id=self.pad_token_id,
        )

        # ---- Pass 3: Adaptive Masked Token Modeling (AMTM, Eq. 7) ----
        K = source_prime.shape[1]
        mask_n = self._compute_amtm_mask(source_kl, source, K, base_pad_mask, seq_pad_mask)

        masked_ids = input_ids.clone()
        masked_ids[mask_n] = self.mask_token_id

        z_l_m, source_m, pos_ids_m, span_ids_m, _, seq_pad_mask_m = local_encoder(
            masked_ids, r_per_layer=r_per_layer,
        )
        z_prime_m = latent_encoder(z_l_m, pos_ids_m, span_ids_m, key_padding_mask=seq_pad_mask_m)
        z_hat_l_m = latent_decoder(z_prime_m, pos_ids_m, span_ids_m, key_padding_mask=seq_pad_mask_m)
        logits_amtm = local_decoder(z_hat_l_m, source_m, base_pad_mask=base_pad_mask)
        l_amtm = local_decoder.loss(
            logits_amtm, input_ids, mask=mask_n, pad_id=self.pad_token_id,
        )

        # ---- Combined loss (Eq. 8) ----
        total = l_mtr + self.lambda_latent_mtr * l_latent_mtr + self.lambda_amtm * l_amtm

        losses = {
            "loss_mtr": l_mtr.detach(),
            "loss_latent_mtr": l_latent_mtr.detach(),
            "loss_amtm": l_amtm.detach(),
        }
        return total, losses

    @staticmethod
    def _compute_amtm_mask(
        source_kl: torch.Tensor,
        source: torch.Tensor,
        K: int,
        base_pad_mask: torch.Tensor,
        seq_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the AMTM mask via importance-weighted sampling (Sec 3.4).

        Operates at the local-token (L) level: samples K local tokens to
        mask, then expands to base positions via the source matrix so that
        all bases belonging to a masked merged token are masked together.

        Args:
            source_kl: (B, K, L) source matrix from pass-2 latent merging,
                mapping K latent tokens to L local tokens.
            source: (B, L, N) source matrix from the local encoder.
            K: number of local tokens to mask.
            base_pad_mask: (B, N) bool — ``True`` for real bases.
            seq_pad_mask: (B, L) bool — ``True`` for real local tokens.

        Returns:
            mask_n: (B, N) boolean base-level mask.
        """
        # gi = number of local tokens grouped into latent token i
        group_sizes = source_kl.sum(dim=-1)  # (B, K)
        weights = 1.0 / (group_sizes + 1e-8)  # (B, K)

        # PL(j) ∝ wi / gi = 1/gi² for local token j in latent group i
        w_per_local = torch.bmm(
            weights.unsqueeze(1) / (group_sizes.unsqueeze(1) + 1e-8),
            source_kl,
        ).squeeze(1)  # (B, L)

        w_per_local = w_per_local.clamp(min=0) * seq_pad_mask.float()

        total = w_per_local.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs = w_per_local / total  # (B, L)

        L = probs.shape[1]
        n_mask = min(K, L)
        indices = torch.multinomial(probs, n_mask, replacement=False)
        mask_l = torch.zeros(probs.shape[0], L, dtype=torch.bool, device=probs.device)
        mask_l.scatter_(1, indices, True)  # (B, L)

        # Expand L-level mask to N bases: MN = S^T @ ML
        mask_n = torch.bmm(
            source.transpose(1, 2).float(),
            mask_l.unsqueeze(-1).float(),
        ).squeeze(-1) > 0.5  # (B, N)

        return mask_n & base_pad_mask
