"""
MergeDNA Trainer.

Provides a Trainer class whose .train() method runs the full
pre-training loop (epoch-based), and a get_cosine_schedule_with_warmup
factory that can be instantiated by Hydra for the LR scheduler.
"""

import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 10_000,
    total_steps: int = 100_000,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing LR scheduler with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Epoch-based trainer for MergeDNA pre-training.

    Compatible with distributed (DDP) and single-GPU workflows.
    """

    def __init__(
        self,
        total_epochs: int = 100,
        log_interval: int = 50,
        save_every_n_epochs: int = 5,
        grad_clip: float = 1.0,
        per_gpu_batch_size: int = 8,
    ):
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.save_every_n_epochs = save_every_n_epochs
        self.grad_clip = grad_clip
        self.logger = logging.getLogger(__name__)

    def train(
        self,
        local_encoder,
        latent_encoder,
        latent_decoder,
        local_decoder,
        dataloader,
        optimizer,
        scheduler,
        loss_manager,
        output_dir: str = "./outputs",
        config=None,
        use_wandb: bool = False,
    ):
        """Run the full pre-training loop.

        Args:
            local_encoder: LocalEncoder module (possibly DDP-wrapped).
            latent_encoder: LatentEncoder module (possibly DDP-wrapped).
            latent_decoder: LatentDecoder module (possibly DDP-wrapped).
            local_decoder: LocalDecoder module (possibly DDP-wrapped).
            dataloader: DataLoader yielding batches of token-id tensors.
            optimizer: configured optimizer over all model parameters.
            scheduler: LR scheduler (stepped every gradient update).
            loss_manager: LossManager instance.
            output_dir: directory for checkpoints.
            config: optional OmegaConf DictConfig (stored in checkpoints).

        Returns:
            (output_dir, stats) tuple.
        """
        device = next(local_encoder.parameters()).device
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        components = {
            "local_encoder": local_encoder,
            "latent_encoder": latent_encoder,
            "latent_decoder": latent_decoder,
            "local_decoder": local_decoder,
        }

        start_epoch, global_step = self._maybe_resume(
            output_dir, components, optimizer, scheduler,
        )

        for comp in components.values():
            comp.train()

        self.logger.info(
            f"Starting training from epoch {start_epoch}, step {global_step} "
            f"on {device} (target {self.total_epochs} epochs)"
        )

        for epoch in range(start_epoch, self.total_epochs):
            running_loss = 0.0
            t0 = time.time()

            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad(set_to_none=True)

                loss, losses = loss_manager.loss(
                    local_encoder, latent_encoder, latent_decoder, local_decoder,
                    batch, device,
                )
                loss.backward()

                all_params = []
                for comp in components.values():
                    all_params.extend(comp.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                global_step += 1

                if global_step % self.log_interval == 0:
                    elapsed = time.time() - t0
                    steps_since_log = self.log_interval
                    throughput = steps_since_log / elapsed
                    lr = optimizer.param_groups[0]["lr"]
                    avg = running_loss / steps_since_log
                    self.logger.info(
                        f"epoch {epoch + 1}/{self.total_epochs} "
                        f"step {global_step:>6d} | loss {avg:.4f} | "
                        f"lr {lr:.2e} | {throughput:.2f} steps/s | "
                        f"mtr {losses['loss_mtr']:.4f} "
                        f"lat_mtr {losses['loss_latent_mtr']:.4f} "
                        f"amtm {losses['loss_amtm']:.4f}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/loss": avg,
                                "train/loss_mtr": losses["loss_mtr"],
                                "train/loss_latent_mtr": losses["loss_latent_mtr"],
                                "train/loss_amtm": losses["loss_amtm"],
                                "train/lr": lr,
                                "train/throughput_steps_per_sec": throughput,
                                "train/epoch": epoch + 1,
                            },
                            step=global_step,
                        )
                    running_loss = 0.0
                    t0 = time.time()

            self.logger.info(
                f"Epoch {epoch + 1}/{self.total_epochs} complete "
                f"(global step {global_step})"
            )

            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(
                    output_dir, epoch + 1, global_step, components,
                    optimizer, scheduler,
                )

        self._save_checkpoint(
            output_dir, self.total_epochs, global_step, components,
            optimizer, scheduler, is_final=True,
        )
        self.logger.info(
            f"Training complete: {self.total_epochs} epochs, "
            f"{global_step} total steps."
        )

        stats = {"final_epoch": self.total_epochs, "final_step": global_step}
        return output_dir, stats

    def _save_checkpoint(
        self, output_dir, epoch, global_step, components,
        optimizer, scheduler, is_final=False,
    ):
        tag = "final" if is_final else f"epoch_{epoch}"
        path = os.path.join(output_dir, f"checkpoint_{tag}.pt")

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        for name, comp in components.items():
            module = comp.module if hasattr(comp, "module") else comp
            state[name] = module.state_dict()

        torch.save(state, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def _maybe_resume(self, output_dir, components, optimizer, scheduler):
        """Look for the latest checkpoint in output_dir and resume.

        Returns:
            (start_epoch, global_step) tuple.
        """
        if not os.path.isdir(output_dir):
            return 0, 0

        best_epoch = 0
        best_path = None
        for fname in os.listdir(output_dir):
            if fname.startswith("checkpoint_epoch_") and fname.endswith(".pt"):
                tag = fname.replace("checkpoint_epoch_", "").replace(".pt", "")
                try:
                    e = int(tag)
                    if e > best_epoch:
                        best_epoch = e
                        best_path = os.path.join(output_dir, fname)
                except ValueError:
                    continue

        if best_path is None:
            return 0, 0

        self.logger.info(f"Resuming from {best_path}")
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)

        for name, comp in components.items():
            module = comp.module if hasattr(comp, "module") else comp
            if name in ckpt:
                module.load_state_dict(ckpt[name])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        return ckpt.get("epoch", 0), ckpt.get("global_step", 0)
