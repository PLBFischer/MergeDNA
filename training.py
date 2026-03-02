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
from typing import Optional

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
    """Epoch-based trainer for MergeDNA pre-training."""

    def __init__(
        self,
        total_epochs: int = 100,
        log_interval: int = 50,
        save_every_n_epochs: int = 5,
        grad_clip: float = 1.0,
        per_gpu_batch_size: int = 8,
        amp_dtype: Optional[str] = "bfloat16",
    ):
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.save_every_n_epochs = save_every_n_epochs
        self.grad_clip = grad_clip
        self.amp_dtype = amp_dtype
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
            local_encoder: LocalEncoder module.
            latent_encoder: LatentEncoder module.
            latent_decoder: LatentDecoder module.
            local_decoder: LocalDecoder module.
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

        # ---- AMP setup ----
        is_cuda = device.type == "cuda"
        if self.amp_dtype and is_cuda:
            torch_dtype = torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16
        else:
            torch_dtype = None

        # GradScaler is only needed for float16 (bfloat16 has full float32 range).
        scaler = (
            torch.amp.GradScaler("cuda")
            if torch_dtype == torch.float16
            else None
        )
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
            if torch_dtype is not None
            else torch.amp.autocast(device_type="cuda", enabled=False)
        )

        components = {
            "local_encoder": local_encoder,
            "latent_encoder": latent_encoder,
            "latent_decoder": latent_decoder,
            "local_decoder": local_decoder,
        }

        start_epoch, global_step = self._maybe_resume(
            output_dir, components, optimizer, scheduler, scaler,
        )

        for comp in components.values():
            comp.train()

        amp_label = str(torch_dtype).split(".")[-1] if torch_dtype else "disabled"
        self.logger.info(
            f"Starting training from epoch {start_epoch}, step {global_step} "
            f"on {device} (target {self.total_epochs} epochs) | AMP: {amp_label}"
        )

        running_loss = 0.0
        steps_in_window = 0
        t0 = time.time()

        for epoch in range(start_epoch, self.total_epochs):
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad(set_to_none=True)

                with autocast_ctx:
                    loss, losses = loss_manager.loss(
                        local_encoder, latent_encoder, latent_decoder, local_decoder,
                        batch, device,
                    )

                all_params = []
                for comp in components.values():
                    all_params.extend(comp.parameters())

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                    optimizer.step()

                scheduler.step()

                running_loss += loss.item()
                steps_in_window += 1
                global_step += 1

                if global_step % self.log_interval == 0:
                    elapsed = time.time() - t0
                    throughput = steps_in_window / elapsed
                    lr = optimizer.param_groups[0]["lr"]
                    avg = running_loss / steps_in_window
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
                    steps_in_window = 0
                    t0 = time.time()

            self.logger.info(
                f"Epoch {epoch + 1}/{self.total_epochs} complete "
                f"(global step {global_step})"
            )

            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(
                    output_dir, epoch + 1, global_step, components,
                    optimizer, scheduler, scaler,
                )

        self._save_checkpoint(
            output_dir, self.total_epochs, global_step, components,
            optimizer, scheduler, scaler, is_final=True,
        )
        self.logger.info(
            f"Training complete: {self.total_epochs} epochs, "
            f"{global_step} total steps."
        )

        stats = {"final_epoch": self.total_epochs, "final_step": global_step}
        return output_dir, stats

    def _save_checkpoint(
        self, output_dir, epoch, global_step, components,
        optimizer, scheduler, scaler=None, is_final=False,
    ):
        tag = "final" if is_final else f"epoch_{epoch}"
        path = os.path.join(output_dir, f"checkpoint_{tag}.pt")

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        for name, comp in components.items():
            state[name] = comp.state_dict()

        torch.save(state, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def _maybe_resume(self, output_dir, components, optimizer, scheduler, scaler=None):
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
            if name in ckpt:
                comp.load_state_dict(ckpt[name])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        return ckpt.get("epoch", 0), ckpt.get("global_step", 0)
