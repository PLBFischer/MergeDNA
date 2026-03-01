"""
MergeDNA Trainer.

Provides a Trainer class whose .train() method runs the full
pre-training loop (iteration-based), and a get_cosine_schedule_with_warmup
factory that can be instantiated by Hydra for the LR scheduler.
"""

import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn


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
    """Iteration-based trainer for MergeDNA pre-training.

    Compatible with distributed (DDP) and single-GPU workflows.
    """

    def __init__(
        self,
        total_iterations: int = 100_000,
        log_interval: int = 50,
        save_interval: int = 5_000,
        grad_clip: float = 1.0,
        per_gpu_batch_size: int = 8,
    ):
        self.total_iterations = total_iterations
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.grad_clip = grad_clip
        self.per_gpu_batch_size = per_gpu_batch_size
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
    ):
        """Run the full pre-training loop.

        Args:
            local_encoder: LocalEncoder module (possibly DDP-wrapped).
            latent_encoder: LatentEncoder module (possibly DDP-wrapped).
            latent_decoder: LatentDecoder module (possibly DDP-wrapped).
            local_decoder: LocalDecoder module (possibly DDP-wrapped).
            dataloader: iterable yielding batches of token-id tensors.
            optimizer: configured optimizer over all model parameters.
            scheduler: LR scheduler (stepped every iteration).
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

        start_step = self._maybe_resume(
            output_dir, components, optimizer, scheduler,
        )

        for comp in components.values():
            comp.train()

        step = start_step
        data_iter = iter(dataloader)
        running_loss = 0.0
        t0 = time.time()

        self.logger.info(
            f"Starting training from step {step} on {device} "
            f"(target {self.total_iterations} iterations)"
        )

        while step < self.total_iterations:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

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
            step += 1

            if step % self.log_interval == 0:
                elapsed = time.time() - t0
                throughput = self.log_interval / elapsed
                lr = optimizer.param_groups[0]["lr"]
                avg = running_loss / self.log_interval
                self.logger.info(
                    f"step {step:>6d} | loss {avg:.4f} | "
                    f"lr {lr:.2e} | {throughput:.2f} steps/s | "
                    f"mtr {losses['loss_mtr']:.4f} "
                    f"lat_mtr {losses['loss_latent_mtr']:.4f} "
                    f"amtm {losses['loss_amtm']:.4f}"
                )
                running_loss = 0.0
                t0 = time.time()

            if step % self.save_interval == 0:
                self._save_checkpoint(
                    output_dir, step, components, optimizer, scheduler,
                )

        self._save_checkpoint(
            output_dir, step, components, optimizer, scheduler, is_final=True,
        )
        self.logger.info(f"Training complete at step {step}.")

        stats = {"final_step": step}
        return output_dir, stats

    def _save_checkpoint(self, output_dir, step, components, optimizer, scheduler, is_final=False):
        tag = "final" if is_final else str(step)
        path = os.path.join(output_dir, f"checkpoint_{tag}.pt")

        state = {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        for name, comp in components.items():
            module = comp.module if hasattr(comp, "module") else comp
            state[name] = module.state_dict()

        torch.save(state, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def _maybe_resume(self, output_dir, components, optimizer, scheduler):
        """Look for the latest checkpoint in output_dir and resume."""
        if not os.path.isdir(output_dir):
            return 0

        best_step = 0
        best_path = None
        for fname in os.listdir(output_dir):
            if fname.startswith("checkpoint_") and fname.endswith(".pt"):
                tag = fname.replace("checkpoint_", "").replace(".pt", "")
                if tag == "final":
                    continue
                try:
                    s = int(tag)
                    if s > best_step:
                        best_step = s
                        best_path = os.path.join(output_dir, fname)
                except ValueError:
                    continue

        if best_path is None:
            return 0

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

        return ckpt.get("step", 0)
