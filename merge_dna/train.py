"""
MergeDNA pre-training script.

Follows Table A1 from the paper:
  - AdamW, betas=(0.9, 0.95), weight_decay=1e-8
  - Base lr=1e-4, cosine annealing with 10K-step warmup
  - 100K total iterations, global batch 256
  - Gradient clipping 1.0
  - 8 GPUs via DDP, per-GPU batch 8, gradient accumulation 16

Usage (single-node, 8 GPUs):
    torchrun --nproc_per_node=8 -m merge_dna.train \
        --data_dir /path/to/multi_species_genomes \
        --output_dir /path/to/checkpoints
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import MergeDNAConfig
from .data import StreamingFASTADataset
from .losses import compute_pretrain_loss
from .model import MergeDNA


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine annealing LR scheduler with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_distributed():
    """Initialise distributed training from torchrun environment variables."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="MergeDNA pre-training")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing FASTA files for pre-training",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--log_interval", type=int, default=50,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_interval", type=int, default=5000,
        help="Save checkpoint every N steps",
    )
    args = parser.parse_args()

    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0

    config = MergeDNAConfig()
    if is_main:
        print(f"World size: {world_size}")
        print(f"Config: {config}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Dataset & DataLoader ---
    dataset = StreamingFASTADataset(
        fasta_dir=args.data_dir,
        max_seq_len=config.max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.per_gpu_batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # --- Model ---
    model = MergeDNA(config).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_iterations, config.total_iterations,
    )

    # --- Resume ---
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=f"cuda:{local_rank}")
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        if is_main:
            print(f"Resumed from step {start_step}")

    # --- Training loop ---
    model.train()
    accum = config.gradient_accumulation_steps
    global_step = start_step
    optimizer.zero_grad()
    data_iter = iter(dataloader)
    running_loss = 0.0
    t0 = time.time()

    while global_step < config.total_iterations:
        for micro_step in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch.cuda(local_rank, non_blocking=True)
            output = model(input_ids, mode="pretrain")
            loss_dict = compute_pretrain_loss(output, input_ids, config)
            loss = loss_dict["loss"] / accum
            loss.backward()
            running_loss += loss_dict["loss"].item() / accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        if is_main and global_step % args.log_interval == 0:
            elapsed = time.time() - t0
            throughput = args.log_interval / elapsed
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {global_step:>6d} | "
                f"loss {running_loss / args.log_interval:.4f} | "
                f"lr {lr:.2e} | "
                f"{throughput:.2f} steps/s"
            )
            running_loss = 0.0
            t0 = time.time()

        if is_main and global_step % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_{global_step}.pt"
            )
            torch.save(
                {
                    "step": global_step,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    if is_main:
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        torch.save(
            {
                "step": global_step,
                "model": model.module.state_dict(),
                "config": config,
            },
            final_path,
        )
        print(f"Training complete. Final checkpoint: {final_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
