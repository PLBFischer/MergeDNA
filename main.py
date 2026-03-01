"""
MergeDNA pre-training entry point (Hydra).

Usage (single-GPU):
    python main.py dataset.fasta_dir=/path/to/genomes

Usage (multi-GPU, 8 GPUs):
    torchrun --nproc_per_node=8 main.py dataset.fasta_dir=/path/to/genomes
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from utils.hash_utils import get_output_dir


def setup_distributed():
    """Initialise distributed training from torchrun environment variables."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        local_rank = setup_distributed()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0

    try:
        # ---- Dataset and DataLoader ----
        dataset = hydra.utils.instantiate(cfg.dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.per_gpu_batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # ---- Model components ----
        local_encoder = hydra.utils.instantiate(cfg.local_encoder).to(device)
        latent_encoder = hydra.utils.instantiate(cfg.latent_encoder).to(device)
        latent_decoder = hydra.utils.instantiate(cfg.latent_decoder).to(device)
        local_decoder = hydra.utils.instantiate(cfg.local_decoder).to(device)

        if distributed:
            ddp_kwargs = dict(find_unused_parameters=True)
            local_encoder = DDP(local_encoder, device_ids=[local_rank], **ddp_kwargs)
            latent_encoder = DDP(latent_encoder, device_ids=[local_rank], **ddp_kwargs)
            latent_decoder = DDP(latent_decoder, device_ids=[local_rank], **ddp_kwargs)
            local_decoder = DDP(local_decoder, device_ids=[local_rank], **ddp_kwargs)

        # ---- Optimizer and Scheduler ----
        model_parameters = (
            list(local_encoder.parameters())
            + list(latent_encoder.parameters())
            + list(latent_decoder.parameters())
            + list(local_decoder.parameters())
        )
        optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model_parameters)
        scheduler = hydra.utils.instantiate(cfg.scheduler)(optimizer=optimizer)

        # ---- Loss manager and Trainer ----
        loss_manager = hydra.utils.instantiate(cfg.loss)
        trainer = hydra.utils.instantiate(cfg.training)

        # ---- Output directory ----
        original_cwd = hydra.utils.get_original_cwd()
        base_dir = os.path.join(original_cwd, "outputs")
        output_dir = get_output_dir(cfg, base_dir=base_dir, create_dir=is_main)

        # ---- Train ----
        output_dir, stats = trainer.train(
            local_encoder=local_encoder,
            latent_encoder=latent_encoder,
            latent_decoder=latent_decoder,
            local_decoder=local_decoder,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            output_dir=output_dir,
            config=cfg,
        )

        if is_main:
            logger.info(
                f"Training complete. "
                f"Epochs: {stats['final_epoch']}, "
                f"Total steps: {stats['final_step']}"
            )

    finally:
        if distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
