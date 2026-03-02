"""
MergeDNA pre-training entry point (Hydra).

Usage:
    python main.py dataset.fasta_path=/path/to/sequences.fa
"""

import logging
import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from utils.hash_utils import get_output_dir


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_wandb = False
    if os.environ.get("WANDB_API_KEY"):
        wcfg = cfg.get("wandb", {})
        wandb.init(
            project=wcfg.get("project", "mergedna") or "mergedna",
            name=wcfg.get("name", None) or None,
            entity=wcfg.get("entity", None) or None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        use_wandb = True

    try:
        # ---- Dataset and DataLoader ----
        dataset = hydra.utils.instantiate(cfg.dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.per_gpu_batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

        # ---- Model components ----
        local_encoder = hydra.utils.instantiate(cfg.local_encoder).to(device)
        latent_encoder = hydra.utils.instantiate(cfg.latent_encoder).to(device)
        latent_decoder = hydra.utils.instantiate(cfg.latent_decoder).to(device)
        local_decoder = hydra.utils.instantiate(cfg.local_decoder).to(device)

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
        output_dir = get_output_dir(cfg, base_dir=base_dir, create_dir=True)

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
            use_wandb=use_wandb,
        )

        logger.info(
            f"Training complete. "
            f"Epochs: {stats['final_epoch']}, "
            f"Total steps: {stats['final_step']}"
        )

    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
