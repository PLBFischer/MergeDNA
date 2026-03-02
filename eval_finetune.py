#!/usr/bin/env python
"""
Fine-tune MergeDNA on genomic classification benchmarks with LoRA + MLP head.

Loads a checkpoint, freezes all pretrained weights, injects low-rank adapters
(LoRA) into every attention projection (wq, wk, wv, wo) of both the local
encoder and the latent encoder, attaches a trainable MLP classification head
on top of mean-pooled latent representations, and trains with AdamW.

Usage:
    python eval_finetune.py --output_dir <dir> [--benchmark <name>] [--epochs 10]

Example:
    python eval_finetune.py --output_dir outputs/mini_c9f625b31218ea2df7bea75a1412a743
    python eval_finetune.py --output_dir outputs/mini_c9f625b31218ea2df7bea75a1412a743 --benchmark human_enhancers_cohn --epochs 20
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_encoder.default import LocalEncoder
from latent_encoder.default import LatentEncoder
from utils.dna import NUCLEOTIDE_MAP, PAD_ID


# ──────────────────────────── LoRA ───────────────────────────────────────────


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a frozen base weight plus a
    trainable low-rank adapter:  output = W_frozen @ x + (B @ A @ x) * scale."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        d_in = original.in_features
        d_out = original.out_features
        device = original.weight.device
        dtype = original.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(d_in, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Inject LoRA adapters into every attention projection (wq/wk/wv/wo).

    Returns the number of adapters created.
    """
    count = 0
    for module in model.modules():
        if not hasattr(module, "attn"):
            continue
        attn = module.attn
        for proj_name in ("wq", "wk", "wv", "wo"):
            orig = getattr(attn, proj_name, None)
            if isinstance(orig, nn.Linear):
                setattr(attn, proj_name, LoRALinear(orig, rank, alpha))
                count += 1
    return count


# ──────────────────────── MLP classification head ────────────────────────────


class MLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────── local benchmark dataset ────────────────────────────

_REPO_ROOT = Path(__file__).parent
_LOCAL_DATASETS_DIR = _REPO_ROOT / "genomic_benchmarks" / "datasets"
_LOCAL_FASTA = _REPO_ROOT / "hg38.fa"

_BENCHMARK_FASTA: dict[str, str] = {
    "human_nontata_promoters":       "hg38.fa",
    "human_enhancers_cohn":          "hg38.fa",
    "human_enhancers_ensembl":       "hg38.fa",
    "human_ensembl_regulatory":      "hg38.fa",
    "human_ocr_ensembl":             "hg38.fa",
    "demo_human_or_worm":            None,
    "demo_coding_vs_intergenomic_seqs": None,
    "dummy_mouse_enhancers_ensembl": "mm10.fa",
    "drosophila_enhancers_stark":    "dm6.fa",
}

_HG38_BENCHMARKS = [k for k, v in _BENCHMARK_FASTA.items() if v == "hg38.fa"]

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


class LocalGenomicDataset(Dataset):
    """Reads from local genomic_benchmarks CSV.gz files and fetches sequences
    on-the-fly from a local FASTA via pyfaidx."""

    def __init__(
        self,
        benchmark_name: str,
        split: str,
        datasets_dir: Path = _LOCAL_DATASETS_DIR,
        fasta_path: Path = _LOCAL_FASTA,
    ):
        split_dir = datasets_dir / benchmark_name / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Local benchmark split not found: {split_dir}\n"
                f"Available benchmarks: {[d.name for d in datasets_dir.iterdir() if d.is_dir()]}"
            )

        print(f"  Building {benchmark_name}/{split} from {split_dir} …", flush=True)
        print(f"  Loading FASTA index for {fasta_path} (first run builds .fai) …", flush=True)
        fasta = pyfaidx.Fasta(str(fasta_path), sequence_always_upper=True)

        self.sequences: list[str] = []
        self.labels: list[int] = []

        csv_files = sorted(split_dir.glob("*.csv.gz"))
        if not csv_files:
            raise FileNotFoundError(f"No *.csv.gz files found in {split_dir}")

        for label_idx, csv_path in enumerate(csv_files):
            label_name = csv_path.name.replace(".csv.gz", "")
            df = pd.read_csv(csv_path)
            print(f"    {label_name}: {len(df)} sequences", flush=True)
            for _, row in df.iterrows():
                chrom = row["region"]
                start, end = int(row["start"]), int(row["end"])
                strand = row["strand"]
                try:
                    seq = str(fasta[chrom][start:end])
                except (KeyError, pyfaidx.FetchError):
                    seq = "N" * (end - start)
                if strand == "-":
                    seq = _revcomp(seq)
                self.sequences.append(seq)
                self.labels.append(label_idx)

        fasta.close()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def list_local_benchmarks(datasets_dir: Path = _LOCAL_DATASETS_DIR) -> list[str]:
    return sorted(d.name for d in datasets_dir.iterdir() if d.is_dir())


# ──────────────────────────── checkpoint helpers ─────────────────────────────


def find_latest_checkpoint(output_dir: str) -> str | None:
    final = os.path.join(output_dir, "checkpoint_final.pt")
    if os.path.isfile(final):
        return final

    best_epoch, best_path = 0, None
    for fname in os.listdir(output_dir):
        if fname.startswith("checkpoint_epoch_") and fname.endswith(".pt"):
            try:
                e = int(fname.removeprefix("checkpoint_epoch_").removesuffix(".pt"))
            except ValueError:
                continue
            if e > best_epoch:
                best_epoch, best_path = e, os.path.join(output_dir, fname)
    return best_path


# ────────────────────────── config / model loading ───────────────────────────


def load_resolved_config(output_dir: str) -> dict:
    cfg = OmegaConf.load(os.path.join(output_dir, "config.yaml"))
    return OmegaConf.to_container(cfg, resolve=True)


def build_models(cfg: dict, device: torch.device):
    exp = cfg["experiment"]
    le_cfg = cfg["local_encoder"]
    la_cfg = cfg["latent_encoder"]

    local_encoder = LocalEncoder(
        vocab_size=cfg["vocab_size"],
        pad_token_id=cfg["pad_token_id"],
        embed_dim=exp["embed_dim"],
        num_heads=exp["num_heads"],
        ffn_dim=exp["ffn_dim"],
        num_layers=le_cfg["num_layers"],
        local_window_size=le_cfg.get("local_window_size", exp.get("local_window_size", 16)),
        merge_group_dim=le_cfg.get("merge_group_dim", 64),
        max_seq_len=exp["max_seq_len"],
        compression_ratio_mean=le_cfg.get("compression_ratio_mean", 0.5),
        compression_ratio_min=le_cfg.get("compression_ratio_min", 0.4),
        compression_ratio_max=le_cfg.get("compression_ratio_max", 0.6),
    ).to(device)

    latent_encoder = LatentEncoder(
        embed_dim=exp["embed_dim"],
        num_heads=exp["num_heads"],
        ffn_dim=exp["ffn_dim"],
        num_layers=la_cfg["num_layers"],
        merge_group_dim=la_cfg.get("merge_group_dim", 64),
        max_seq_len=exp["max_seq_len"],
    ).to(device)

    return local_encoder, latent_encoder


# ────────────────────────── tokenisation helpers ─────────────────────────────


def encode_dna_string(seq: str, max_seq_len: int) -> torch.Tensor:
    ids = [NUCLEOTIDE_MAP.get(c, NUCLEOTIDE_MAP["N"]) for c in seq[:max_seq_len]]
    if len(ids) < max_seq_len:
        ids += [PAD_ID] * (max_seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def _collate(batch, max_seq_len):
    seqs, labels = zip(*batch)
    input_ids = torch.stack([encode_dna_string(s, max_seq_len) for s in seqs])
    return input_ids, torch.tensor(labels, dtype=torch.long)


# ────────────────────── forward pass through backbone ────────────────────────


def forward_embed(local_encoder, latent_encoder, input_ids, device):
    """Run input_ids through both encoders and return mean-pooled embeddings."""
    input_ids = input_ids.to(device)

    z_l, source, pos_ids, span_ids, _, seq_pad_mask = local_encoder(input_ids)
    z_prime = latent_encoder(z_l, pos_ids, span_ids, key_padding_mask=seq_pad_mask)

    pad_mask = (input_ids != PAD_ID).float()                         # (B, N)
    content = torch.bmm(source, pad_mask.unsqueeze(-1)).squeeze(-1)  # (B, L)
    token_mask = (content > 1e-6).float().unsqueeze(-1)              # (B, L, 1)

    masked = z_prime.float() * token_mask
    emb = masked.sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)  # (B, D)
    return emb


# ──────────────────────── training utilities ─────────────────────────────────


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


# ──────────────────────────── per-benchmark runner ───────────────────────────


def run_benchmark(
    benchmark_name: str,
    fasta_path: Path,
    cfg: dict,
    ckpt: dict,
    device: torch.device,
    args,
) -> dict:
    """Fine-tune from scratch on a single benchmark.  Returns a results dict."""
    use_amp = device.type == "cuda"
    max_seq_len = cfg["experiment"]["max_seq_len"]
    embed_dim = cfg["experiment"]["embed_dim"]

    # ── Fresh models from checkpoint ─────────────────────────────────────
    local_encoder, latent_encoder = build_models(cfg, device)
    local_encoder.load_state_dict(ckpt["local_encoder"])
    latent_encoder.load_state_dict(ckpt["latent_encoder"])

    for p in local_encoder.parameters():
        p.requires_grad_(False)
    for p in latent_encoder.parameters():
        p.requires_grad_(False)

    apply_lora(local_encoder, rank=args.lora_rank, alpha=args.lora_alpha)
    apply_lora(latent_encoder, rank=args.lora_rank, alpha=args.lora_alpha)

    # ── Data ─────────────────────────────────────────────────────────────
    train_dset = LocalGenomicDataset(benchmark_name, "train", fasta_path=fasta_path)
    test_dset = LocalGenomicDataset(benchmark_name, "test", fasta_path=fasta_path)
    n_classes = len(set(train_dset.labels))
    print(f"  train={len(train_dset)}  test={len(test_dset)}  classes={n_classes}")

    collate = lambda batch: _collate(batch, max_seq_len)
    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=device.type == "cuda",
    )

    # ── MLP head ─────────────────────────────────────────────────────────
    mlp_head = MLPHead(
        embed_dim=embed_dim,
        num_classes=n_classes,
        hidden_dim=args.mlp_hidden,
        dropout=args.mlp_dropout,
    ).to(device)

    # ── Parameter summary ────────────────────────────────────────────────
    all_modules = [local_encoder, latent_encoder, mlp_head]
    total_p = sum(count_params(m)[0] for m in all_modules)
    trainable_p = sum(count_params(m)[1] for m in all_modules)
    print(f"  Parameters: {total_p:,} total, {trainable_p:,} trainable "
          f"({100 * trainable_p / total_p:.2f}%)")

    # ── Optimizer & scheduler ────────────────────────────────────────────
    trainable_params = [p for m in all_modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup = min(args.warmup_steps, total_steps // 5)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop ────────────────────────────────────────────────────
    print(f"  Training: {args.epochs} epochs, {len(train_loader)} steps/epoch, "
          f"{total_steps} total steps")
    print(f"  batch_size={args.batch_size}  lr={args.lr}  warmup={warmup}  "
          f"weight_decay={args.weight_decay}")

    local_encoder.train()
    latent_encoder.train()
    global_step = 0
    train_t0 = time.time()
    best_test_acc = 0.0
    best_test_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step, (input_ids, labels) in enumerate(train_loader, 1):
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                emb = forward_embed(local_encoder, latent_encoder, input_ids, device)
                logits = mlp_head(emb)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            bs = labels.size(0)
            epoch_loss += loss.item() * bs
            epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
            epoch_total += bs

            if step % args.log_every == 0 or step == len(train_loader):
                elapsed = time.time() - train_t0
                steps_left = total_steps - global_step
                eta = elapsed / max(1, global_step) * steps_left
                running_acc = epoch_correct / epoch_total
                running_loss = epoch_loss / epoch_total
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"    [{epoch}/{args.epochs}] step {step}/{len(train_loader)}  "
                    f"loss={running_loss:.4f}  acc={running_acc:.4f}  "
                    f"lr={lr_now:.2e}  "
                    f"elapsed={fmt_time(elapsed)}  eta={fmt_time(eta)}",
                    flush=True,
                )

        epoch_time = time.time() - epoch_t0
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        # ── Evaluate on test set ─────────────────────────────────────────
        local_encoder.eval()
        latent_encoder.eval()
        mlp_head.eval()

        test_correct = 0
        test_total = 0
        test_loss_sum = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, labels in test_loader:
                labels = labels.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    emb = forward_embed(local_encoder, latent_encoder, input_ids, device)
                    logits = mlp_head(emb)
                    loss = F.cross_entropy(logits, labels)

                preds = logits.argmax(dim=-1)
                bs = labels.size(0)
                test_loss_sum += loss.item() * bs
                test_correct += (preds == labels).sum().item()
                test_total += bs
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        test_acc = test_correct / test_total
        test_loss = test_loss_sum / test_total
        all_preds_np = np.concatenate(all_preds)
        all_labels_np = np.concatenate(all_labels)
        test_f1 = f1_score(all_labels_np, all_preds_np, average="macro")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
        marker = " *best*" if test_acc >= best_test_acc else ""

        print(
            f"    Epoch {epoch} done in {fmt_time(epoch_time)}  |  "
            f"train loss={train_loss:.4f} acc={train_acc:.4f}  |  "
            f"test loss={test_loss:.4f} acc={test_acc:.4f} F1={test_f1:.4f}{marker}",
            flush=True,
        )

        local_encoder.train()
        latent_encoder.train()
        mlp_head.train()

    # ── Final evaluation ─────────────────────────────────────────────────
    total_time = time.time() - train_t0

    local_encoder.eval()
    latent_encoder.eval()
    mlp_head.eval()

    all_preds_test, all_labels_test = [], []
    with torch.no_grad():
        for input_ids, labels in test_loader:
            labels = labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                emb = forward_embed(local_encoder, latent_encoder, input_ids, device)
                logits = mlp_head(emb)
            all_preds_test.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels_test.append(labels.cpu().numpy())

    y_test = np.concatenate(all_labels_test)
    y_pred_test = np.concatenate(all_preds_test)

    final_acc = accuracy_score(y_test, y_pred_test)
    final_f1 = f1_score(y_test, y_pred_test, average="macro")

    print(f"\n  {benchmark_name} — final results:")
    print(f"    Test accuracy  : {final_acc:.4f}")
    print(f"    Test F1 (macro): {final_f1:.4f}")
    print(f"    Best test acc  : {best_test_acc:.4f}  (best F1: {best_test_f1:.4f})")
    print(f"    Training time  : {fmt_time(total_time)}")
    print(f"\n{classification_report(y_test, y_pred_test)}")

    # Free GPU memory before next benchmark
    del local_encoder, latent_encoder, mlp_head, optimizer, scaler
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return {
        "benchmark": benchmark_name,
        "n_classes": n_classes,
        "train_size": len(train_dset),
        "test_size": len(test_dset),
        "best_test_acc": best_test_acc,
        "best_test_f1": best_test_f1,
        "final_test_acc": final_acc,
        "final_test_f1": final_f1,
        "time": total_time,
    }


# ──────────────────────────────── main ───────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MergeDNA with LoRA + MLP head on genomic benchmarks",
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to training output folder (checkpoints + config.yaml)")
    parser.add_argument("--benchmark", type=str, default="human_nontata_promoters",
                        help="Benchmark name, or 'all_hg38' to run all hg38 benchmarks "
                             "(default: human_nontata_promoters)")
    parser.add_argument("--fasta", type=str, default=None,
                        help="Path to reference FASTA (auto-selected if omitted)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--mlp_hidden", type=int, default=512)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print training loss every N steps")
    parser.add_argument("--list_benchmarks", action="store_true",
                        help="Print available benchmarks and exit")
    args = parser.parse_args()

    if args.list_benchmarks:
        print("Available local genomic benchmarks:")
        for name in list_local_benchmarks():
            default_fa = _BENCHMARK_FASTA.get(name, "hg38.fa")
            supported = "Y" if default_fa else "N (unsupported)"
            print(f"  - {name}  [{default_fa or 'unsupported'}]  {supported}")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    print(f"Device: {device}  (AMP: {use_amp})")

    # ── Config & checkpoint (loaded once, reused across benchmarks) ───────
    cfg = load_resolved_config(args.output_dir)
    max_seq_len = cfg["experiment"]["max_seq_len"]
    embed_dim = cfg["experiment"]["embed_dim"]
    print(f"Model: embed_dim={embed_dim}, max_seq_len={max_seq_len}")

    ckpt_path = find_latest_checkpoint(args.output_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {args.output_dir}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  epoch={ckpt.get('epoch', '?')}  step={ckpt.get('global_step', '?')}")
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")

    # ── Resolve benchmark list ───────────────────────────────────────────
    if args.benchmark == "all_hg38":
        benchmarks = _HG38_BENCHMARKS
        print(f"\nRunning all hg38 benchmarks: {benchmarks}")
    else:
        benchmarks = [args.benchmark]

    # ── Resolve FASTA ────────────────────────────────────────────────────
    all_results = []
    total_t0 = time.time()

    for bench_idx, benchmark_name in enumerate(benchmarks, 1):
        print(f"\n{'=' * 72}")
        print(f"  [{bench_idx}/{len(benchmarks)}] {benchmark_name}")
        print(f"{'=' * 72}")

        if args.fasta:
            fasta_path = Path(args.fasta)
        else:
            default_fa = _BENCHMARK_FASTA.get(benchmark_name)
            if default_fa is None:
                print(f"  SKIPPING: unsupported (mixed or transcript-based coords)")
                continue
            fasta_path = _REPO_ROOT / default_fa
        if not fasta_path.exists():
            print(f"  SKIPPING: FASTA not found at {fasta_path}")
            continue

        result = run_benchmark(
            benchmark_name=benchmark_name,
            fasta_path=fasta_path,
            cfg=cfg,
            ckpt=ckpt,
            device=device,
            args=args,
        )
        all_results.append(result)

    # ── Summary table ────────────────────────────────────────────────────
    if len(all_results) > 1:
        total_time = time.time() - total_t0
        print(f"\n{'=' * 72}")
        print(f"  SUMMARY  (LoRA + MLP fine-tuning, {len(all_results)} benchmarks)")
        print(f"{'=' * 72}")
        print(f"  {'Benchmark':<32s}  {'Acc':>6s}  {'F1':>6s}  {'Best Acc':>8s}  {'Time':>8s}")
        print(f"  {'-' * 32}  {'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 8}")
        for r in all_results:
            print(f"  {r['benchmark']:<32s}  {r['final_test_acc']:6.4f}  {r['final_test_f1']:6.4f}"
                  f"  {r['best_test_acc']:8.4f}  {fmt_time(r['time']):>8s}")
        mean_acc = np.mean([r["final_test_acc"] for r in all_results])
        mean_f1 = np.mean([r["final_test_f1"] for r in all_results])
        print(f"  {'-' * 32}  {'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 8}")
        print(f"  {'MEAN':<32s}  {mean_acc:6.4f}  {mean_f1:6.4f}")
        print(f"\n  Total wall time: {fmt_time(total_time)}")


if __name__ == "__main__":
    main()
