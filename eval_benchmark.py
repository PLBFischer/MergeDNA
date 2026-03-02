#!/usr/bin/env python
"""
Evaluate MergeDNA embeddings on genomic classification benchmarks.

Loads a checkpoint from a training output directory, encodes benchmark
sequences through local_encoder + latent_encoder (no token merging in the
latent encoder), mean-pools the latent representations into fixed-size
embeddings, and trains a logistic regression probe on top.

Usage:
    python eval_benchmark.py <output_dir> [--benchmark <name>] [--batch_size 64]

Example:
    python eval_benchmark.py outputs/mini_c9f625b31218ea2df7bea75a1412a743
    python eval_benchmark.py outputs/mini_c9f625b31218ea2df7bea75a1412a743 --benchmark human_enhancers_cohn
    python eval_benchmark.py outputs/mini_c9f625b31218ea2df7bea75a1412a743 --list_benchmarks
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_encoder.default import LocalEncoder
from latent_encoder.default import LatentEncoder
from utils.dna import NUCLEOTIDE_MAP, PAD_ID

# ──────────────────────── local benchmark dataset ───────────────────────────

_REPO_ROOT = Path(__file__).parent
_LOCAL_DATASETS_DIR = _REPO_ROOT / "genomic_benchmarks" / "datasets"
_LOCAL_FASTA = _REPO_ROOT / "hg38.fa"

# Maps benchmark name → default FASTA filename (in the repo root).
# Override with --fasta if you have the file under a different name.
_BENCHMARK_FASTA: dict[str, str] = {
    "human_nontata_promoters":       "hg38.fa",
    "human_enhancers_cohn":          "hg38.fa",
    "human_enhancers_ensembl":       "hg38.fa",
    "human_ensembl_regulatory":      "hg38.fa",
    "human_ocr_ensembl":             "hg38.fa",
    "demo_human_or_worm":            None,   # mixed — pass --fasta per class; unsupported
    "demo_coding_vs_intergenomic_seqs": None,  # transcript coords; unsupported
    "dummy_mouse_enhancers_ensembl": "mm10.fa",
    "drosophila_enhancers_stark":    "dm6.fa",
}

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


class LocalGenomicDataset(Dataset):
    """
    Reads directly from the local cloned genomic_benchmarks repo (CSV.gz files)
    and fetches sequences on-the-fly from a local FASTA via pyfaidx.
    No downloads and no per-sequence cache files are ever written to disk.
    """

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
                chrom, start, end, strand = row["region"], int(row["start"]), int(row["end"]), row["strand"]
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


# ──────────────────────────── checkpoint helpers ────────────────────────────


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


# ────────────────────────── config / model loading ──────────────────────────


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


# ────────────────────────── tokenisation helpers ────────────────────────────


def encode_dna_string(seq: str, max_seq_len: int) -> torch.Tensor:
    ids = [NUCLEOTIDE_MAP.get(c, NUCLEOTIDE_MAP["N"]) for c in seq[:max_seq_len]]
    if len(ids) < max_seq_len:
        ids += [PAD_ID] * (max_seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def _collate(batch, max_seq_len):
    seqs, labels = zip(*batch)
    input_ids = torch.stack([encode_dna_string(s, max_seq_len) for s in seqs])
    return input_ids, torch.tensor(labels, dtype=torch.long)


# ─────────────────────── span-length analysis ───────────────────────────────


@torch.no_grad()
def collect_span_lengths(
    local_encoder: LocalEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Return a 1-D array of merged-token span lengths (bases per token)
    for every content token across all sequences in *dataloader*.

    span_ids[b, l] is the number of original base tokens that were merged
    into latent token l.  Tokens that cover only padding are excluded.
    """
    local_encoder.eval()
    all_spans: list[np.ndarray] = []

    for input_ids, _ in dataloader:
        input_ids = input_ids.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, source, _, span_ids, _, _ = local_encoder(input_ids)

        # Mask out tokens whose content comes entirely from padding bases
        pad_mask = (input_ids != PAD_ID).float()                          # (B, N)
        content  = torch.bmm(source, pad_mask.unsqueeze(-1)).squeeze(-1)  # (B, L)
        token_mask = content > 1e-6                                        # (B, L)

        for b in range(input_ids.size(0)):
            spans = span_ids[b][token_mask[b]].cpu().numpy()  # (n_content_tokens,)
            all_spans.append(spans)

    return np.concatenate(all_spans)


# ─────────────────────── embedding extraction ───────────────────────────────


@torch.no_grad()
def extract_embeddings(
    local_encoder: LocalEncoder,
    latent_encoder: LatentEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    local_encoder.eval()
    latent_encoder.eval()

    all_emb, all_lab = [], []
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z_l, source, pos_ids, span_ids, _, seq_pad_mask = local_encoder(input_ids)
            z_prime = latent_encoder(z_l, pos_ids, span_ids, key_padding_mask=seq_pad_mask)

        # Masked mean-pool: ignore merged tokens that stem entirely from padding
        pad_mask = (input_ids != PAD_ID).float()                         # (B, N)
        content = torch.bmm(source, pad_mask.unsqueeze(-1)).squeeze(-1)  # (B, L)
        token_mask = (content > 1e-6).float().unsqueeze(-1)              # (B, L, 1)

        masked = z_prime.float() * token_mask
        emb = masked.sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)  # (B, D)

        all_emb.append(emb.cpu().numpy())
        all_lab.append(labels.numpy())

    return np.concatenate(all_emb), np.concatenate(all_lab)


# ──────────────────────────────── main ──────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MergeDNA embeddings on genomic benchmarks",
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the training output folder (contains checkpoints + config.yaml)")
    parser.add_argument("--benchmark", type=str, default="human_nontata_promoters",
                        help="Benchmark name (default: human_nontata_promoters)")
    parser.add_argument("--fasta", type=str, default=None,
                        help="Path to the reference FASTA (auto-selected by benchmark if omitted)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--list_benchmarks", action="store_true",
                        help="Print available benchmarks and exit")
    parser.add_argument("--save_spans", type=str, default=None, metavar="PATH",
                        help="If set, collect merged-token span lengths (bases/token) for the "
                             "positive class of human_nontata_promoters and save to PATH (.npz)")
    args = parser.parse_args()

    if args.list_benchmarks:
        print("Available local genomic benchmarks:")
        for name in list_local_benchmarks():
            default_fa = _BENCHMARK_FASTA.get(name, "hg38.fa")
            supported = "✓" if default_fa else "✗ (unsupported)"
            print(f"  - {name}  [{default_fa or 'unsupported'}]  {supported}")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ── Config & models ──
    cfg = load_resolved_config(args.output_dir)
    max_seq_len = cfg["experiment"]["max_seq_len"]
    embed_dim = cfg["experiment"]["embed_dim"]
    print(f"Model: embed_dim={embed_dim}, max_seq_len={max_seq_len}")

    local_encoder, latent_encoder = build_models(cfg, device)

    ckpt_path = find_latest_checkpoint(args.output_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {args.output_dir}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    local_encoder.load_state_dict(ckpt["local_encoder"])
    latent_encoder.load_state_dict(ckpt["latent_encoder"])
    print(f"  epoch={ckpt.get('epoch', '?')}  step={ckpt.get('global_step', '?')}")

    # ── Benchmark data ──
    print(f"\nBenchmark: {args.benchmark}")
    if args.fasta:
        fasta_path = Path(args.fasta)
    else:
        default_fa = _BENCHMARK_FASTA.get(args.benchmark)
        if default_fa is None:
            sys.exit(
                f"Benchmark '{args.benchmark}' is not supported (mixed or transcript-based coords).\n"
                f"Supported benchmarks: {[k for k, v in _BENCHMARK_FASTA.items() if v]}"
            )
        fasta_path = _REPO_ROOT / default_fa
    if not fasta_path.exists():
        sys.exit(
            f"FASTA not found: {fasta_path}\n"
            f"Download it and place it at that path, or pass --fasta <path>."
        )
    train_dset = LocalGenomicDataset(args.benchmark, "train", fasta_path=fasta_path)
    test_dset = LocalGenomicDataset(args.benchmark, "test", fasta_path=fasta_path)
    print(f"  train={len(train_dset)}  test={len(test_dset)}")

    collate = lambda batch: _collate(batch, max_seq_len)
    loader_kw = dict(batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    train_loader = DataLoader(train_dset, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_dset, shuffle=False, **loader_kw)

    # ── Extract embeddings ──
    print("\nExtracting embeddings …")
    X_train, y_train = extract_embeddings(local_encoder, latent_encoder, train_loader, device)
    X_test, y_test = extract_embeddings(local_encoder, latent_encoder, test_loader, device)
    print(f"  X_train {X_train.shape}  X_test {X_test.shape}")

    # ── Train linear probe ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_classes = len(set(y_train))
    print(f"\nTraining logistic regression ({n_classes} classes) …")
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    # ── Results ──
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(f"\n{'=' * 55}")
    print(f"  {args.benchmark}")
    print(f"{'=' * 55}")
    print(f"  Train accuracy : {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"  Test accuracy  : {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  Test F1 (macro): {f1_score(y_test, y_pred_test, average='macro'):.4f}")
    print(f"\n{classification_report(y_test, y_pred_test)}")

    # ── Span-length analysis (promoter sequences only) ──
    if args.save_spans:
        print("\nCollecting span lengths for human_nontata_promoters positive (train + test) …")
        hg38 = _REPO_ROOT / "hg38.fa"
        pos_train = LocalGenomicDataset(
            "human_nontata_promoters", "train", fasta_path=hg38,
        )
        pos_test = LocalGenomicDataset(
            "human_nontata_promoters", "test", fasta_path=hg38,
        )
        # Keep only the positive class (label 1: positive.csv.gz)
        pos_train_seqs = [(s, l) for s, l in pos_train if l == 1]
        pos_test_seqs  = [(s, l) for s, l in pos_test  if l == 1]
        all_pos = pos_train_seqs + pos_test_seqs
        print(f"  {len(all_pos)} promoter sequences")

        pos_loader = DataLoader(
            all_pos,
            batch_size=args.batch_size,
            collate_fn=lambda batch: _collate(batch, max_seq_len),
            num_workers=2,
        )
        span_lengths = collect_span_lengths(local_encoder, pos_loader, device)

        out_path = Path(args.save_spans)
        np.savez(out_path, span_lengths=span_lengths)
        print(f"  Saved {len(span_lengths):,} token span lengths → {out_path}")
        print(f"  mean={span_lengths.mean():.2f}  median={np.median(span_lengths):.0f}"
              f"  min={span_lengths.min()}  max={span_lengths.max()}")


if __name__ == "__main__":
    main()
