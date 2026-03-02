#!/usr/bin/env python
"""
Evaluate whether a trained MergeDNA local encoder learns to align its
token-merging boundaries with the fragment (k-mer) boundaries of the
synthetic vocabulary used during training.

Loads the latest checkpoint from a synthetic_vocab training run, regenerates
the vocabulary and 1000 sequences (with fragment-level provenance tracking),
runs the local encoder, and produces two analyses:

1. **Fragment purity** — for each merged token, what fraction of its bases
   belong to the single most common source fragment?  A perfectly vocabulary-
   aligned tokenizer would score 1.0.

2. **Fragment-length vs token-span correlation** — do bases from longer
   fragments end up in longer merged tokens?  Reports Pearson and Spearman
   correlations plus a scatter plot.

Usage:
    python eval_vocab_merge.py --output_dir outputs/<run_dir>
"""

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.local_encoder import LocalEncoder
from utils.dna import encode_sequence

BASES = "ATCG"


# ──────────────────── checkpoint / config helpers ────────────────────────────


def find_latest_checkpoint(output_dir: str):
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


def load_config(output_dir: str) -> dict:
    cfg = OmegaConf.load(os.path.join(output_dir, "config.yaml"))
    return OmegaConf.to_container(cfg, resolve=True)


def build_local_encoder(cfg: dict, device: torch.device) -> LocalEncoder:
    exp = cfg["experiment"]
    le = cfg["local_encoder"]
    return LocalEncoder(
        vocab_size=cfg["vocab_size"],
        pad_token_id=cfg["pad_token_id"],
        embed_dim=exp["embed_dim"],
        num_heads=exp["num_heads"],
        ffn_dim=exp["ffn_dim"],
        num_layers=le["num_layers"],
        local_window_size=le.get("local_window_size", exp.get("local_window_size", 16)),
        merge_group_dim=le.get("merge_group_dim", 64),
        max_seq_len=exp["max_seq_len"],
        compression_ratio_mean=le.get("compression_ratio_mean", 0.5),
        compression_ratio_min=le.get("compression_ratio_min", 0.4),
        compression_ratio_max=le.get("compression_ratio_max", 0.6),
    ).to(device)


# ──────────── vocabulary & sequence generation with provenance ───────────────


def build_vocab(vocab_size: int, min_k: int, max_k: int, seed: int):
    """Reconstruct the deterministic vocabulary used by SyntheticVocabDNADataset."""
    rng = random.Random(seed)
    vocab = []
    for _ in range(vocab_size):
        k = rng.randint(min_k, max_k)
        vocab.append("".join(rng.choice(BASES) for _ in range(k)))
    return vocab, rng


def generate_sequences_with_provenance(
    rng: random.Random,
    vocab: list[str],
    num_sequences: int,
    num_to_skip: int,
    min_seq_len: int,
    max_seq_len: int,
):
    """Generate sequences exactly as SyntheticVocabDNADataset does, but also
    record, for every base position, which fragment index it came from and
    the length of that fragment.

    We first advance the RNG by `num_to_skip` sequences (to match the dataset
    state after generating its sequences), then generate `num_sequences` fresh
    ones.  However — the dataset generates all sequences in __init__ and then
    the dataloader indexes into those same sequences.  So we just regenerate
    from the same RNG continuation used during dataset construction, which
    means we replay the first `num_sequences` sequences from the pool.

    Returns:
        sequences: list of DNA strings
        fragment_ids: list of np.ndarray (per-base fragment index within the seq)
        fragment_lens: list of np.ndarray (per-base length of the owning fragment)
        all_parts: list of list[str] (the k-mers concatenated for each sequence)
    """
    sequences = []
    fragment_ids = []
    fragment_lens = []
    all_parts = []

    for _ in range(num_sequences):
        target_len = rng.randint(min_seq_len, max_seq_len)
        parts: list[str] = []
        length = 0
        while length < target_len:
            kmer = rng.choice(vocab)
            parts.append(kmer)
            length += len(kmer)

        seq = "".join(parts)[:target_len]

        frag_id = np.empty(len(seq), dtype=np.int32)
        frag_len = np.empty(len(seq), dtype=np.int32)
        pos = 0
        for fi, part in enumerate(parts):
            end = min(pos + len(part), target_len)
            frag_id[pos:end] = fi
            frag_len[pos:end] = len(part)
            pos = end
            if pos >= target_len:
                break

        sequences.append(seq)
        fragment_ids.append(frag_id)
        fragment_lens.append(frag_len)
        all_parts.append(parts)

    return sequences, fragment_ids, fragment_lens, all_parts


# ──────────────────────── merge analysis ─────────────────────────────────────


def analyse_merge_alignment(
    source: torch.Tensor,
    span_ids: torch.Tensor,
    seq_pad_mask: torch.Tensor,
    fragment_ids: list[np.ndarray],
    fragment_lens: list[np.ndarray],
    max_seq_len: int,
):
    """Compute per-merged-token purity and per-base (frag_len, token_span) pairs.

    Args:
        source: (B, L, N) source matrix from local encoder.
        span_ids: (B, L) span sizes of merged tokens.
        seq_pad_mask: (B, L) True for real merged tokens.
        fragment_ids: per-sequence arrays of fragment indices for each base.
        fragment_lens: per-sequence arrays of fragment lengths for each base.
        max_seq_len: N dimension of source matrix.

    Returns:
        purities: list of floats — one per real merged token.
        base_frag_lens: list of ints — fragment length for each base.
        base_token_spans: list of ints — merged-token span for each base.
        boundary_stats: dict with boundary alignment metrics.
    """
    B = source.shape[0]
    source_np = source.detach().cpu().numpy()
    span_np = span_ids.detach().cpu().long().numpy()
    mask_np = seq_pad_mask.detach().cpu().numpy()

    purities = []
    base_frag_lens = []
    base_token_spans = []

    boundary_match = 0
    boundary_total = 0

    for b in range(B):
        seq_len = len(fragment_ids[b])
        fid = fragment_ids[b]
        flen = fragment_lens[b]

        for l in range(source_np.shape[1]):
            if not mask_np[b, l]:
                continue

            covered = source_np[b, l, :seq_len]
            base_indices = np.where(covered > 0.5)[0]
            if len(base_indices) == 0:
                continue

            span = int(span_np[b, l])

            frags_in_token = fid[base_indices]
            counter = Counter(frags_in_token.tolist())
            most_common_count = counter.most_common(1)[0][1]
            purity = most_common_count / len(base_indices)
            purities.append(purity)

            for bi in base_indices:
                base_frag_lens.append(int(flen[bi]))
                base_token_spans.append(span)

        # Boundary alignment: compare fragment boundaries to merge boundaries
        # Fragment boundaries are positions where fragment_id changes
        frag_boundaries = set()
        for i in range(1, seq_len):
            if fid[i] != fid[i - 1]:
                frag_boundaries.add(i)

        # Merge boundaries: positions where one merged token ends and another begins
        # Reconstruct base-to-token assignment
        base_to_token = np.full(seq_len, -1, dtype=np.int32)
        for l in range(source_np.shape[1]):
            if not mask_np[b, l]:
                continue
            covered = source_np[b, l, :seq_len]
            base_indices = np.where(covered > 0.5)[0]
            for bi in base_indices:
                base_to_token[bi] = l

        merge_boundaries = set()
        for i in range(1, seq_len):
            if base_to_token[i] != base_to_token[i - 1] and base_to_token[i] >= 0 and base_to_token[i - 1] >= 0:
                merge_boundaries.add(i)

        if frag_boundaries:
            matched = len(frag_boundaries & merge_boundaries)
            boundary_match += matched
            boundary_total += len(frag_boundaries)

    boundary_stats = {
        "boundary_recall": boundary_match / max(1, boundary_total),
        "boundary_match": boundary_match,
        "boundary_total": boundary_total,
    }

    return purities, base_frag_lens, base_token_spans, boundary_stats


# ──────────────────── random-merge baseline ──────────────────────────────────


def random_merge_baseline(
    fragment_ids: list[np.ndarray],
    fragment_lens: list[np.ndarray],
    real_span_distribution: list[int],
    num_trials: int = 5,
):
    """Compute expected purity / correlation under random contiguous merges
    that produce the same span-size distribution as the trained model.

    For each sequence, randomly partition it into contiguous segments whose
    sizes are drawn (with replacement) from the empirical span distribution.
    """
    span_arr = np.array(real_span_distribution)
    all_purities = []
    all_base_frag_lens = []
    all_base_token_spans = []
    rng = np.random.RandomState(123)

    for _ in range(num_trials):
        for b in range(len(fragment_ids)):
            seq_len = len(fragment_ids[b])
            fid = fragment_ids[b]
            flen = fragment_lens[b]

            pos = 0
            while pos < seq_len:
                span = int(rng.choice(span_arr))
                span = min(span, seq_len - pos)
                if span == 0:
                    span = 1
                base_indices = np.arange(pos, pos + span)

                frags_in_token = fid[base_indices]
                counter = Counter(frags_in_token.tolist())
                most_common_count = counter.most_common(1)[0][1]
                purity = most_common_count / len(base_indices)
                all_purities.append(purity)

                for bi in base_indices:
                    all_base_frag_lens.append(int(flen[bi]))
                    all_base_token_spans.append(span)

                pos += span

    return all_purities, all_base_frag_lens, all_base_token_spans


# ──────────────────────── plotting ───────────────────────────────────────────


def plot_results(
    purities,
    base_frag_lens,
    base_token_spans,
    rand_purities,
    rand_frag_lens,
    rand_token_spans,
    boundary_stats,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Purity histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(purities, bins=bins, alpha=0.7, label=f"Trained (mean={np.mean(purities):.3f})", density=True)
    ax.hist(rand_purities, bins=bins, alpha=0.5, label=f"Random baseline (mean={np.mean(rand_purities):.3f})", density=True)
    ax.set_xlabel("Purity (fraction of bases from dominant fragment)")
    ax.set_ylabel("Density")
    ax.set_title("Merged-Token Fragment Purity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "purity_histogram.png"), dpi=150)
    plt.close(fig)

    # --- 2. Fragment length vs token span scatter + heatmap ---
    frag_arr = np.array(base_frag_lens)
    span_arr = np.array(base_token_spans)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trained model
    ax = axes[0]
    unique_frags = sorted(set(frag_arr))
    unique_spans = sorted(set(span_arr))
    max_frag = max(unique_frags)
    max_span = max(unique_spans)
    heatmap = np.zeros((max_frag + 1, max_span + 1))
    for fl, ts in zip(frag_arr, span_arr):
        heatmap[fl, ts] += 1
    # Normalize each row
    row_sums = heatmap.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    heatmap_norm = heatmap / row_sums

    im = ax.imshow(
        heatmap_norm[1:max_frag + 1, 1:max_span + 1],
        aspect="auto", origin="lower",
        extent=[0.5, max_span + 0.5, 0.5, max_frag + 0.5],
        cmap="YlOrRd",
    )
    ax.set_xlabel("Merged token span (# bases)")
    ax.set_ylabel("Fragment length (k-mer size)")
    ax.set_title("Trained Model")
    plt.colorbar(im, ax=ax, label="P(token span | frag length)")

    # Random baseline
    ax = axes[1]
    rfrag_arr = np.array(rand_frag_lens)
    rspan_arr = np.array(rand_token_spans)
    r_max_span = max(rspan_arr) if len(rspan_arr) > 0 else max_span
    all_max_span = max(max_span, r_max_span)
    heatmap_r = np.zeros((max_frag + 1, all_max_span + 1))
    for fl, ts in zip(rfrag_arr, rspan_arr):
        if fl <= max_frag and ts <= all_max_span:
            heatmap_r[fl, ts] += 1
    row_sums_r = heatmap_r.sum(axis=1, keepdims=True)
    row_sums_r[row_sums_r == 0] = 1
    heatmap_r_norm = heatmap_r / row_sums_r

    im2 = ax.imshow(
        heatmap_r_norm[1:max_frag + 1, 1:all_max_span + 1],
        aspect="auto", origin="lower",
        extent=[0.5, all_max_span + 0.5, 0.5, max_frag + 0.5],
        cmap="YlOrRd",
    )
    ax.set_xlabel("Merged token span (# bases)")
    ax.set_ylabel("Fragment length (k-mer size)")
    ax.set_title("Random Baseline")
    plt.colorbar(im2, ax=ax, label="P(token span | frag length)")

    fig.suptitle("Fragment Length vs Merged Token Span", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "frag_len_vs_token_span.png"), dpi=150)
    plt.close(fig)

    # --- 3. Mean token span per fragment length (bar chart) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    frag_to_spans: dict[int, list] = {}
    for fl, ts in zip(frag_arr, span_arr):
        frag_to_spans.setdefault(fl, []).append(ts)
    frag_to_spans_rand: dict[int, list] = {}
    for fl, ts in zip(rfrag_arr, rspan_arr):
        frag_to_spans_rand.setdefault(fl, []).append(ts)

    frag_lengths = sorted(frag_to_spans.keys())
    means_trained = [np.mean(frag_to_spans[k]) for k in frag_lengths]
    stds_trained = [np.std(frag_to_spans[k]) for k in frag_lengths]
    means_rand = [np.mean(frag_to_spans_rand.get(k, [0])) for k in frag_lengths]

    x = np.arange(len(frag_lengths))
    width = 0.35
    ax.bar(x - width / 2, means_trained, width, label="Trained", alpha=0.8, yerr=stds_trained, capsize=3)
    ax.bar(x + width / 2, means_rand, width, label="Random baseline", alpha=0.6)
    ax.plot(x, frag_lengths, "k--", label="y = fragment length", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(frag_lengths)
    ax.set_xlabel("Fragment length (k-mer size)")
    ax.set_ylabel("Mean merged token span")
    ax.set_title("Mean Token Span per Fragment Length")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "mean_span_per_frag_len.png"), dpi=150)
    plt.close(fig)

    # --- 4. Boundary recall bar ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Boundary Recall"], [boundary_stats["boundary_recall"]], color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title(
        f"Fragment Boundary Recall\n"
        f"({boundary_stats['boundary_match']}/{boundary_stats['boundary_total']} boundaries matched)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "boundary_recall.png"), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {save_dir}/")


# ──────────────────────────── main ───────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local-encoder merge alignment with synthetic vocabulary fragments",
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Training output directory (contains config.yaml + checkpoints)")
    parser.add_argument("--num_sequences", type=int, default=1000,
                        help="Number of sequences to evaluate (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Where to save plots (default: <output_dir>/eval_vocab_merge)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ── Load config ───────────────────────────────────────────────────────
    cfg = load_config(args.output_dir)
    ds_cfg = cfg["dataset"]
    exp_cfg = cfg["experiment"]
    max_seq_len = exp_cfg["max_seq_len"]

    print(f"Experiment: {exp_cfg.get('name', '?')}")
    print(f"  max_seq_len={max_seq_len}, embed_dim={exp_cfg['embed_dim']}")
    print(f"Dataset: vocab_size={ds_cfg['vocab_size']}, "
          f"k=[{ds_cfg['min_k']}, {ds_cfg['max_k']}], "
          f"seq_len=[{ds_cfg['min_seq_len']}, {ds_cfg['max_seq_len']}], "
          f"seed={ds_cfg['seed']}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = find_latest_checkpoint(args.output_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {args.output_dir}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  epoch={ckpt.get('epoch', '?')}, step={ckpt.get('global_step', '?')}")

    local_encoder = build_local_encoder(cfg, device)
    local_encoder.load_state_dict(ckpt["local_encoder"])
    local_encoder.eval()
    print(f"  Local encoder loaded ({sum(p.numel() for p in local_encoder.parameters()):,} params)")

    # ── Reconstruct vocabulary and generate sequences ─────────────────────
    print(f"\nReconstructing vocabulary (seed={ds_cfg['seed']}) ...")
    vocab, rng = build_vocab(
        vocab_size=ds_cfg["vocab_size"],
        min_k=ds_cfg["min_k"],
        max_k=ds_cfg["max_k"],
        seed=ds_cfg["seed"],
    )
    print(f"  Vocabulary: {len(vocab)} k-mers, lengths {min(len(v) for v in vocab)}-{max(len(v) for v in vocab)}")
    print(f"  Examples: {vocab[:8]}")

    # The dataset RNG is used first for vocab, then for all num_sequences sequences.
    # We need to advance past the dataset's sequences to stay in sync, OR
    # simply regenerate the dataset's sequences (since those ARE what was trained on).
    # We want to evaluate on the actual training sequences, so we regenerate them.
    print(f"\nGenerating {args.num_sequences} sequences with fragment tracking ...")
    sequences, fragment_ids, fragment_lens, all_parts = generate_sequences_with_provenance(
        rng=rng,
        vocab=vocab,
        num_sequences=args.num_sequences,
        num_to_skip=0,
        min_seq_len=ds_cfg["min_seq_len"],
        max_seq_len=ds_cfg["max_seq_len"],
    )
    print(f"  Generated {len(sequences)} sequences")
    print(f"  Seq lengths: min={min(len(s) for s in sequences)}, max={max(len(s) for s in sequences)}")
    print(f"  Example seq 0: '{sequences[0][:60]}...' ({len(sequences[0])} bases)")
    print(f"  Example parts 0: {all_parts[0][:6]}...")

    # ── Encode and run local encoder ──────────────────────────────────────
    print(f"\nRunning local encoder on {len(sequences)} sequences (batch_size={args.batch_size}) ...")
    all_source = []
    all_span_ids = []
    all_seq_pad_mask = []

    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            end = min(start + args.batch_size, len(sequences))
            batch_seqs = sequences[start:end]
            input_ids = torch.stack([
                encode_sequence(s, max_seq_len) for s in batch_seqs
            ]).to(device)

            z_l, source, pos_ids, span_ids, r_per_layer, seq_pad_mask = local_encoder(input_ids)

            all_source.append(source.cpu())
            all_span_ids.append(span_ids.cpu())
            all_seq_pad_mask.append(seq_pad_mask.cpu())

            if start == 0:
                print(f"  Input shape: {input_ids.shape}, Output shape: z_l={z_l.shape}, "
                      f"source={source.shape}, span_ids={span_ids.shape}")

    source_cat = torch.cat(all_source, dim=0)
    span_cat = torch.cat(all_span_ids, dim=0)
    mask_cat = torch.cat(all_seq_pad_mask, dim=0)

    # ── Analyse alignment ─────────────────────────────────────────────────
    print("\nAnalysing merge alignment ...")
    purities, base_frag_lens, base_token_spans, boundary_stats = analyse_merge_alignment(
        source_cat, span_cat, mask_cat,
        fragment_ids, fragment_lens, max_seq_len,
    )

    # ── Random baseline ───────────────────────────────────────────────────
    print("Computing random-merge baseline ...")
    rand_purities, rand_frag_lens, rand_token_spans = random_merge_baseline(
        fragment_ids, fragment_lens,
        real_span_distribution=base_token_spans,
        num_trials=5,
    )

    # ── Statistics ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    print(f"\n  Fragment purity (fraction of bases from dominant fragment per merged token):")
    print(f"    Trained model : {np.mean(purities):.4f} +/- {np.std(purities):.4f}  "
          f"(median={np.median(purities):.4f})")
    print(f"    Random baseline: {np.mean(rand_purities):.4f} +/- {np.std(rand_purities):.4f}  "
          f"(median={np.median(rand_purities):.4f})")

    print(f"\n  Fragment boundary recall (fraction of true boundaries present in merge boundaries):")
    print(f"    {boundary_stats['boundary_recall']:.4f}  "
          f"({boundary_stats['boundary_match']}/{boundary_stats['boundary_total']})")

    frag_arr = np.array(base_frag_lens, dtype=float)
    span_arr = np.array(base_token_spans, dtype=float)
    pearson_r, pearson_p = sp_stats.pearsonr(frag_arr, span_arr)
    spearman_r, spearman_p = sp_stats.spearmanr(frag_arr, span_arr)

    rfrag_arr = np.array(rand_frag_lens, dtype=float)
    rspan_arr = np.array(rand_token_spans, dtype=float)
    rand_pearson_r, rand_pearson_p = sp_stats.pearsonr(rfrag_arr, rspan_arr)
    rand_spearman_r, rand_spearman_p = sp_stats.spearmanr(rfrag_arr, rspan_arr)

    print(f"\n  Correlation: fragment length vs merged token span")
    print(f"    Trained model:")
    print(f"      Pearson  r={pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"      Spearman r={spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"    Random baseline:")
    print(f"      Pearson  r={rand_pearson_r:.4f}  (p={rand_pearson_p:.2e})")
    print(f"      Spearman r={rand_spearman_r:.4f}  (p={rand_spearman_p:.2e})")

    # Per-fragment-length breakdown
    print(f"\n  Mean merged token span per fragment length:")
    frag_to_spans: dict[int, list] = {}
    for fl, ts in zip(base_frag_lens, base_token_spans):
        frag_to_spans.setdefault(fl, []).append(ts)
    print(f"    {'Frag len':>8s}  {'Mean span':>9s}  {'Std':>6s}  {'Count':>7s}")
    for k in sorted(frag_to_spans.keys()):
        vals = frag_to_spans[k]
        print(f"    {k:>8d}  {np.mean(vals):>9.3f}  {np.std(vals):>6.3f}  {len(vals):>7d}")

    # ── Plots ─────────────────────────────────────────────────────────────
    save_dir = args.save_dir or os.path.join(args.output_dir, "eval_vocab_merge")
    plot_results(
        purities, base_frag_lens, base_token_spans,
        rand_purities, rand_frag_lens, rand_token_spans,
        boundary_stats, save_dir,
    )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
