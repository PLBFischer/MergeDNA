#!/usr/bin/env python
"""
Plot a histogram of merged-token span lengths from a span_lengths.npz file.

Usage:
    python plot_spans.py <path/to/span_lengths.npz> [--out <figure.png>]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("npz", type=str, help="Path to span_lengths.npz")
parser.add_argument("--out", type=str, default=None,
                    help="Output image path (default: same dir as npz, span_lengths_hist.png)")
args = parser.parse_args()

npz_path = Path(args.npz)
out_path = Path(args.out) if args.out else npz_path.parent / "span_lengths_hist.png"

spans = np.load(npz_path)["span_lengths"]
counts = np.bincount(spans)

print(f"Total tokens : {len(spans):,}")
print(f"Mean span    : {spans.mean():.3f} bp")
print(f"Median span  : {np.median(spans):.0f} bp")
print(f"Min / Max    : {spans.min()} / {spans.max()} bp")
print()
print("Span  Count    Fraction")
for length, count in enumerate(counts):
    if count > 0:
        print(f"  {length:3d}  {count:8,d}   {count / len(spans) * 100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 5))

bins = np.arange(spans.min(), spans.max() + 2) - 0.5
ax.hist(spans, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)

ax.set_xlabel("Bases per merged token", fontsize=13)
ax.set_ylabel("Token count", fontsize=13)
ax.set_title("Merged-token span lengths\n(human_nontata_promoters, positive class)", fontsize=13)
ax.set_xticks(range(spans.min(), spans.max() + 1))

mean_val = spans.mean()
ax.axvline(mean_val, color="tomato", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.2f}")
ax.legend(fontsize=11)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"\nSaved → {out_path}")
