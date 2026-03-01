import genomic_benchmarks.utils.paths as _gb_paths
from pathlib import Path
_gb_paths.CACHE_PATH = Path("/orcd/data/omarabu/001/paolo/.genomic_benchmarks")
_gb_paths.CACHE_PATH.mkdir(parents=True, exist_ok=True)

from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset

datasets = [
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "drosophila_enhancers_stark",
    "dummy_mouse_enhancers_ensembl",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
]

VALID = set("ACTGactg")

for name in datasets:
    ds = get_dataset(name, split="train")
    counts = {}
    total_bases = 0
    for seq, _ in ds:
        total_bases += len(seq)
        for ch in seq:
            if ch not in VALID:
                counts[ch] = counts.get(ch, 0) + 1
    extra = f"  non-ACTG chars: { {k: counts[k] for k in sorted(counts)} }" if counts else ""
    print(f"{name}: {len(ds)} seqs, {total_bases} bases{extra}")
