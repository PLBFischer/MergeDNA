import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from utils.dna import encode_sequence

LETTER_TO_K = {"C": 4, "T": 8, "G": 16}
SEQ_LEN = 64
TOTAL_LETTER_COUNT = 32


class SyntheticStructuredDNADataset(Dataset):
    """Generates structured DNA sequences with deterministic k-mer patterns.

    Each sequence has length 64.  A letter is chosen uniformly from {C, T, G},
    which fixes a k-mer size (C→4, T→8, G→16).  Enough copies of that
    homopolymer k-mer are placed at random non-overlapping positions to produce
    exactly 32 occurrences of the chosen letter; the remaining 32 positions are
    filled with A.

    Sequences are persisted to a FASTA file and reloaded on subsequent
    instantiations to avoid redundant generation.
    """

    def __init__(
        self,
        num_sequences: int = 100_000,
        seed: int = 42,
        fasta_path: Optional[str] = None,
        dataset_len: int = 10_000,
    ):
        self.max_seq_len = SEQ_LEN
        self.sequences: List[str] = []
        self.dataset_len = dataset_len

        if fasta_path is None:
            fasta_path = f"synthetic_structured_n{num_sequences}_seed{seed}.fasta"
        self._fasta_path = Path(fasta_path)

        if self._fasta_path.exists():
            self.sequences = self._load_fasta(self._fasta_path)
        else:
            rng = random.Random(seed)
            letters = list(LETTER_TO_K.keys())

            for _ in range(num_sequences):
                letter = rng.choice(letters)
                k = LETTER_TO_K[letter]
                num_kmers = TOTAL_LETTER_COUNT // k
                num_a = SEQ_LEN - TOTAL_LETTER_COUNT

                # Random composition of num_a A's into (num_kmers + 1) gaps
                # by sampling num_kmers cut-points in [0, num_a] and sorting.
                cuts = sorted(rng.randint(0, num_a) for _ in range(num_kmers))
                gaps = []
                prev = 0
                for c in cuts:
                    gaps.append(c - prev)
                    prev = c
                gaps.append(num_a - prev)

                parts: List[str] = []
                for i, g in enumerate(gaps):
                    parts.append("A" * g)
                    if i < num_kmers:
                        parts.append(letter * k)
                self.sequences.append("".join(parts))

            self._save_fasta(self._fasta_path, self.sequences)

    @staticmethod
    def _load_fasta(path: Path) -> List[str]:
        sequences: List[str] = []
        current: List[str] = []
        with path.open() as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if current:
                        sequences.append("".join(current))
                        current = []
                else:
                    current.append(line)
        if current:
            sequences.append("".join(current))
        return sequences

    @staticmethod
    def _save_fasta(path: Path, sequences: List[str]) -> None:
        with path.open("w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq{i}\n{seq}\n")

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        return encode_sequence(self.sequences[idx], self.max_seq_len)
