import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from utils.dna import encode_sequence

BASES = "ATCG"


class SyntheticDNADataset(Dataset):
    """Generates periodic DNA sequences from random k-mers for quick testing.

    Each sequence is built by sampling a random k-mer (with k drawn uniformly
    from [min_k, max_k]) and repeating it (max_seq_len // k) times, producing
    a periodically repeating pattern truncated to max_seq_len.

    Sequences are persisted to a FASTA file and reloaded on subsequent
    instantiations to avoid redundant generation.
    """

    def __init__(
        self,
        num_sequences: int = 1000,
        max_seq_len: int = 32,
        min_k: int = 2,
        max_k: int = 8,
        seed: int = 42,
        fasta_path: Optional[str] = None,
    ):
        self.max_seq_len = max_seq_len
        self.sequences: List[str] = []

        if fasta_path is None:
            fasta_path = (
                f"synthetic_n{num_sequences}_len{max_seq_len}"
                f"_k{min_k}-{max_k}_seed{seed}.fasta"
            )
        self._fasta_path = Path(fasta_path)

        if self._fasta_path.exists():
            self.sequences = self._load_fasta(self._fasta_path)
        else:
            rng = random.Random(seed)
            for _ in range(num_sequences):
                k = rng.randint(min_k, max_k)
                kmer = "".join(rng.choice(BASES) for _ in range(k))
                repeats = max_seq_len // k
                seq = (kmer * repeats)[:max_seq_len]
                self.sequences.append(seq)
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
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return encode_sequence(self.sequences[idx], self.max_seq_len)
