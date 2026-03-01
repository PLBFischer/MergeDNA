import random
from typing import List

import torch
from torch.utils.data import Dataset

from utils.dna import encode_sequence

BASES = "ATCG"


class SyntheticDNADataset(Dataset):
    """Generates periodic DNA sequences from random k-mers for quick testing.

    Each sequence is built by sampling a random k-mer (with k drawn uniformly
    from [min_k, max_k]) and repeating it (max_seq_len // k) times, producing
    a periodically repeating pattern truncated to max_seq_len.
    """

    def __init__(
        self,
        num_sequences: int = 1000,
        max_seq_len: int = 32,
        min_k: int = 2,
        max_k: int = 8,
        seed: int = 42,
    ):
        self.max_seq_len = max_seq_len
        self.sequences: List[str] = []

        rng = random.Random(seed)
        for _ in range(num_sequences):
            k = rng.randint(min_k, max_k)
            kmer = "".join(rng.choice(BASES) for _ in range(k))
            repeats = max_seq_len // k
            seq = (kmer * repeats)[:max_seq_len]
            self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return encode_sequence(self.sequences[idx], self.max_seq_len)
