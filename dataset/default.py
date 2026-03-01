import random
from typing import List

import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset

from utils.dna import encode_sequence


class DNADataset(Dataset):
    def __init__(self, fasta_path: str, max_seq_len: int = 4096, dataset_len: int = 1000):
        self.max_seq_len = max_seq_len
        self.dataset_len = dataset_len
        self.sequences: List[str] = []
        with open(to_absolute_path(fasta_path), "r") as f:
            current_seq: List[str] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_seq:
                        self.sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                self.sequences.append("".join(current_seq))

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[random.randint(0, len(self.sequences) - 1)]
        if len(seq) > self.max_seq_len:
            start = random.randint(0, len(seq) - self.max_seq_len)
            seq = seq[start : start + self.max_seq_len]
        return encode_sequence(seq, self.max_seq_len)