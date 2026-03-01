import random
from typing import List

import torch
from torch.utils.data import Dataset

from utils.dna import encode_sequence


class DNADataset(Dataset):
    def __init__(self, txt_path: str, max_seq_len: int = 4096):
        self.max_seq_len = max_seq_len
        with open(txt_path, "r") as f:
            self.sequences: List[str] = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        if len(seq) > self.max_seq_len:
            start = random.randint(0, len(seq) - self.max_seq_len)
            seq = seq[start : start + self.max_seq_len]
        return encode_sequence(seq, self.max_seq_len)