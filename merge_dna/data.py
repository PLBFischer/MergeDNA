"""
FASTA dataset loader for MergeDNA pre-training.

Reads multi-FASTA files (the Multi-Species Genomes corpus referenced in the
paper) and yields fixed-length chunks of nucleotide-encoded sequences.

DESIGN DECISIONS:
  - Characters outside {A, T, C, G, N} are mapped to N (ambiguous).
  - Sequences shorter than max_seq_len are right-padded with PAD.
  - Sequences are randomly cropped to max_seq_len during training.
"""

import glob
import os
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

NUCLEOTIDE_MAP = {
    "A": 1, "a": 1,
    "T": 2, "t": 2,
    "C": 3, "c": 3,
    "G": 4, "g": 4,
    "N": 5, "n": 5,
}
PAD_ID = 0


def encode_sequence(seq: str, max_len: int) -> torch.Tensor:
    """Encode a nucleotide string into integer token ids.

    Unknown characters are mapped to N (5).  Output is right-padded to
    *max_len* with PAD (0).
    """
    ids = [NUCLEOTIDE_MAP.get(c, 5) for c in seq[:max_len]]
    if len(ids) < max_len:
        ids.extend([PAD_ID] * (max_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


class FASTADataset(Dataset):
    """Map-style dataset that pre-loads all sequences from FASTA files.

    Good for smaller corpora that fit in memory.
    """

    def __init__(
        self,
        fasta_paths: List[str],
        max_seq_len: int = 4096,
        random_crop: bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.random_crop = random_crop
        self.sequences: List[str] = []
        for path in fasta_paths:
            self.sequences.extend(_parse_fasta(path))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        if self.random_crop and len(seq) > self.max_seq_len:
            start = random.randint(0, len(seq) - self.max_seq_len)
            seq = seq[start : start + self.max_seq_len]
        return encode_sequence(seq, self.max_seq_len)


class StreamingFASTADataset(IterableDataset):
    """Iterable dataset that streams chunks from large FASTA files.

    Suitable for the Multi-Species Genomes corpus which may be too large
    to hold entirely in memory.  Each worker handles a shard of files.
    """

    def __init__(
        self,
        fasta_dir: str,
        max_seq_len: int = 4096,
        file_glob: str = "*.fa*",
    ):
        self.fasta_dir = fasta_dir
        self.max_seq_len = max_seq_len
        self.file_glob = file_glob

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = sorted(glob.glob(os.path.join(self.fasta_dir, self.file_glob)))
        if not files:
            return

        if worker_info is not None:
            per_worker = len(files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(files)
            files = files[start:end]

        random.shuffle(files)
        for fpath in files:
            for seq in _parse_fasta(fpath):
                if len(seq) <= self.max_seq_len:
                    yield encode_sequence(seq, self.max_seq_len)
                else:
                    start = random.randint(0, len(seq) - self.max_seq_len)
                    yield encode_sequence(
                        seq[start : start + self.max_seq_len],
                        self.max_seq_len,
                    )


def _parse_fasta(path: str) -> List[str]:
    """Parse a FASTA file into a list of sequence strings."""
    sequences = []
    current: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences
