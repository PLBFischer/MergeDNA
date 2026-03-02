import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from utils.dna import encode_sequence

BASES = "ATCG"


class SyntheticVocabDNADataset(Dataset):
    """Generates DNA sequences by concatenating k-mers from a shared vocabulary.

    A fixed vocabulary of k-mers (lengths drawn uniformly from [min_k, max_k])
    is sampled once.  Each sequence is then built by randomly concatenating
    vocabulary k-mers until the target length (drawn uniformly from
    [min_seq_len, max_seq_len]) is reached, then truncating to that length.

    Sequences are persisted to a FASTA file and reloaded on subsequent
    instantiations to avoid redundant generation.
    """

    def __init__(
        self,
        num_sequences: int = 100_000,
        min_seq_len: int = 24,
        max_seq_len: int = 32,
        vocab_size: int = 100,
        min_k: int = 2,
        max_k: int = 8,
        seed: int = 42,
        fasta_path: Optional[str] = None,
        dataset_len: int = 10_000,
    ):
        self.max_seq_len = max_seq_len
        self.sequences: List[str] = []
        self.dataset_len = dataset_len

        if fasta_path is None:
            fasta_path = (
                f"synthetic_vocab_n{num_sequences}_len{min_seq_len}-{max_seq_len}"
                f"_v{vocab_size}_k{min_k}-{max_k}_seed{seed}.fasta"
            )
        self._fasta_path = Path(fasta_path)

        if self._fasta_path.exists():
            self.sequences = self._load_fasta(self._fasta_path)
        else:
            rng = random.Random(seed)

            vocab = []
            for _ in range(vocab_size):
                k = rng.randint(min_k, max_k)
                vocab.append("".join(rng.choice(BASES) for _ in range(k)))

            for _ in range(num_sequences):
                target_len = rng.randint(min_seq_len, max_seq_len)
                parts: List[str] = []
                length = 0
                while length < target_len:
                    kmer = rng.choice(vocab)
                    parts.append(kmer)
                    length += len(kmer)
                self.sequences.append("".join(parts)[:target_len])

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
