import torch

NUCLEOTIDE_MAP = {"A": 1, "a": 1, "T": 2, "t": 2, "C": 3, "c": 3, "G": 4, "g": 4, "N": 5, "n": 5}
PAD_ID = 0
MASK_ID = 6


def encode_sequence(seq: str, max_len: int) -> torch.Tensor:
    ids = []
    for c in seq[:max_len]:
        token_id = NUCLEOTIDE_MAP.get(c)
        if token_id is None:
            raise ValueError(f"Invalid nucleotide character {c!r}. Only {{A, T, C, G, N}} are allowed.")
        ids.append(token_id)
    if len(ids) < max_len:
        ids.extend([PAD_ID] * (max_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long)
