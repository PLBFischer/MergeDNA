"""Tests for FASTA data loading and nucleotide encoding."""

import os
import tempfile

import pytest
import torch

from merge_dna.data import (
    FASTADataset,
    PAD_ID,
    _parse_fasta,
    encode_sequence,
)


class TestEncodeSequence:
    def test_basic_encoding(self):
        t = encode_sequence("ATCG", max_len=4)
        assert t.tolist() == [1, 2, 3, 4]

    def test_padding(self):
        t = encode_sequence("AT", max_len=4)
        assert t.tolist() == [1, 2, 0, 0]

    def test_truncation(self):
        t = encode_sequence("ATCGATCG", max_len=4)
        assert len(t) == 4
        assert t.tolist() == [1, 2, 3, 4]

    def test_unknown_maps_to_n(self):
        t = encode_sequence("AXYZ", max_len=4)
        assert t[0].item() == 1  # A
        assert t[1].item() == 5  # X -> N
        assert t[2].item() == 5  # Y -> N
        assert t[3].item() == 5  # Z -> N

    def test_case_insensitive(self):
        t1 = encode_sequence("atcg", max_len=4)
        t2 = encode_sequence("ATCG", max_len=4)
        assert t1.tolist() == t2.tolist()

    def test_n_base(self):
        t = encode_sequence("ANNC", max_len=4)
        assert t.tolist() == [1, 5, 5, 3]


class TestParseFasta:
    def test_single_sequence(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">seq1\nATCGATCG\nATCG\n")
        seqs = _parse_fasta(str(fasta))
        assert len(seqs) == 1
        assert seqs[0] == "ATCGATCGATCG"

    def test_multiple_sequences(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">seq1\nATCG\n>seq2\nGGGG\n>seq3\nAAAA\n")
        seqs = _parse_fasta(str(fasta))
        assert len(seqs) == 3
        assert seqs[0] == "ATCG"
        assert seqs[1] == "GGGG"
        assert seqs[2] == "AAAA"

    def test_empty_lines_ignored(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">seq1\n\nATCG\n\nGGGG\n\n")
        seqs = _parse_fasta(str(fasta))
        assert len(seqs) == 1
        assert seqs[0] == "ATCGGGGG"


class TestFASTADataset:
    def test_dataset_length(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">s1\nATCGATCG\n>s2\nGGGGGGGG\n")
        ds = FASTADataset([str(fasta)], max_seq_len=8, random_crop=False)
        assert len(ds) == 2

    def test_dataset_encoding(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">s1\nATCG\n")
        ds = FASTADataset([str(fasta)], max_seq_len=8, random_crop=False)
        item = ds[0]
        assert item.shape == (8,)
        assert item[:4].tolist() == [1, 2, 3, 4]
        assert item[4:].tolist() == [0, 0, 0, 0]
