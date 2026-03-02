# MergeDNA

A reproduction of **MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging** ([Li et al., 2025](https://arxiv.org/abs/2511.14806)).

MergeDNA is a hierarchical autoencoder-style Transformer that jointly optimises a differentiable DNA tokeniser and latent context model. It uses progressive token merging to dynamically segment nucleotide sequences into variable-length tokens, then trains with two complementary objectives — Merged Token Reconstruction (MTR) and Adaptive Masked Token Modeling (AMTM).

## Architecture

```
Input DNA (N bases)
       │
  Local Encoder          4 layers of local-window attention + token merging
       │                 N → L tokens  (compression ≈ 50%)
  Latent Encoder         Full self-attention Transformer
       │                 L → L tokens  (global context)
  Latent Decoder         Full self-attention Transformer
       │                 L → L tokens  (reconstruction)
  Local Decoder          Token unmerge (S^T @ Z) + local-window attention
       │                 L → N bases
  Output logits (N × vocab)
```

All blocks follow a LLaMA-style design: RMSNorm, SwiGLU FFN, and Rotary Position Embeddings (RoPE).

## Pre-training Losses

Three forward passes per step (Eq. 8 in the paper):

| Pass | Loss | Description |
|------|------|-------------|
| 1 | L\_MTR(θ) | Full pipeline reconstruction (N→L→L→N). All parameters updated. |
| 2 | λ · L\_MTR(θ\\{φ}) | Latent encoder merges L→K; local encoder frozen. Reconstructs N from K compressed tokens. |
| 3 | L\_AMTM(θ) | Importance-weighted masking over L local tokens (guided by pass-2 source matrix), then full pipeline predicts masked positions. |

**Total: L\_total = L\_MTR + 0.25 · L\_latent\_MTR + L\_AMTM**

## Repository Structure

```
MergeDNA/
├── main.py                    # Hydra entry point for pre-training
├── training.py                # Trainer class + cosine-warmup LR scheduler
├── config/
│   ├── config.yaml            # Root Hydra config
│   ├── experiment/            # Experiment presets (nano, mini, default, …)
│   ├── dataset/               # Dataset configs (FASTA, synthetic, …)
│   ├── local_encoder/         # Local encoder hyperparameters
│   ├── latent_encoder/        # Latent encoder hyperparameters
│   ├── latent_decoder/        # Latent decoder hyperparameters
│   ├── local_decoder/         # Local decoder hyperparameters
│   ├── optimizer/             # AdamW settings
│   ├── scheduler/             # LR schedule settings
│   ├── loss/                  # Loss manager config (lambda, token IDs)
│   └── wandb/                 # Weights & Biases logging config
├── model/
│   ├── attention.py           # Multi-head attention (full + sliding-window)
│   ├── layers.py              # RMSNorm, RoPE, SpanEncoding, SwiGLU FFN
│   ├── transformer_block.py   # Pre-norm Transformer block (full attention)
│   ├── local_blocks.py        # Local attention blocks (with/without merging)
│   └── token_merge.py         # TokenMergeModule + token_unmerge
├── local_encoder/default.py   # Local Encoder (learnable tokeniser)
├── latent_encoder/default.py  # Latent Encoder (global context model)
├── latent_decoder/default.py  # Latent Decoder (token-level reconstruction)
├── local_decoder/default.py   # Local Decoder (base-level detokeniser)
├── loss/default.py            # LossManager (3-pass forward, Eq. 8)
├── dataset/
│   ├── default.py             # DNADataset (loads FASTA files)
│   ├── synthetic.py           # SyntheticDNADataset (periodic k-mer sequences)
│   └── synthetic_vocab.py     # SyntheticVocabDNADataset (shared k-mer vocab)
├── utils/
│   ├── dna.py                 # Nucleotide encoding (A=1, T=2, C=3, G=4, N=5)
│   └── hash_utils.py          # Config hashing for deterministic output dirs
├── eval_benchmark.py          # Linear-probe evaluation (logistic regression)
├── eval_finetune.py           # LoRA + MLP head fine-tuning evaluation
└── plot_spans.py              # Visualise merged-token span length distribution
```

## Quick Start

### Pre-training

```bash
# Nano experiment (lightweight, for testing)
python main.py experiment=nano

# With a real FASTA file
python main.py experiment=nano dataset.fasta_path=/path/to/hg38.fa

# Synthetic data (no FASTA needed)
python main.py experiment=synthetic
```

Hydra configs are composed from `config/config.yaml`. Override any parameter on the command line:

```bash
python main.py experiment=nano training.total_epochs=500 optimizer.lr=5e-5
```

### Evaluation

**Linear probe** (frozen encoder, logistic regression on mean-pooled embeddings):

```bash
python eval_benchmark.py --output_dir outputs/<run_dir> --benchmark human_nontata_promoters
```

**LoRA fine-tuning** (low-rank adapters + MLP classification head):

```bash
python eval_finetune.py --output_dir outputs/<run_dir> --benchmark human_nontata_promoters --epochs 10
```

Run all hg38 benchmarks:

```bash
python eval_finetune.py --output_dir outputs/<run_dir> --benchmark all_hg38
```

List available benchmarks:

```bash
python eval_benchmark.py --output_dir outputs/<run_dir> --list_benchmarks
```

### Span Length Analysis

Collect and plot the distribution of bases per merged token:

```bash
python eval_benchmark.py --output_dir outputs/<run_dir> --save_spans spans.npz
python plot_spans.py spans.npz
```

## Experiment Configs

| Config | Embed dim | Seq len | Local Enc | Latent Enc | Latent Dec | Local Dec |
|--------|-----------|---------|-----------|------------|------------|-----------|
| `nano` | 256 | 512 | 4 layers | 6 layers | 2 layers | 2 layers |
| `mini` | 256 | 512 | 4 layers | 6 layers | 2 layers | 2 layers |
| `default` | 1024 | 4096 | 4 layers | 20 layers | 4 layers | 2 layers |

## Token Vocabulary

| ID | Token |
|----|-------|
| 0 | PAD |
| 1 | A |
| 2 | T |
| 3 | C |
| 4 | G |
| 5 | N (ambiguous) |
| 6 | MASK (AMTM) |

## Logging

Set the `WANDB_API_KEY` environment variable to enable Weights & Biases logging. Tracked metrics include per-component losses (`loss_mtr`, `loss_latent_mtr`, `loss_amtm`), learning rate, and throughput.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Hydra (`hydra-core`, `omegaconf`)
- wandb (optional, for logging)
- pyfaidx, pandas, scikit-learn, numpy, matplotlib (for evaluation scripts)

## Citation

```bibtex
@article{li2025mergedna,
  title={MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging},
  author={Li, Siyuan and Yu, Kai and Wang, Anna and Liu, Zicheng and Yu, Chang and Zhou, Jingbo and Yang, Qirong and Guo, Yucheng and Zhang, Xiaoming and Li, Stan Z.},
  journal={arXiv preprint arXiv:2511.14806},
  year={2025}
}
```
