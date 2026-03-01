# MergeDNA — Reproduction

Reproduction of the **MergeDNA** model architecture, pre-training losses, and
training loop from:

> Li et al., "MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization
> through Token Merging", arXiv:2511.14806

There is no publicly available codebase for MergeDNA.  This implementation is
based solely on the paper description.  All points where the paper is
underspecified and an independent design decision was made are documented below
and marked with `DESIGN DECISION` comments in the source code.

---

## Quick start

### Requirements

- Python >= 3.10
- PyTorch >= 2.1 with CUDA
- flash-attn >= 2.5

```bash
pip install -r requirements.txt
```

### Pre-training

```bash
torchrun --nproc_per_node=8 -m merge_dna.train \
    --data_dir /path/to/multi_species_genomes \
    --output_dir ./checkpoints
```

### Encoder-only inference (downstream classification)

```python
from merge_dna import MergeDNA, MergeDNAConfig

config = MergeDNAConfig()
model = MergeDNA(config).cuda().eval()

# input_ids: (B, N) tensor with nucleotide encoding
# A=1, T=2, C=3, G=4, N=5, PAD=0
latent_embeds = model(input_ids, mode="encoder_only")  # (B, L, 1024)
pooled = latent_embeds.mean(dim=1)  # mean-pool for classification
```

---

## Architecture overview

```
Input X (N bases)
    |
    v
[Local Encoder]  -- 4 LocalToMeAttention layers (local-window attn + merge)
    |                produces Z_L (L = N/2) + source matrix S
    v
[Latent Encoder] -- 20 full-attention Transformer blocks
    |                produces Z'_L
    v
[Latent Decoder] -- 4 full-attention Transformer blocks (pre-training only)
    |                produces Z_hat_L
    v
[Unmerge via S^T + Local Decoder] -- 2 local-window attention blocks
    |
    v
Output X_hat (N bases) -> logits (N, vocab)
```

**Total parameters: ~380M** (51M + 253M + 51M + 25M)

---

## Hyperparameters from the paper (Table A1)

| Parameter             | Value       | Source          |
|-----------------------|-------------|-----------------|
| Embedding dim (D)     | 1024        | Paper           |
| Local window size     | 16          | Paper           |
| Local Encoder layers  | 4           | Paper           |
| Latent Encoder layers | 20          | Paper           |
| Latent Decoder layers | 4           | Paper           |
| Local Decoder layers  | 2           | Paper           |
| Pre-training iters    | 100,000     | Paper           |
| Max sequence length   | 4096        | Paper           |
| Batch size            | 256         | Paper           |
| Learning rate         | 1e-4        | Paper           |
| AdamW betas           | (0.9, 0.95) | Paper          |
| Weight decay          | 1e-8        | Paper           |
| Warmup iterations     | 10,000      | Paper           |
| Gradient clipping     | 1.0         | Paper           |
| lambda (latent MTR)   | 0.25        | Paper           |
| Target compression    | L=N/2, K=L/2 | Paper         |

---

## Independent design decisions

The following choices were made where the paper does not provide enough detail
for unambiguous reproduction.  Each is marked with a `DESIGN DECISION` comment
in the source code.

### 1. Attention heads and head dimension
**Choice:** 16 heads, head_dim = 64.
**Rationale:** Not stated in the paper.  Inferred from LLaMA-1 convention at
D = 1024 (16 heads x 64).

### 2. FFN hidden dimension (SwiGLU)
**Choice:** 2752.
**Rationale:** Reverse-engineered from the 380M parameter budget.  The Latent
Encoder has 253M parameters across 20 layers.  Per-block formula for SwiGLU:
`4 * D^2 (attn) + 3 * D * ffn_dim (SwiGLU) + 2 * D (norms)`.  Solving for
ffn_dim gives ~2752.

### 3. Vocabulary
**Choice:** {PAD=0, A=1, T=2, C=3, G=4, N=5} (6 tokens).
**Rationale:** Paper defines {A, T, C, G}.  We add PAD for batching and N for
ambiguous/unknown bases commonly found in reference genomes.

### 4. Token merging similarity metric
**Choice:** Cosine similarity between DTEM-style grouping embeddings
(linear projection to 64-dim space).
**Rationale:** Paper says "lightweight grouping embedding as in DTEM
(Lee & Hong, 2024)" but does not specify the projection dimension or
exact similarity function.

### 5. Merge schedule across layers
**Choice:** r = 2 pairs per window per layer (4 layers: 16 -> 14 -> 12 -> 10 -> 8).
**Rationale:** Paper says r_l pairs are merged per window per layer to reach
L = N/2.  With 4 layers and window size 16, r = 2 per layer achieves exactly
50% compression.  During training, compression-ratio sampling varies this.

### 6. Window management after merging
**Choice:** Original window partitioning is fixed; each window of 16 tokens
shrinks progressively (not re-partitioned between layers).
**Rationale:** Paper does not specify.  Keeping windows fixed preserves the
local sequential structure and is simpler to implement.

### 7. Merged token position (RoPE)
**Choice:** The merged ("keeper") token inherits the position index of the
left (smaller-index) constituent.
**Rationale:** Not specified in the paper.  The left token is the "keeper" in
the ToMe framework.

### 8. Pooling for classification
**Choice:** Mean pooling over Latent Encoder outputs.
**Rationale:** Paper says "fine-tune a classification head on the latent
encoder's output" but does not specify the pooling strategy.

### 9. Bias in linear layers
**Choice:** No bias (bias=False) in all attention and FFN projections.
**Rationale:** Following LLaMA convention.  Paper does not state this.

### 10. Weight initialization
**Choice:** Normal distribution with std = 0.02.
**Rationale:** Standard Transformer initialization.  Paper does not specify.

### 11. Dropout
**Choice:** Configurable, default 0.0.
**Rationale:** Paper does not mention dropout.

### 12. Global token merging in Latent Encoder
**Choice:** A single global merge pass after all Latent Encoder layers
(rather than per-layer merging).
**Rationale:** Paper says "replaces standard attention with ToMe-style
Attention" for the K-token selection.  We implement this as a post-encoder
global merge step, which is simpler and achieves the same selectivity goal.

### 13. Output head
**Choice:** A single linear projection D -> vocab_size (no bias).
**Rationale:** Not explicitly described but required for the cross-entropy
loss formulation.

### 14. Adjacency constraint in token merging
**Choice:** Only adjacent token pairs within each window are candidates.
**Rationale:** Paper says merging happens "within each window" but does not
specify adjacency.  We enforce it because DNA is sequential and merging
non-adjacent tokens would break positional semantics and produce
non-contiguous "words".

---

## File structure

```
merge_dna/
  __init__.py        -- Package exports
  config.py          -- MergeDNAConfig dataclass
  layers.py          -- RMSNorm, SwiGLU FFN, RoPE
  attention.py       -- FullAttention, LocalWindowAttention
  token_merge.py     -- TokenMergeModule, GlobalTokenMergeModule, token_unmerge
  blocks.py          -- TransformerBlock, LocalToMeAttentionBlock, LocalAttentionBlock
  model.py           -- MergeDNA model (all 4 components)
  losses.py          -- MTR, AMTM, and combined pre-training loss
  data.py            -- FASTA dataset loaders
  train.py           -- Pre-training script (DDP, AdamW, cosine LR)
requirements.txt     -- Python dependencies
README.md            -- This file
```
