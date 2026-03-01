"""
Differentiable token merging and unmerging for MergeDNA.

Implements the local-window token merging mechanism described in Sec 3.3
of the paper, adapted from ToMe (Bolya et al., 2023) with DTEM-style
decoupled grouping embeddings (Lee & Hong, 2024).

DESIGN DECISIONS documented inline where the paper is underspecified.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMergeModule(nn.Module):
    """Differentiable token merging within local windows.

    For each window of tokens, computes pairwise similarity using a
    lightweight *grouping embedding* (DTEM-style), selects the top-r most
    similar adjacent pairs, and merges them via weighted averaging.

    DESIGN DECISIONS:
      - Grouping embedding dimension is a hyperparameter (default 64).
        Paper says "lightweight grouping embedding as in DTEM" but does not
        specify the dimension.
      - Similarity is cosine similarity between grouping-projected features.
      - Only *adjacent* pairs within each window are candidates for merging,
        preserving the sequential structure of DNA.  The paper says "within
        each window" but does not clarify adjacency constraint; we enforce it
        because DNA is sequential and merging non-adjacent tokens would break
        positional semantics.
      - Merge operation: weighted average by L2 norm of the grouping
        embeddings (soft merge from ToMe).  Paper says "adding their
        representations, or a weighted average".
    """

    def __init__(self, dim: int, group_dim: int = 64):
        super().__init__()
        self.group_proj = nn.Linear(dim, group_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        position_ids: torch.Tensor,
        r: int,
        window_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge *r* pairs per window.

        Args:
            x: (B, S, D) token embeddings.
            source: (B, S, N_orig) source matrix mapping current tokens to
                    original input positions.  Each row sums to >= 1.
            position_ids: (B, S) absolute position indices for each token.
            r: number of adjacent pairs to merge per window.
            window_size: current effective window size for this layer.

        Returns:
            x_merged: (B, S - n_merged, D)
            source_merged: (B, S - n_merged, N_orig)
            position_ids_merged: (B, S - n_merged)
        """
        B, S, D = x.shape
        if r <= 0 or S <= 1:
            return x, source, position_ids

        W = window_size
        n_windows = S // W
        remainder = S % W

        merged_parts = []
        source_parts = []
        pos_parts = []

        for win_idx in range(n_windows):
            start = win_idx * W
            end = start + W
            xw, sw, pw = self._merge_window(
                x[:, start:end],
                source[:, start:end],
                position_ids[:, start:end],
                r,
            )
            merged_parts.append(xw)
            source_parts.append(sw)
            pos_parts.append(pw)

        if remainder > 0:
            start = n_windows * W
            r_adj = min(r, remainder - 1)
            xw, sw, pw = self._merge_window(
                x[:, start:],
                source[:, start:],
                position_ids[:, start:],
                r_adj,
            )
            merged_parts.append(xw)
            source_parts.append(sw)
            pos_parts.append(pw)

        x_merged = torch.cat(merged_parts, dim=1)
        source_merged = torch.cat(source_parts, dim=1)
        pos_merged = torch.cat(pos_parts, dim=1)
        return x_merged, source_merged, pos_merged

    def _merge_window(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        position_ids: torch.Tensor,
        r: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge r adjacent pairs in a single window.

        Args:
            x: (B, W, D)
            source: (B, W, N_orig)
            position_ids: (B, W)
            r: pairs to merge

        Returns:
            x_out: (B, W - r, D)
            source_out: (B, W - r, N_orig)
            pos_out: (B, W - r)
        """
        B, W, D = x.shape
        if r <= 0 or W <= 1:
            return x, source, position_ids

        r = min(r, W - 1)

        g = self.group_proj(x)  # (B, W, group_dim)
        g_norm = F.normalize(g, dim=-1)
        sim = (g_norm[:, :-1] * g_norm[:, 1:]).sum(dim=-1)  # (B, W-1)

        g_norms = g.norm(dim=-1)  # (B, W)

        results_x = []
        results_s = []
        results_p = []

        for b in range(B):
            results = self._greedy_merge_single(
                x[b], source[b], position_ids[b],
                sim[b], g_norms[b], r,
            )
            results_x.append(results[0])
            results_s.append(results[1])
            results_p.append(results[2])

        return (
            torch.stack(results_x),
            torch.stack(results_s),
            torch.stack(results_p),
        )

    @staticmethod
    def _greedy_merge_single(
        x: torch.Tensor,
        source: torch.Tensor,
        pos: torch.Tensor,
        sim: torch.Tensor,
        norms: torch.Tensor,
        r: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy non-overlapping merge for a single example.

        Args:
            x: (W, D)
            source: (W, N_orig)
            pos: (W,)
            sim: (W-1,)  adjacent cosine similarities
            norms: (W,) grouping-embedding norms
            r: number of pairs to merge
        """
        W = x.shape[0]
        merged_mask = torch.zeros(W, dtype=torch.bool, device=x.device)

        _, order = sim.sort(descending=True)

        pairs = []
        for idx in order:
            i = idx.item()
            j = i + 1
            if not merged_mask[i] and not merged_mask[j]:
                pairs.append((i, j))
                merged_mask[i] = True
                merged_mask[j] = True
                if len(pairs) == r:
                    break

        pair_set = {i: j for i, j in pairs}

        out_x = []
        out_s = []
        out_p = []
        skip = set()
        for k in range(W):
            if k in skip:
                continue
            if k in pair_set:
                j = pair_set[k]
                skip.add(j)
                w_i = norms[k]
                w_j = norms[j]
                total = w_i + w_j + 1e-8
                merged_tok = (w_i * x[k] + w_j * x[j]) / total
                merged_src = source[k] + source[j]
                out_x.append(merged_tok)
                out_s.append(merged_src)
                out_p.append(pos[k])
            else:
                out_x.append(x[k])
                out_s.append(source[k])
                out_p.append(pos[k])

        return (
            torch.stack(out_x),
            torch.stack(out_s),
            torch.stack(out_p),
        )


class GlobalTokenMergeModule(nn.Module):
    """Token merging at the global (full-sequence) level.

    Used inside the Latent Encoder during the pre-training selection step
    (Sec 3.4) to compress L tokens down to K salient tokens.

    Unlike the local version, this operates over the full sequence without
    windowing and selects globally important tokens.

    DESIGN DECISION: The paper says the Latent Encoder "replaces the standard
    attention with a ToMe-style Attention that merges tokens at the global
    scale".  We interpret this as: after each latent encoder layer, a global
    merge step removes some tokens.  For simplicity we perform a single
    global merge pass after all encoder layers, removing (L - K) tokens by
    iteratively merging the most similar adjacent pairs. This is more
    tractable and achieves the same effect of selecting K salient tokens.
    """

    def __init__(self, dim: int, group_dim: int = 64):
        super().__init__()
        self.group_proj = nn.Linear(dim, group_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        target_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge tokens globally to reach target_len.

        Args:
            x: (B, L, D)
            source: (B, L, N_orig) source matrix from local encoder
            target_len: K, desired output length

        Returns:
            x_merged: (B, K, D)
            source_merged: (B, K, N_orig)  -- called S' in the paper
        """
        B, L, D = x.shape
        r_total = L - target_len
        if r_total <= 0:
            return x, source

        g = self.group_proj(x)
        g_norm = F.normalize(g, dim=-1)

        results_x = []
        results_s = []
        for b in range(B):
            xb, sb = self._merge_single(
                x[b], source[b], g_norm[b], g[b].norm(dim=-1), r_total,
            )
            results_x.append(xb)
            results_s.append(sb)

        return torch.stack(results_x), torch.stack(results_s)

    @staticmethod
    def _merge_single(
        x: torch.Tensor,
        source: torch.Tensor,
        g_norm: torch.Tensor,
        norms: torch.Tensor,
        r_total: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterative greedy global merge for a single sequence."""
        cur_x = x
        cur_s = source
        cur_gn = g_norm
        cur_norms = norms

        remaining = r_total
        while remaining > 0:
            L_cur = cur_x.shape[0]
            sim = (cur_gn[:-1] * cur_gn[1:]).sum(dim=-1)
            r_step = min(remaining, L_cur // 2)
            if r_step <= 0:
                break

            _, order = sim.sort(descending=True)
            merged_mask = torch.zeros(
                L_cur, dtype=torch.bool, device=x.device
            )
            pairs = []
            for idx in order:
                i = idx.item()
                j = i + 1
                if not merged_mask[i] and not merged_mask[j]:
                    pairs.append((i, j))
                    merged_mask[i] = True
                    merged_mask[j] = True
                    if len(pairs) == r_step:
                        break

            pair_dict = {i: j for i, j in pairs}
            skip = set()
            out_x, out_s, out_gn, out_norms = [], [], [], []
            for k in range(L_cur):
                if k in skip:
                    continue
                if k in pair_dict:
                    jj = pair_dict[k]
                    skip.add(jj)
                    w_i = cur_norms[k]
                    w_j = cur_norms[jj]
                    total = w_i + w_j + 1e-8
                    out_x.append((w_i * cur_x[k] + w_j * cur_x[jj]) / total)
                    out_s.append(cur_s[k] + cur_s[jj])
                    out_gn.append(
                        F.normalize(
                            (w_i * cur_gn[k] + w_j * cur_gn[jj]) / total,
                            dim=-1,
                        )
                    )
                    out_norms.append((w_i + w_j) / 2)
                else:
                    out_x.append(cur_x[k])
                    out_s.append(cur_s[k])
                    out_gn.append(cur_gn[k])
                    out_norms.append(cur_norms[k])

            cur_x = torch.stack(out_x)
            cur_s = torch.stack(out_s)
            cur_gn = torch.stack(out_gn)
            cur_norms = torch.stack(out_norms)
            remaining -= len(pairs)

        return cur_x, cur_s


def token_unmerge(
    z: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Unmerge (upsample) tokens back to original length using the source
    matrix: Z_N = S^T @ Z  (Sec 3.2).

    Args:
        z: (B, L, D) merged token embeddings
        source: (B, L, N) source matrix (binary, rows sum >= 1)

    Returns:
        z_unmerged: (B, N, D) unmerged embeddings
    """
    return torch.bmm(source.transpose(1, 2).float(), z)
