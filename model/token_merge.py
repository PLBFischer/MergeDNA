"""
Differentiable token merging and unmerging for MergeDNA.

Implements distance-based locality-constrained token merging (Sec 3.3)
adapted from ToMe (Bolya et al., 2023) with DTEM-style decoupled grouping
embeddings (Lee & Hong, 2024).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMergeModule(nn.Module):
    """Differentiable token merging with distance-based locality constraint.

    Computes pairwise similarity (via grouping projection) between all token
    pairs within a local neighborhood defined by |i - j| < window_size.
    Pair selection is global over all valid local candidates, using
    deterministic tie-breaking (score DESC, i ASC, j ASC) and greedy
    disjoint matching.  No disjoint block partitioning is used.
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
        """Merge *r* token pairs using a distance-based locality constraint.

        Locality constraint is distance-based, not block partitioning:
        token i can merge with token j only if |i - j| < window_size.
        Selection is global over all valid local candidates with
        deterministic tie-breaks (score DESC, i ASC, j ASC).

        Args:
            x: (B, S, D) token embeddings.
            source: (B, S, N_orig) source matrix mapping current tokens to
                    original input positions.
            position_ids: (B, S) absolute position indices for each token.
            r: number of pairs to merge (per layer, global — not per block).
            window_size: locality radius; pair (i, j) is valid iff
                         |i - j| < window_size.

        Returns:
            x_merged: (B, S', D)
            source_merged: (B, S', N_orig)
            position_ids_merged: (B, S')
        """
        B, S, D = x.shape
        if r <= 0 or S <= 1:
            return x, source, position_ids

        r = min(r, S // 2)
        if r <= 0:
            return x, source, position_ids

        # --- Candidate pairs: (i, j) with 0 < j - i < window_size ----------
        # Distance-based locality: every pair within the neighbourhood is
        # scored, not just adjacent tokens.
        idx = torch.arange(S, device=x.device)
        pi_parts, pj_parts = [], []
        for d in range(1, min(window_size, S)):
            pi_parts.append(idx[: S - d])
            pj_parts.append(idx[d:])
        if not pi_parts:
            return x, source, position_ids
        pairs_i = torch.cat(pi_parts)
        pairs_j = torch.cat(pj_parts)

        # --- Grouping projection + cosine similarity for all candidates -----
        g = self.group_proj(x)                                  # (B, S, gd)
        g_norm = F.normalize(g, dim=-1)
        g_norms = g.norm(dim=-1)                                # (B, S)
        sim = (g_norm[:, pairs_i] * g_norm[:, pairs_j]).sum(-1) # (B, P)

        # --- Per-batch greedy selection + simultaneous merge ----------------
        pairs_i_list = pairs_i.tolist()
        pairs_j_list = pairs_j.tolist()

        results_x, results_s, results_p = [], [], []
        for b in range(B):
            xb, sb, pb = self._select_and_merge(
                x[b], source[b], position_ids[b],
                sim[b], pairs_i_list, pairs_j_list,
                g_norms[b], r,
            )
            results_x.append(xb)
            results_s.append(sb)
            results_p.append(pb)

        return (
            torch.stack(results_x),
            torch.stack(results_s),
            torch.stack(results_p),
        )

    @staticmethod
    def _select_and_merge(
        x: torch.Tensor,
        source: torch.Tensor,
        pos: torch.Tensor,
        sim: torch.Tensor,
        pairs_i: list,
        pairs_j: list,
        norms: torch.Tensor,
        r: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy disjoint pair selection + simultaneous merge (single batch).

        Candidates are sorted by (score DESC, i ASC, j ASC) so that
        tie-breaking is fully deterministic.
        """
        sim_list = sim.tolist()

        # Deterministic lexicographic sort: highest score first, then
        # smallest i, then smallest j.
        order = sorted(
            range(len(sim_list)),
            key=lambda p: (-sim_list[p], pairs_i[p], pairs_j[p]),
        )

        # Greedy disjoint matching — each token in at most one pair
        used: set = set()
        selected: list = []
        for p in order:
            i, j = pairs_i[p], pairs_j[p]
            if i not in used and j not in used:
                selected.append((i, j))
                used.add(i)
                used.add(j)
                if len(selected) == r:
                    break

        # Simultaneous merge: keepers are i, removed are j
        pair_dict = {i: j for i, j in selected}
        skip = {j for _, j in selected}
        out_x, out_s, out_p = [], [], []
        S = x.shape[0]

        for k in range(S):
            if k in skip:
                continue
            if k in pair_dict:
                j = pair_dict[k]
                w_i = norms[k]
                w_j = norms[j]
                total = w_i + w_j + 1e-8
                out_x.append((w_i * x[k] + w_j * x[j]) / total)
                out_s.append(source[k] + source[j])
                out_p.append(pos[k])
            else:
                out_x.append(x[k])
                out_s.append(source[k])
                out_p.append(pos[k])

        return torch.stack(out_x), torch.stack(out_s), torch.stack(out_p)


class GlobalTokenMergeModule(nn.Module):
    """Token merging at the global (full-sequence) level.

    Same greedy disjoint matching as TokenMergeModule but without a locality
    constraint: all pairs (i, j) with i < j are candidates.  When
    r_total > floor(S/2) an iterative loop is used, each step selecting
    up to floor(S_cur/2) disjoint pairs in one shot.
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
        """Merge tokens globally to reach *target_len*.

        Args:
            x: (B, L, D)
            source: (B, L, N_orig)
            target_len: K, desired output length

        Returns:
            x_merged: (B, K, D)
            source_merged: (B, K, N_orig)
        """
        B, L, D = x.shape
        r_total = L - target_len
        if r_total <= 0:
            return x, source

        results_x, results_s = [], []
        for b in range(B):
            xb, sb = self._merge_single(x[b], source[b], r_total)
            results_x.append(xb)
            results_s.append(sb)

        return torch.stack(results_x), torch.stack(results_s)

    def _merge_single(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        r_total: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterative greedy global merge for a single sequence.

        Each iteration selects up to floor(S_cur/2) disjoint pairs via the
        same deterministic greedy strategy, then applies merges
        simultaneously.
        """
        cur_x = x
        cur_s = source
        remaining = r_total

        while remaining > 0:
            S = cur_x.shape[0]
            r_step = min(remaining, S // 2)
            if r_step <= 0:
                break

            g = self.group_proj(cur_x)
            g_norm = F.normalize(g, dim=-1)
            g_norms = g.norm(dim=-1)

            # All pairs (i, j) with i < j — no locality constraint
            pairs_i, pairs_j = torch.triu_indices(
                S, S, offset=1, device=cur_x.device,
            )
            sim = (g_norm[pairs_i] * g_norm[pairs_j]).sum(-1)

            pairs_i_list = pairs_i.tolist()
            pairs_j_list = pairs_j.tolist()
            sim_list = sim.tolist()

            order = sorted(
                range(len(sim_list)),
                key=lambda p: (-sim_list[p], pairs_i_list[p], pairs_j_list[p]),
            )

            used: set = set()
            selected: list = []
            for p in order:
                i, j = pairs_i_list[p], pairs_j_list[p]
                if i not in used and j not in used:
                    selected.append((i, j))
                    used.add(i)
                    used.add(j)
                    if len(selected) == r_step:
                        break

            pair_dict = {i: j for i, j in selected}
            skip = {j for _, j in selected}
            out_x, out_s = [], []

            for k in range(S):
                if k in skip:
                    continue
                if k in pair_dict:
                    j = pair_dict[k]
                    w_i = g_norms[k]
                    w_j = g_norms[j]
                    total = w_i + w_j + 1e-8
                    out_x.append(
                        (w_i * cur_x[k] + w_j * cur_x[j]) / total
                    )
                    out_s.append(cur_s[k] + cur_s[j])
                else:
                    out_x.append(cur_x[k])
                    out_s.append(cur_s[k])

            cur_x = torch.stack(out_x)
            cur_s = torch.stack(out_s)
            remaining -= len(selected)

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
