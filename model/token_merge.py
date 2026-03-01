from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMergeModule(nn.Module):

    def __init__(self, dim: int, group_dim: int = 64, global_merge: bool = False):
        super().__init__()
        self.group_proj = nn.Linear(dim, group_dim, bias=False)
        self.global_merge = global_merge

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        position_ids: torch.Tensor,
        r: int,
        window_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge *r* token pairs.

        Args:
            x: (B, S, D) token embeddings.
            source: (B, S, N_orig) source matrix mapping current tokens to
                    original input positions.
            position_ids: (B, S) absolute position indices for each token.
            r: number of pairs to merge.

        Returns:
            x_merged: (B, S', D)
            source_merged: (B, S', N_orig)
            position_ids_merged: (B, S')
        """
        B, S, D = x.shape

        # --- Candidate pairs ------------------------------------------------
        if self.global_merge:
            pairs_i, pairs_j = torch.triu_indices(S, S, offset=1, device=x.device)
        else:
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
        g = self.group_proj(x)                                   # (B, S, gd)
        g_norm = F.normalize(g, dim=-1)
        g_norms = g.norm(dim=-1)                                 # (B, S)
        sim = (g_norm[:, pairs_i] * g_norm[:, pairs_j]).sum(-1)  # (B, P)

        # --- Pre-compute tie-breaking order (shared across all batch elems) -
        # Encodes (i, j) as a single integer so one argsort gives (i ASC, j ASC).
        pairs_i_list = pairs_i.tolist()
        pairs_j_list = pairs_j.tolist()
        sec_order = torch.argsort(pairs_i * S + pairs_j, stable=True)

        # --- Per-batch greedy selection + simultaneous merge ----------------
        results_x, results_s, results_p = [], [], []
        for b in range(B):
            xb, sb, pb = self._select_and_merge(
                x[b], source[b], position_ids[b],
                sim[b], pairs_i_list, pairs_j_list,
                g_norms[b], r, sec_order,
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
        sec_order: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy disjoint pair selection + simultaneous merge (single batch).

        Candidates are sorted by (score DESC, i ASC, j ASC) so that
        tie-breaking is fully deterministic.  ``sec_order`` encodes the
        (i ASC, j ASC) tie-breaking permutation and is pre-computed once
        outside the batch loop.
        """
        S = x.shape[0]

        # Stable sort by sim DESC on top of the pre-computed (i, j) order.
        order_list = sec_order[torch.argsort(-sim[sec_order], stable=True)].tolist()

        # Greedy disjoint matching — each token appears in at most one pair.
        # This selection is inherently sequential and cannot be vectorized.
        used: set = set()
        sel_i: list = []
        sel_j: list = []
        for p in order_list:
            i, j = pairs_i[p], pairs_j[p]
            if i not in used and j not in used:
                sel_i.append(i)
                sel_j.append(j)
                used.add(i)
                used.add(j)
                if len(sel_i) == r:
                    break

        if not sel_i:
            return x, source, pos

        keep_i = torch.tensor(sel_i, device=x.device)  # (K,)
        drop_j = torch.tensor(sel_j, device=x.device)  # (K,)

        # Vectorized weighted merge.
        # index_put is out-of-place and differentiable through both x and the
        # merged values, unlike clone() + in-place indexed assignment.
        w_i = norms[keep_i].unsqueeze(-1)               # (K, 1)
        w_j = norms[drop_j].unsqueeze(-1)               # (K, 1)
        merged_x = (w_i * x[keep_i] + w_j * x[drop_j]) / (w_i + w_j + 1e-8)
        x_out = x.index_put((keep_i,), merged_x)

        # source tracks binary provenance — differentiability not needed.
        source_out = source.clone()
        source_out[keep_i] = source[keep_i] + source[drop_j]

        # Drop the absorbed (j) tokens via a boolean mask.
        keep_mask = torch.ones(S, dtype=torch.bool, device=x.device)
        keep_mask[drop_j] = False

        return x_out[keep_mask], source_out[keep_mask], pos[keep_mask]


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
