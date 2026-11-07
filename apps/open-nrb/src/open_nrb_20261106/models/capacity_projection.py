#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///

import torch

try:
    from .config import CapacityProjectionConfig
except ImportError:
    from config import CapacityProjectionConfig


def capacitated_projection(
    W_tilde: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    cap: torch.Tensor,
    config: CapacityProjectionConfig,
) -> torch.Tensor:
    """Algorithm 1 (paper1_network_reconstruction.md Sec 5/11, Eq. 8).

    Projects raw/sampled edge weights onto the intersection of three convex sets via
    Dykstra's alternating-projections algorithm (Boyle & Dykstra, 1986), an ADMM-family
    method that provably converges to the exact Euclidean-nearest point in the
    intersection -- not a heuristic, unlike the earlier Sinkhorn-clip draft (see Sec 5's
    revision note), which collapsed the per-entry cap into a row-sum cap.

        R = {W : row_sums(W) = r}              row marginal
        C = {W : col_sums(W) = c}              column marginal
        B = {W : 0 <= W_ij <= cap_i for all j}  per-row obligor cap, PER ENTRY

    All three per-set projections are closed-form and differentiable, so gradients flow
    through to W_tilde for Algorithm 3's step 10, exactly as Eq. 8 requires.

    Args:
        W_tilde: Raw/sampled weight matrix (N, N), the pre-projection guess.
        r: Observed row marginals (e.g. total interbank lending/out-strength).
        c: Observed column marginals (e.g. total interbank borrowing/in-strength).
        cap: Per-row regulatory single-obligor exposure limit (kappa * K_t); the
            constraint is W_ij <= cap_i for every column j, not a bound on the row sum.
        config: CapacityProjectionConfig (n_iters, tol). n_iters counts full
            row->col->box passes, not single-set applications.

    Returns:
        Weight matrix satisfying row/col marginals and the per-entry cap exactly (up to
        config.tol), when R ∩ C ∩ B is non-empty. If cap is set so tight that even
        maxing out every entry in a row can't reach that row's target (N * cap_i <
        r_i), the intersection is empty and this converges to a least-violation
        compromise instead -- that should be treated as an infeasible-cap data/
        parameter error, not silently accepted (see test_capacity_projection.py's
        infeasible_case).
    """
    n = W_tilde.shape[0]
    W = W_tilde
    I_row = torch.zeros_like(W_tilde)
    I_col = torch.zeros_like(W_tilde)
    I_box = torch.zeros_like(W_tilde)

    zero = torch.zeros_like(cap).unsqueeze(1)
    cap_col = cap.unsqueeze(1)

    for _ in range(config.n_iters):
        W_prev = W

        # 1. Project onto the row-marginal hyperplane {sum_j W_ij = r_i}: shift every
        # entry in row i by that row's residual, split evenly across its n columns.
        Y = W + I_row
        W = Y + ((r - Y.sum(dim=1)) / n).unsqueeze(1)
        I_row = Y - W

        # 2. Project onto the column-marginal hyperplane {sum_i W_ij = c_j}, same idea.
        Y = W + I_col
        W = Y + ((c - Y.sum(dim=0)) / n).unsqueeze(0)
        I_col = Y - W

        # 3. Project onto the per-entry box {0 <= W_ij <= cap_i}: elementwise clamp,
        # cap broadcast per row so every entry in row i is bounded by cap_i.
        Y = W + I_box
        W = torch.minimum(torch.maximum(Y, zero), cap_col)
        I_box = Y - W

        with torch.no_grad():
            converged = torch.max(torch.abs(W - W_prev)).item() < config.tol
        if converged:
            break

    return W
