#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///

import torch

try:
    from .config import SinkhornConfig
except ImportError:
    from config import SinkhornConfig


def sinkhorn_projection(
    W_tilde: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    config: SinkhornConfig,
) -> torch.Tensor:
    """Uncapped Sinkhorn-Knopp/IPF marginal projection (Sec 5/7) -- the classical
    Gandy-Veraart/RAS maximum-entropy baseline this paper compares against, NOT the
    model's own constraint layer (that's capacitated_projection in
    capacity_projection.py, Sec 11 Algorithm 1). This function takes no cap: an
    earlier draft tried to fold a per-row obligor cap into this via a row-scalar
    `min(...)` clip, but a single row-scalar u_i can only rescale an entire row
    uniformly, so the clip collapsed into capping the row SUM, not each entry --
    the wrong constraint (see Sec 5's revision note). The capped projection now
    lives in its own module instead.

    All ops are differentiable torch ops so gradients flow through to W_tilde.

    Mathematical formulation:
        Kernel, from the raw guess and entropic regularization `config.eps`:
            K_ij = exp( log(W_tilde_ij + delta) / eps )
        Iterative scaling:
            u_new = r / (K v)
            v_new = c / (K^T u_new)

    Args:
        W_tilde: Raw/sampled weight matrix (N, N), the pre-projection guess.
        r: Observed row marginals (e.g. total interbank lending/out-strength).
        c: Observed column marginals (e.g. total interbank borrowing/in-strength).
        config: SinkhornConfig (eps, delta, n_iters, tol).

    Returns:
        Weight matrix satisfying row/column marginals exactly (up to config.tol) --
        this part was always exact; see test_sinkhorn.py.
    """
    # 1. Entropic kernel from the raw weights (Eq. 8)
    kernel = torch.exp(torch.log(W_tilde + config.delta) / config.eps)

    n = kernel.shape[0]
    u = torch.ones(n, dtype=kernel.dtype, device=kernel.device)
    v = torch.ones(n, dtype=kernel.dtype, device=kernel.device)

    # 2. Iterative Sinkhorn scaling
    for _ in range(config.n_iters):
        kv = kernel @ v
        u_new = r / kv
        ktu = kernel.T @ u_new
        v_new = c / ktu

        with torch.no_grad():
            converged = (
                torch.max(torch.abs(u_new - u)).item() < config.tol
                and torch.max(torch.abs(v_new - v)).item() < config.tol
            )
        u, v = u_new, v_new
        if converged:
            break

    # 3. Reconstruct projected matrix: diag(u) @ kernel @ diag(v)
    return torch.diag(u) @ kernel @ torch.diag(v)
