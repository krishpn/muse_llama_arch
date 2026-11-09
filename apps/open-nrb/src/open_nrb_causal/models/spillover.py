#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Compute Sec 4c spillover exposure and full Aronow-Samii/Horvitz-Thompson weights.

Full AS weight, not a simplified/truncated variant (design decision).

exposure_i,t = sum_j W_hat_ij,t * dose_j,t literally matches Sec 4c's
equation -- an unnormalized, weighted sum of neighbors' dose, not a
degree-normalized average.
"""

from __future__ import annotations

import torch


def compute_exposure(W_hat: torch.Tensor, dose: torch.Tensor) -> torch.Tensor:
    """Compute spillover exposure, the weighted sum of neighbours' dose.

    W_hat: (T, N, N) with W_hat[t, i, j] = exposure of i to j at t. dose: (N, T).
    Returns (N, T) exposure_i,t = sum_j W_hat[t, i, j] * dose[j, t].
    """
    t_dim, n_rows, n_cols = W_hat.shape
    n, t = dose.shape
    if t_dim != t or n_rows != n or n_cols != n:
        raise ValueError(f"shape mismatch: W_hat={tuple(W_hat.shape)}, dose={tuple(dose.shape)}")
    # einsum over (t, i, j), (j, t) -> (i, t)
    return torch.einsum("tij,jt->it", W_hat, dose)


def horvitz_thompson_weights(
    exposure: torch.Tensor, propensity: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Full (unnormalized) Horvitz-Thompson re-weighting: w_i,t = 1 / propensity_i,t.

    propensity is the generalized-propensity-score estimate of the density of
    exposure_i,t given controls, from propensity.py -- this function only
    does the HT division + the degenerate-density guard (eps), it does not
    estimate the propensity itself.
    """
    if exposure.shape != propensity.shape:
        raise ValueError(f"shape mismatch: exposure={tuple(exposure.shape)}, propensity={tuple(propensity.shape)}")
    return 1.0 / propensity.clamp_min(eps)


def stabilized_ht_weights(ht_weights: torch.Tensor) -> torch.Tensor:
    """Stabilize HT weights by dividing by their mean.

    w_i,t / mean(w): the standard IPW stabilization (Aronow & Samii 2017 Sec 5;
    also the usual fix for HT weight variance blowing up near-degenerate
    propensities).
    """
    return ht_weights / ht_weights.mean()
