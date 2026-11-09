#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Estimate a generalized propensity score for continuous exposure.

Sec 3's observational downgrade of Aronow-Samii: no randomized dose here, so
exposure's conditional density given controls stands in for the
randomization-inference exposure probability AS's HT estimator was originally
built around -- Hirano-Imbens 2004's GPS approach, the standard way to adapt
propensity weighting to a continuous (not binary) treatment/exposure.

Model: exposure_i,t | controls_i,t ~ Normal(X @ beta, sigma^2), beta fit by
plain OLS (closed-form normal equations -- no statsmodels/sklearn dependency,
matching this project's existing scipy-only convention). propensity_i,t is
the fitted Normal density at the actually-observed exposure_i,t.
"""

from __future__ import annotations

import torch


def fit_gps(exposure: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the GPS mean model exposure = X @ beta + e by OLS.

    beta = (X'X)^-1 X'y, the ordinary-least-squares normal equations (solved via
    lstsq rather than a manual inverse, stabler under collinear controls); sigma
    is the unbiased residual std, std(y - X @ beta).

    exposure: (N, T) flattened to (N*T,). controls: (N, T, K) flattened to
    (N*T, K), a column of 1s appended for the intercept. Returns (beta, sigma):
    beta is (K+1,), sigma is a scalar residual std.
    """
    n, t = exposure.shape
    y = exposure.reshape(n * t)
    k = controls.shape[-1]
    design = torch.cat([controls.reshape(n * t, k), torch.ones(n * t, 1)], dim=1)

    # beta = (X'X)^-1 X'y via lstsq, not a manual inverse -- stabler if controls
    # are collinear.
    beta = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
    resid = y - design @ beta
    sigma = resid.std(unbiased=True).clamp_min(1e-6)
    return beta, sigma


def gps_density(exposure: torch.Tensor, controls: torch.Tensor, beta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Evaluate the fitted GPS density at the observed exposure.

    Normal(X @ beta, sigma^2) density at the observed exposure, reshaped back to
    (N, T).
    """
    n, t = exposure.shape
    k = controls.shape[-1]
    design = torch.cat([controls.reshape(n * t, k), torch.ones(n * t, 1)], dim=1)
    mean = (design @ beta).reshape(n, t)
    dist = torch.distributions.Normal(mean, sigma)
    return dist.log_prob(exposure).exp()


def estimate_propensity_period_effects(exposure: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    """Estimate the GPS with period intercepts and a per-period residual std.

    exposure_i,t | controls_i,t ~ Normal(X_i,t @ beta + alpha_t, sigma_t^2).

    With time-invariant controls (the frozen pre-mandate design) the pooled model's mean is
    constant over t, so any common trend in exposure (W in nominal NPR mn grows over time)
    pushes whole periods into the density tails and their HT weights explode. alpha_t and
    sigma_t absorb that trend and scale; the weights then reflect only cross-bank variation
    within a period. beta stays pooled: N=16 is too few to fit it per period.
    """
    n, t = exposure.shape
    k = controls.shape[-1]
    period_dummies = torch.eye(t).unsqueeze(0).expand(n, t, t).reshape(n * t, t)
    # One dummy per period spans the intercept, so none is appended.
    design = torch.cat([controls.reshape(n * t, k), period_dummies], dim=1)
    y = exposure.reshape(n * t)
    beta = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
    mean = (design @ beta).reshape(n, t)
    sigma = (exposure - mean).std(dim=0, unbiased=True).clamp_min(1e-6)  # (T,)
    return torch.distributions.Normal(mean, sigma.unsqueeze(0)).log_prob(exposure).exp()


def estimate_propensity(exposure: torch.Tensor, controls: torch.Tensor, period_effects: bool = False) -> torch.Tensor:
    """Estimate the (N, T) GPS propensity density at each observed exposure.

    One-shot fit_gps + gps_density, the propensity density at each unit-period's
    own observed exposure, for spillover.horvitz_thompson_weights.
    period_effects=True: estimate_propensity_period_effects instead of the pooled model.
    """
    if period_effects:
        return estimate_propensity_period_effects(exposure, controls)
    beta, sigma = fit_gps(exposure, controls)
    return gps_density(exposure, controls, beta, sigma)
