#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Tests for the generalized propensity score and propensity density (propensity.py)."""

import sys
from pathlib import Path

import torch

try:
    from open_nrb_causal.models.propensity import estimate_propensity, fit_gps
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from propensity import estimate_propensity, fit_gps  # type: ignore[no-redef]


def fit_gps_recovers_known_beta_case() -> bool:
    """Assert fit_gps recovers a known beta from a near-noiseless linear exposure.

    Exposure is a known linear function of the controls plus an intercept and
    tiny noise; the fitted beta matches true_beta elementwise within
    max|beta - true_beta| < 0.1.
    """
    torch.manual_seed(0)
    n, t, k = 50, 10, 2
    true_beta = torch.tensor([1.5, -2.0, 0.5])  # last entry is the intercept
    controls = torch.randn(n, t, k)
    design = torch.cat([controls.reshape(-1, k), torch.ones(n * t, 1)], dim=1)
    exposure = (design @ true_beta).reshape(n, t) + torch.randn(n, t) * 0.01

    beta, sigma = fit_gps(exposure, controls)

    err = (beta - true_beta).abs().max().item()
    passed = err < 0.1
    print(f"[fit_gps_recovers_known_beta_case] beta={beta.tolist()} err={err:.4f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def estimate_propensity_is_positive_and_finite_case() -> bool:
    """Assert estimate_propensity returns strictly positive, finite densities.

    Every entry of the returned propensity satisfies p > 0 and is finite.
    """
    torch.manual_seed(1)
    n, t, k = 8, 5, 3
    controls = torch.randn(n, t, k)
    exposure = torch.randn(n, t)
    propensity = estimate_propensity(exposure, controls)
    passed = bool((propensity > 0).all() and torch.isfinite(propensity).all())
    print(f"[estimate_propensity_is_positive_and_finite_case] min={propensity.min().item():.2e} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def estimate_propensity_peaks_near_predicted_mean_case() -> bool:
    """Assert estimate_propensity peaks near the fitted regression mean.

    A point sitting exactly on the fitted regression line gets a higher density
    than one shifted far off it, for the same controls row: mean density
    on-line > mean density far-off.
    """
    torch.manual_seed(2)
    n, t, k = 30, 8, 2
    controls = torch.randn(n, t, k)
    true_beta = torch.tensor([1.0, -1.0, 0.0])
    design = torch.cat([controls.reshape(-1, k), torch.ones(n * t, 1)], dim=1)
    on_line = (design @ true_beta).reshape(n, t)
    far_off = on_line + 10.0

    propensity_on_line = estimate_propensity(on_line, controls)
    propensity_far_off = estimate_propensity(far_off, controls)

    passed = bool((propensity_on_line.mean() > propensity_far_off.mean()).item())
    print(f"[estimate_propensity_peaks_near_predicted_mean_case] on_line_mean="
          f"{propensity_on_line.mean().item():.4f} far_off_mean="
          f"{propensity_far_off.mean().item():.4f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def period_effects_ignore_common_trend_case() -> bool:
    """Assert period-effects propensity is invariant to a common per-period shift.

    Adding any per-period shift c_t to every bank's exposure (W's nominal
    growth) leaves the period-effects propensity unchanged (allclose), while the
    pooled model's propensity moves.
    """
    torch.manual_seed(3)
    n, t, k = 16, 20, 2
    controls = torch.randn(n, 1, k).expand(n, t, k)  # frozen pre-mandate covariates
    exposure = controls[..., 0] - controls[..., 1] + 0.5 * torch.randn(n, t)
    shift = torch.exp(torch.linspace(0.0, 3.0, t)).unsqueeze(0)
    base = estimate_propensity(exposure, controls, period_effects=True)
    shifted = estimate_propensity(exposure + shift, controls, period_effects=True)
    pooled_moves = not torch.allclose(estimate_propensity(exposure, controls),
                                      estimate_propensity(exposure + shift, controls), rtol=1e-2)
    passed = bool(torch.allclose(base, shifted, rtol=1e-3, atol=1e-6)) and pooled_moves
    print(f"[period_effects_ignore_common_trend_case] max diff={(base - shifted).abs().max().item():.2e} "
          f"pooled_moves={pooled_moves} -> {'PASS' if passed else 'FAIL'}")
    return passed


CASES = [
    fit_gps_recovers_known_beta_case,
    estimate_propensity_is_positive_and_finite_case,
    estimate_propensity_peaks_near_predicted_mean_case,
    period_effects_ignore_common_trend_case,
]


def test_fit_gps_recovers_known_beta():
    """Assert fit_gps_recovers_known_beta passes."""
    assert fit_gps_recovers_known_beta_case()


def test_estimate_propensity_is_positive_and_finite():
    """Assert estimate_propensity_is_positive_and_finite passes."""
    assert estimate_propensity_is_positive_and_finite_case()


def test_estimate_propensity_peaks_near_predicted_mean():
    """Assert estimate_propensity_peaks_near_predicted_mean passes."""
    assert estimate_propensity_peaks_near_predicted_mean_case()


def test_period_effects_ignore_common_trend():
    """Assert period_effects_ignore_common_trend passes."""
    assert period_effects_ignore_common_trend_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
