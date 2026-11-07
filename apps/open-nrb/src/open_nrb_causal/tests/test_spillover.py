#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Tests for network-exposure aggregation and Horvitz-Thompson weighting (spillover.py)."""

import sys
from pathlib import Path

import torch

try:
    from open_nrb_causal.models.spillover import (
        compute_exposure,
        horvitz_thompson_weights,
        stabilized_ht_weights,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from spillover import (  # type: ignore[no-redef]
        compute_exposure,
        horvitz_thompson_weights,
        stabilized_ht_weights,
    )


def compute_exposure_matches_manual_sum_case() -> bool:
    """Assert compute_exposure equals the manual network-weighted dose sum.

    For every bank i and period t the exposure equals the row-sum
    exposure[i, t] = sum_j W_hat[t, i, j] * dose[j, t] (matched to atol 1e-5).
    """
    torch.manual_seed(0)
    n, t = 4, 3
    W_hat = torch.rand(t, n, n)
    dose = torch.rand(n, t)

    exposure = compute_exposure(W_hat, dose)

    manual = torch.zeros(n, t)
    for time in range(t):
        for i in range(n):
            manual[i, time] = sum(W_hat[time, i, j] * dose[j, time] for j in range(n))

    passed = torch.allclose(exposure, manual, atol=1e-5)
    print(f"[compute_exposure_matches_manual_sum_case] max_err="
          f"{(exposure - manual).abs().max().item():.2e} -> {'PASS' if passed else 'FAIL'}")
    return passed


def compute_exposure_shape_mismatch_raises_case() -> bool:
    """Assert compute_exposure raises ValueError when W_hat and dose disagree on N."""
    W_hat = torch.rand(3, 4, 4)
    dose = torch.rand(5, 3)  # wrong N
    try:
        compute_exposure(W_hat, dose)
        passed = False
    except ValueError:
        passed = True
    print(f"[compute_exposure_shape_mismatch_raises_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def ht_weights_are_inverse_propensity_case() -> bool:
    """Assert Horvitz-Thompson weights equal the inverse propensity.

    The weights equal w = 1 / propensity elementwise (matched to atol 1e-6).
    """
    exposure = torch.rand(4, 3)
    propensity = torch.rand(4, 3) + 0.1
    w = horvitz_thompson_weights(exposure, propensity)
    passed = torch.allclose(w, 1.0 / propensity, atol=1e-6)
    print(f"[ht_weights_are_inverse_propensity_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def stabilized_weights_have_unit_mean_case() -> bool:
    """Assert stabilized HT weights have unit mean.

    The stabilized weights satisfy mean(w) == 1 (within 1e-5).
    """
    raw = torch.rand(10, 10) + 0.1
    stabilized = stabilized_ht_weights(raw)
    passed = abs(stabilized.mean().item() - 1.0) < 1e-5
    print(f"[stabilized_weights_have_unit_mean_case] mean={stabilized.mean().item():.4f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def degenerate_propensity_is_clamped_case() -> bool:
    """Assert a degenerate (zero) propensity is eps-clamped to keep weights finite.

    With a zero propensity density, the eps clamp (eps=1e-6) prevents division
    by exactly zero, so every resulting weight stays finite.
    """
    exposure = torch.rand(2, 2)
    propensity = torch.zeros(2, 2)  # degenerate density -- must not divide by exactly 0
    w = horvitz_thompson_weights(exposure, propensity, eps=1e-6)
    passed = bool(torch.isfinite(w).all())
    print(f"[degenerate_propensity_is_clamped_case] finite={bool(torch.isfinite(w).all())} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


CASES = [
    compute_exposure_matches_manual_sum_case,
    compute_exposure_shape_mismatch_raises_case,
    ht_weights_are_inverse_propensity_case,
    stabilized_weights_have_unit_mean_case,
    degenerate_propensity_is_clamped_case,
]


def test_compute_exposure_matches_manual_sum():
    """Assert compute_exposure_matches_manual_sum passes."""
    assert compute_exposure_matches_manual_sum_case()


def test_compute_exposure_shape_mismatch_raises():
    """Assert compute_exposure_shape_mismatch_raises passes."""
    assert compute_exposure_shape_mismatch_raises_case()


def test_ht_weights_are_inverse_propensity():
    """Assert ht_weights_are_inverse_propensity passes."""
    assert ht_weights_are_inverse_propensity_case()


def test_stabilized_weights_have_unit_mean():
    """Assert stabilized_weights_have_unit_mean passes."""
    assert stabilized_weights_have_unit_mean_case()


def test_degenerate_propensity_is_clamped():
    """Assert degenerate_propensity_is_clamped passes."""
    assert degenerate_propensity_is_clamped_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
