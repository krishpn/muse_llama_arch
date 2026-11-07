#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Test the eda_model_checks network-structure metrics on hand-built graphs."""

import sys
from pathlib import Path

import torch

try:
    from open_nrb.evaluation.eda_model_checks import (
        debtrank,
        density,
        eigenvector_centrality,
        marginal_preservation_error,
        rank_correlation,
        weighted_reciprocity,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
    from eda_model_checks import (
        debtrank,
        density,
        eigenvector_centrality,
        marginal_preservation_error,
        rank_correlation,
        weighted_reciprocity,
    )


def weighted_reciprocity_symmetric_case() -> bool:
    """Assert weighted reciprocity is exactly 1.0 for fully symmetric off-diagonal weights.

    Fully symmetric off-diagonal weights -- min(w_ij, w_ji) == w_ij everywhere, so
    reciprocity should be exactly 1.0.
    """
    W = torch.tensor([[[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]]])
    valid = torch.tensor([True])
    result = weighted_reciprocity(W, valid)
    passed = abs(result["mean"] - 1.0) < 1e-6
    print(f"[weighted_reciprocity_symmetric_case] reciprocity={result['mean']:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def weighted_reciprocity_one_directional_case() -> bool:
    """Assert weighted reciprocity is exactly 0.0 when all flows are one-directional.

    All lending flows one direction only (upper triangle) -- min(w_ij, w_ji) == 0
    everywhere off-diagonal, so reciprocity should be exactly 0.0.
    """
    W = torch.tensor([[[0.0, 2.0, 3.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    valid = torch.tensor([True])
    result = weighted_reciprocity(W, valid)
    passed = abs(result["mean"] - 0.0) < 1e-6
    print(f"[weighted_reciprocity_one_directional_case] reciprocity={result['mean']:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def marginal_preservation_exact_case() -> bool:
    """Assert marginal-preservation error is 0 when targets equal W's own row/col sums.

    Targets built from W's own row/col sums -- error should be exactly 0.
    """
    torch.manual_seed(0)
    W = torch.rand(1, 5, 5)
    r_target = W.sum(dim=2)
    c_target = W.sum(dim=1)
    valid = torch.tensor([True])
    result = marginal_preservation_error(W, r_target, c_target, valid)
    passed = result["row"]["mean"] < 1e-6 and result["col"]["mean"] < 1e-6
    print(
        f"[marginal_preservation_exact_case] row_err={result['row']['mean']:.2e} "
        f"col_err={result['col']['mean']:.2e} -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def density_known_fraction_case() -> bool:
    """Assert density is exactly 0.5 when 6 of 12 off-diagonal cells exceed tol.

    4x4 matrix, 12 off-diagonal cells, exactly 6 set above tol -- density should be 0.5.
    """
    W = torch.zeros(1, 4, 4)
    W[0, 0, 1] = W[0, 0, 2] = W[0, 0, 3] = 1.0
    W[0, 1, 0] = W[0, 1, 2] = W[0, 1, 3] = 1.0
    valid = torch.tensor([True])
    result = density(W, valid, tol=1e-3)
    passed = abs(result["mean"] - 0.5) < 1e-6
    print(f"[density_known_fraction_case] density={result['mean']:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def eigenvector_centrality_star_graph_case() -> bool:
    """Assert the hub has strictly the highest eigenvector centrality in a star graph.

    Star graph: node 0 is the hub, connected to all others, which have no other edges
    -- the hub must have strictly the highest centrality score.
    """
    n = 5
    W = torch.zeros(1, n, n)
    for i in range(1, n):
        W[0, 0, i] = W[0, i, 0] = 1.0
    scores = eigenvector_centrality(W)
    hub_score = scores[0, 0].item()
    passed = all(hub_score > scores[0, i].item() for i in range(1, n))
    print(f"[eigenvector_centrality_star_graph_case] hub_score={hub_score:.4f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def rank_correlation_identical_case() -> bool:
    """Assert Spearman rank correlation of a vector with itself is exactly 1.0."""
    a = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0])
    corr = rank_correlation(a, a)
    passed = abs(corr - 1.0) < 1e-6
    print(f"[rank_correlation_identical_case] corr={corr:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def rank_correlation_reversed_case() -> bool:
    """Assert Spearman rank correlation of a vector with its reverse is exactly -1.0."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    b = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    corr = rank_correlation(a, b)
    passed = abs(corr - (-1.0)) < 1e-6
    print(f"[rank_correlation_reversed_case] corr={corr:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def debtrank_isolated_bank_case() -> bool:
    """Assert DebtRank total impact is exactly 1.0 when shocking a bank with zero exposures.

    Shocking a bank with zero exposure to anyone else should leave total impact at
    exactly 1.0 (just the shocked bank itself, no propagation).
    """
    W = torch.zeros(3, 3)
    equity = torch.ones(3)
    impact = debtrank(W, equity, shocked=0)
    passed = abs(impact - 1.0) < 1e-6
    print(f"[debtrank_isolated_bank_case] impact={impact:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def debtrank_full_wipeout_case() -> bool:
    """Assert DebtRank total impact is exactly 2.0 when the shock fully wipes out the creditor.

    Bank 1 lends exactly its own equity to bank 0; bank 0 defaulting wipes out all of
    bank 1's equity (A[1, 0] = 1.0 exactly) -- total impact should be exactly 2.0
    (both banks at h=1).
    """
    W = torch.tensor([[0.0, 0.0], [5.0, 0.0]])  # row=lending, col=borrowing: bank 1 lends 5 to bank 0
    equity = torch.tensor([1.0, 5.0])
    impact = debtrank(W, equity, shocked=0)
    passed = abs(impact - 2.0) < 1e-6
    print(f"[debtrank_full_wipeout_case] impact={impact:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_weighted_reciprocity_symmetric():
    """Assert weighted_reciprocity_symmetric_case passes."""
    assert weighted_reciprocity_symmetric_case()


def test_weighted_reciprocity_one_directional():
    """Assert weighted_reciprocity_one_directional_case passes."""
    assert weighted_reciprocity_one_directional_case()


def test_marginal_preservation_exact():
    """Assert marginal_preservation_exact_case passes."""
    assert marginal_preservation_exact_case()


def test_density_known_fraction():
    """Assert density_known_fraction_case passes."""
    assert density_known_fraction_case()


def test_eigenvector_centrality_star_graph():
    """Assert eigenvector_centrality_star_graph_case passes."""
    assert eigenvector_centrality_star_graph_case()


def test_rank_correlation_identical():
    """Assert rank_correlation_identical_case passes."""
    assert rank_correlation_identical_case()


def test_rank_correlation_reversed():
    """Assert rank_correlation_reversed_case passes."""
    assert rank_correlation_reversed_case()


def test_debtrank_isolated_bank():
    """Assert debtrank_isolated_bank_case passes."""
    assert debtrank_isolated_bank_case()


def test_debtrank_full_wipeout():
    """Assert debtrank_full_wipeout_case passes."""
    assert debtrank_full_wipeout_case()


if __name__ == "__main__":
    results = {
        "weighted_reciprocity_symmetric_case": weighted_reciprocity_symmetric_case(),
        "weighted_reciprocity_one_directional_case": weighted_reciprocity_one_directional_case(),
        "marginal_preservation_exact_case": marginal_preservation_exact_case(),
        "density_known_fraction_case": density_known_fraction_case(),
        "eigenvector_centrality_star_graph_case": eigenvector_centrality_star_graph_case(),
        "rank_correlation_identical_case": rank_correlation_identical_case(),
        "rank_correlation_reversed_case": rank_correlation_reversed_case(),
        "debtrank_isolated_bank_case": debtrank_isolated_bank_case(),
        "debtrank_full_wipeout_case": debtrank_full_wipeout_case(),
    }
    print("\n=== SUMMARY (eda_model_checks.py network-structure metrics) ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
