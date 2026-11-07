#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Tests for aligning a raw Panel to a Paper 2 artifact (panel_alignment.py)."""

import sys
from pathlib import Path

import torch

try:
    from open_nrb.models.data_loader import Panel
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from models.data_loader import Panel  # type: ignore[no-redef]

try:
    from open_nrb_causal.models.artifact_io import Paper2Artifact
    from open_nrb_causal.models.panel_alignment import align_to_artifact, capital_ratio
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from artifact_io import Paper2Artifact  # type: ignore[no-redef]
    from panel_alignment import align_to_artifact, capital_ratio  # type: ignore[no-redef]


def _make_panel(names: list[str], periods: list[tuple[int, str]]) -> Panel:
    """Build a Panel fixture with sequential X and random r/c/K for the given names/periods."""
    n, t = len(names), len(periods)
    return Panel(
        bfi_names=names,
        periods=periods,
        X=torch.arange(n * t * 5, dtype=torch.float32).reshape(n, t, 5),
        r=torch.rand(n, t),
        c=torch.rand(n, t),
        K=torch.rand(n, t),
    )


def align_subsets_and_reorders_case() -> bool:
    """Assert align_to_artifact reindexes a panel to match the artifact exactly.

    The artifact's bank/period order may differ from (and be a strict subset of)
    the raw panel's own order; align_to_artifact must reindex to match the
    artifact exactly.
    """
    panel = _make_panel(["A", "B", "C"], [(2080, "Asar"), (2080, "Shrawan")])
    artifact = Paper2Artifact(
        bfi_names=["C", "A"],  # reordered subset
        periods=[(2080, "Shrawan")],  # subset of periods
        W_hat=torch.zeros(1, 2, 2),
        W_baseline=torch.zeros(1, 2, 2),
        seed=0,
    )
    aligned = align_to_artifact(panel, artifact)

    c_idx, a_idx = 2, 0
    t_idx = 1  # Shrawan
    expected = panel.X[torch.tensor([c_idx, a_idx])][:, torch.tensor([t_idx]), :]
    passed = (
        aligned.bfi_names == ["C", "A"]
        and aligned.periods == [(2080, "Shrawan")]
        and torch.equal(aligned.X, expected)
    )
    print(f"[align_subsets_and_reorders_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def align_missing_bank_raises_case() -> bool:
    """Assert align_to_artifact raises ValueError when the artifact names a bank absent from the panel."""
    panel = _make_panel(["A", "B"], [(2080, "Asar")])
    artifact = Paper2Artifact(
        bfi_names=["A", "Z"],  # "Z" not in panel
        periods=[(2080, "Asar")],
        W_hat=torch.zeros(1, 2, 2),
        W_baseline=torch.zeros(1, 2, 2),
        seed=0,
    )
    try:
        align_to_artifact(panel, artifact)
        passed = False
    except ValueError:
        passed = True
    print(f"[align_missing_bank_raises_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def capital_ratio_reads_covariate_order_case() -> bool:
    """Assert capital_ratio reads the capital_ratio column from the panel's covariate order."""
    panel = _make_panel(["A"], [(2080, "Asar")])
    panel.X[0, 0, 1] = 0.42  # index 1 == COVARIATE_ORDER.index("capital_ratio")
    ratio = capital_ratio(panel)
    passed = abs(ratio[0, 0].item() - 0.42) < 1e-6
    print(f"[capital_ratio_reads_covariate_order_case] ratio={ratio[0, 0].item():.4f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


CASES = [
    align_subsets_and_reorders_case,
    align_missing_bank_raises_case,
    capital_ratio_reads_covariate_order_case,
]


def test_align_subsets_and_reorders():
    """Assert align_subsets_and_reorders passes."""
    assert align_subsets_and_reorders_case()


def test_align_missing_bank_raises():
    """Assert align_missing_bank_raises passes."""
    assert align_missing_bank_raises_case()


def test_capital_ratio_reads_covariate_order():
    """Assert capital_ratio_reads_covariate_order passes."""
    assert capital_ratio_reads_covariate_order_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
