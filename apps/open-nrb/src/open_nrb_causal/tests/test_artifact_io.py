#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Tests for loading and validating a saved Paper 2 artifact file (artifact_io.py)."""

import sys
import tempfile
from pathlib import Path

import torch

try:
    from open_nrb_causal.models.artifact_io import load_artifact
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from artifact_io import load_artifact  # type: ignore[no-redef]


def _write_fake_artifact(path: Path, *, extra_keys: bool = True) -> None:
    """Write a fake saved-artifact dict to `path`, optionally including extra non-schema keys."""
    n, t = 3, 2
    data = {
        "bfi_names": ["A", "B", "C"],
        "periods": [(2080, "Asar"), (2080, "Shrawan")],
        "W_hat": torch.rand(t, n, n),
        "W_baseline": torch.rand(t, n, n),
        "q": torch.rand(t, n, n),
        "cap": torch.rand(n, t),
        "r": torch.rand(n, t),
        "c": torch.rand(n, t),
        "losses": [1.0, 0.5],
        "components_trend": {},
        "non_finite_periods": [],
        "degenerate_periods": [],
        "model_stability": [],
        "baseline_stability": [],
        "baseline_diff": [],
        "seed": 0,
    }
    if not extra_keys:
        for k in ("losses", "components_trend", "non_finite_periods", "degenerate_periods",
                  "model_stability", "baseline_stability", "baseline_diff", "q", "cap", "r", "c"):
            data.pop(k, None)
    torch.save(data, path)


def extra_keys_dont_break_loading_case() -> bool:
    """Assert load_artifact ignores extra keys in the saved dict.

    The real save dict has more keys than Paper2Artifact declares -- exactly the
    Paper1Artifact(**data) bug the plan called out, now fixed via explicit field
    pulls; loading still returns the declared fields intact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fake.pt"
        _write_fake_artifact(path, extra_keys=True)
        artifact = load_artifact(path)
        passed = (
            artifact.bfi_names == ["A", "B", "C"]
            and artifact.periods == [(2080, "Asar"), (2080, "Shrawan")]
            and artifact.W_hat.shape == (2, 3, 3)
            and artifact.seed == 0
        )
    print(f"[extra_keys_dont_break_loading_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def missing_required_key_raises_case() -> bool:
    """Assert load_artifact raises KeyError when a required key is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fake.pt"
        torch.save({"bfi_names": ["A"], "periods": [(2080, "Asar")]}, path)
        try:
            load_artifact(path)
            passed = False
        except KeyError:
            passed = True
    print(f"[missing_required_key_raises_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def shape_mismatch_raises_case() -> bool:
    """Assert load_artifact raises ValueError when W_hat's shape disagrees with the bank count."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fake.pt"
        data = {
            "bfi_names": ["A", "B"],
            "periods": [(2080, "Asar")],
            "W_hat": torch.rand(1, 3, 3),  # 3x3 but only 2 bank names
            "W_baseline": torch.rand(1, 3, 3),
            "seed": 0,
        }
        torch.save(data, path)
        try:
            load_artifact(path)
            passed = False
        except ValueError:
            passed = True
    print(f"[shape_mismatch_raises_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


CASES = [
    extra_keys_dont_break_loading_case,
    missing_required_key_raises_case,
    shape_mismatch_raises_case,
]


def test_extra_keys_dont_break_loading():
    """Assert extra_keys_dont_break_loading passes."""
    assert extra_keys_dont_break_loading_case()


def test_missing_required_key_raises():
    """Assert missing_required_key_raises passes."""
    assert missing_required_key_raises_case()


def test_shape_mismatch_raises():
    """Assert shape_mismatch_raises passes."""
    assert shape_mismatch_raises_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
