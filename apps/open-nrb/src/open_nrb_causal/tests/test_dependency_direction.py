#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Guard the one-directional dependency from open_nrb_causal on Paper 1.

open_nrb_causal depends one-directionally on open_nrb/paper1_model -- open_nrb
must never import open_nrb_causal. Scans paper1_model's source tree for any
reference to open_nrb_causal rather than relying on convention alone.
"""

from __future__ import annotations

from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent.parent
# sandbox layout (kris_sandobox/paper1_model) and repo layout (apps/open-nrb/src/open_nrb)
PAPER1_CANDIDATES = [_PARENT / "paper1_model", _PARENT / "open_nrb"]


def no_reverse_import_case() -> bool:
    """Assert no Paper 1 source file references open_nrb_causal.

    Scans every .py file under the Paper 1 roots; passes only when no file
    contains the string "open_nrb_causal" (offenders is empty).
    """
    roots = [p for p in PAPER1_CANDIDATES if p.exists()]
    if not roots:
        print(f"[no_reverse_import_case] FAIL -- Paper 1 source not found at any of {PAPER1_CANDIDATES}")
        return False

    offenders = []
    for root in roots:
        for py_file in root.rglob("*.py"):
            text = py_file.read_text()
            if "open_nrb_causal" in text:
                offenders.append(str(py_file))

    passed = not offenders
    print(f"[no_reverse_import_case] scanned {[str(r) for r in roots]} offenders={offenders} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


CASES = [no_reverse_import_case]


def test_no_reverse_import():
    """Assert no_reverse_import passes."""
    assert no_reverse_import_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
