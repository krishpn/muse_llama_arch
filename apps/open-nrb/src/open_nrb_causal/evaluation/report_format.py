"""Share number formatting across the evaluation/*.py report tables.

Used by reviewer_checks.py and power_mde.py so a missing or non-finite value
prints the same "n/a" everywhere. Not run directly -- imported by the other
evaluation/*.py scripts.
"""

from __future__ import annotations

import math


def fmt_num(x: float | None, f: str) -> str:
    """Format `x` with format spec `f`, or "n/a" if it is None or non-finite."""
    return "n/a" if x is None or not math.isfinite(x) else format(x, f)
