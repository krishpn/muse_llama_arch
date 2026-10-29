#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///


"""Load the long-format NRB balance-sheet CSV into model tensors (design doc Sec 8).

Produces, per bfi x period: X in R^5 (size, capital ratio, liquidity ratio,
loan-to-deposit, NRB-dependence), the two interbank marginals r/c, and the raw
capital fund K (for the single-obligor cap kappa*K). See Sec 8e for the
equation-symbol <-> raw-field map this mirrors field-for-field.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

"""Find (source_file, bfi_name) groups in the real CSV missing a TOTAL Liabilities
or TOTAL ASSETS marker row -- a missing marker leaves infer_sides's per-group state
machine stuck on one side forever, mis-tagging that group's own later rows and
poisoning the shared `merged` label->side map for every group processed after it.

Usage: uv run diagnose_sides.py path/to/nrb_balance_sheet.csv
"""

import sys
from collections import defaultdict
from pathlib import Path


from data_loader import _TOTAL_ASSETS_RE, _TOTAL_LIABILITIES_RE, read_rows


def main() -> None:
    csv_path = Path(sys.argv[1])
    rows = read_rows(csv_path)

    by_bank: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        by_bank[(r.source_file, r.bfi_name)].append(r)

    missing_liab, missing_assets = [], []
    for key, group in by_bank.items():
        group.sort(key=lambda r: r.order)
        has_liab = any(_TOTAL_LIABILITIES_RE.search(r.line_item) for r in group)
        has_assets = any(_TOTAL_ASSETS_RE.search(r.line_item) for r in group)
        if not has_liab:
            missing_liab.append(key)
        if not has_assets:
            missing_assets.append(key)

    print(f"{len(by_bank)} (file, bank) groups total")
    print(f"{len(missing_liab)} missing a TOTAL LIABILITIES marker row:")
    for key in missing_liab[:15]:
        print(f"  {key}")
    print(f"{len(missing_assets)} missing a TOTAL ASSETS marker row:")
    for key in missing_assets[:15]:
        print(f"  {key}")


if __name__ == "__main__":
    main()

