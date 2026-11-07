#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Diagnose the capital_ratio break (2026-11-03).

Median CAPITAL FUND / size falls from ~0.08-0.15 to ~0.001-0.05 for most banks from
FY2080, and a few banks jump to ~0.5 in FY2077. Splits the ratio into
numerator and denominator by fiscal year, then dumps the raw balance-sheet
rows for one bank either side of each break, to tell an extraction/label
change from a real movement.

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.evaluation.diagnose_capital_break [--bank EBL]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from open_nrb.models.data_loader import read_rows
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from models.data_loader import read_rows  # type: ignore[no-redef]

try:
    from ..models.mandate_design import bank_size, fiscal_year
    from ..models.panel_alignment import capital_ratio, load_commercial_panel
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from mandate_design import bank_size, fiscal_year  # type: ignore[no-redef]
    from panel_alignment import capital_ratio, load_commercial_panel  # type: ignore[no-redef]

CSV = Path("src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv")
# Either side of the FY2077 jump and the FY2080 collapse.
COMPARE = [(2077, "Asar"), (2078, "Poush"), (2080, "Asar"), (2081, "Poush")]


def main() -> None:
    """Dump the capital-ratio split by fiscal year and raw rows around each break."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="EBL")
    parser.add_argument("--csv", type=Path, default=CSV)
    args = parser.parse_args()

    p = load_commercial_panel(args.csv)
    size = bank_size(p)
    cap = capital_ratio(p) * size
    i = p.bfi_names.index(args.bank)
    print(f"{args.bank}: FY  median size (NPR mn)  median CAPITAL FUND  ratio")
    for fy in sorted({fiscal_year(q) for q in p.periods}):
        idx = [t for t, q in enumerate(p.periods) if fiscal_year(q) == fy]
        s, c = size[i, idx].nanmedian().item(), cap[i, idx].nanmedian().item()
        print(f"  FY{fy}  {s:>14,.0f}  {c:>14,.0f}  {c / s if s else float('nan'):.3f}")

    rows = [r for r in read_rows(args.csv)
            if r.bfi_name == args.bank and (r.year_bs, r.month_bs) in COMPARE]
    for period in COMPARE:
        print(f"\n--- raw rows {args.bank} @ {period} ---")
        for r in rows:
            if (r.year_bs, r.month_bs) == period:
                print(f"  {r.line_item[:70]:<70} {r.value:>14,.1f}  [{r.source_file}]")


if __name__ == "__main__":
    main()
