#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Audit the balance-sheet extraction for digit-fragment corruption (2026-11-03). Read-only.

The diagnose_capital_break dumps showed some bulletin PDFs extract cleanly
(Poush-2078_Publish2.pdf: every identity holds) while others carry digit
fragments -- line items read as 1.0..9.0 where the real value is in the
thousands -- so sub-items no longer sum to their parents. TOTAL rows looked
intact in every dump. Per (bank, period, source_file) sheet, checks:

  liab:    top-level liability items (rows "N <label>", before TOTAL Liabilities) == TOTAL Liabilities
  assets:  top-level asset items (after TOTAL Liabilities, before TOTAL ASSETS) == TOTAL ASSETS
           -- Paper 1's `size` is this item sum, not the TOTAL row
  capital: sub-items a.-g. under 1 CAPITAL FUND == CAPITAL FUND
  balance: TOTAL Liabilities == TOTAL ASSETS
  frag:    count of values that are exactly an integer 1..9

Tolerance: max(1.0, 0.5% of the total). Relies on read_rows preserving the
CSV's within-sheet row order (the diagnose dumps confirm it does). Never
edits Paper 1 data or code; writes results/bs_extraction_audit.csv.

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.evaluation.audit_bs_extraction
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import math
import re
import sys
from pathlib import Path

try:
    from open_nrb.models.data_loader import read_rows
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from models.data_loader import read_rows  # type: ignore[no-redef]

try:
    from ..models.mandate_design import fiscal_year, period_key
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from mandate_design import fiscal_year, period_key  # type: ignore[no-redef]

CSV = Path("src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv")
OUT = Path(__file__).resolve().parent.parent / "results" / "bs_extraction_audit.csv"
TOP_LEVEL = re.compile(r"^(\d+) ")  # "1 CAPITAL FUND", "10 Reconcillation Account"; not "4.1 ..." / "1. ..."
CAPITAL_SUB = re.compile(r"^[a-g]\. ")
CHECKS = ("liab", "assets", "capital", "balance")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, 0.005 * abs(b))


def audit_sheet(rows: list) -> dict:
    """Check the four balance-sheet identities on one extracted sheet.

    `rows` is one (bank, period, source_file) sheet, in CSV order. With
    tolerance |a - b| <= max(1.0, 0.005 |b|), checks:

      liab:    sum of top-level liability items == TOTAL Liabilities
      assets:  sum of top-level asset items == TOTAL ASSETS
      capital: sum of sub-items a.-g. under 1 CAPITAL FUND == CAPITAL FUND
      balance: TOTAL Liabilities == TOTAL ASSETS

    and counts `frag`, the values that are exactly an integer 1..9.
    """
    labels = [r.line_item.strip() for r in rows]
    values = [float(r.value) if r.value is not None else math.nan for r in rows]
    upper = [lb.upper() for lb in labels]
    out: dict = {"frag": sum(1 for v in values if v in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0})}

    totals = [i for i, u in enumerate(upper) if u.startswith("TOTAL")]
    i_tl = next((i for i in totals if "LIABILIT" in upper[i]), totals[0] if len(totals) >= 2 else None)
    i_ta = next((i for i in totals if "ASSET" in upper[i]), totals[-1] if len(totals) >= 2 else None)
    if i_tl is not None and i_ta is not None and i_tl < i_ta:
        layout, liab_end, asset_start, liab_total = "total_rows", i_tl, i_tl + 1, values[i_tl]
    else:
        # Bulletins before Baishakh 2075 have no TOTAL Liabilities row, only TOTAL ASSETS
        # (second audit run, ADBNL Baisakh 2070: "1 CAPITAL FUND | ... | 7 Profit & Loss A/c |
        # 1 LIQUID FUNDS | ... | TOTAL ASSETS"). Assets start where the top-level numbering
        # restarts; liabilities are checked against TOTAL ASSETS, so `balance` is n/a.
        nums = [(i, int(m.group(1))) for i, lb in enumerate(labels[:i_ta or 0]) if (m := TOP_LEVEL.match(lb))]
        reset = next((i for (i, k), (_, k_prev) in zip(nums[1:], nums) if k < k_prev), None)
        if i_ta is None or reset is None:
            top = [lb for lb in labels if TOP_LEVEL.match(lb) or lb.upper().startswith("TOT")]
            return {**out, **{c: None for c in CHECKS}, "assets_gap": math.nan, "layout": "unrecognised",
                    "note": "no TOTAL rows recognised; top-level labels: " + " | ".join(top[:25])}
        layout, liab_end, asset_start, liab_total = "numbering_reset", reset, reset, values[i_ta]
    out["layout"] = layout

    liab_sum = sum(v for lb, v in zip(labels[:liab_end], values[:liab_end]) if TOP_LEVEL.match(lb))
    asset_sum = sum(v for lb, v in zip(labels[asset_start:i_ta], values[asset_start:i_ta]) if TOP_LEVEL.match(lb))
    out["liab"] = _close(liab_sum, liab_total)
    out["assets"] = _close(asset_sum, values[i_ta])
    out["balance"] = _close(values[i_tl], values[i_ta]) if layout == "total_rows" else None
    out["assets_gap"] = (asset_sum - values[i_ta]) / values[i_ta] if values[i_ta] else math.nan

    cap_idx = next((i for i, u in enumerate(upper[:liab_end]) if u.startswith("1 CAPITAL FUND")), None)
    if cap_idx is None:
        out["capital"] = None
    else:
        sub = []
        for lb, v in zip(labels[cap_idx + 1:liab_end], values[cap_idx + 1:liab_end]):
            if TOP_LEVEL.match(lb):
                break
            if CAPITAL_SUB.match(lb):
                sub.append(v)
        out["capital"] = _close(sum(sub), values[cap_idx]) if sub else None
    out["note"] = ""
    return out


def _rate(sheets: list[dict], check: str) -> str:
    vals = [s[check] for s in sheets if s[check] is not None]
    return f"{sum(vals) / len(vals):5.0%}" if vals else "  n/a"


def main() -> None:
    """Audit every sheet in the CSV and write per-sheet flags and summary tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--banks", nargs="*", default=None,
                        help="restrict to these bfi_names (default: all banks in the CSV)")
    args = parser.parse_args()

    groups: dict[tuple, list] = defaultdict(list)
    for r in read_rows(args.csv):
        if args.banks is None or r.bfi_name in args.banks:
            groups[(r.bfi_name, (r.year_bs, r.month_bs), r.source_file)].append(r)

    sheets = []
    for (bank, period, src), rows in groups.items():
        sheets.append({"bank": bank, "period": period, "source_file": src, **audit_sheet(rows)})
    sheets.sort(key=lambda s: (period_key(s["period"]), s["bank"]))

    header = "pass rate:  liab  assets capital balance  mean frag  median |assets gap|"
    by_fy: dict[int, list[dict]] = defaultdict(list)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for s in sheets:
        by_fy[fiscal_year(s["period"])].append(s)
        by_file[s["source_file"]].append(s)

    def _line(group: list[dict]) -> str:
        gaps = sorted(abs(s["assets_gap"]) for s in group if not math.isnan(s["assets_gap"]))
        med_gap = gaps[len(gaps) // 2] if gaps else math.nan
        frag = sum(s["frag"] for s in group) / len(group)
        return (f"{_rate(group, 'liab')}  {_rate(group, 'assets')}  {_rate(group, 'capital')}  "
                f"{_rate(group, 'balance')}  {frag:9.1f}  {med_gap:10.2%}")

    print(f"{len(sheets)} sheets ({len({s['bank'] for s in sheets})} banks, {len(by_file)} source files)")
    print("layout: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(s["layout"] for s in sheets).items()))
          + "  (numbering_reset = no TOTAL Liabilities row; liab checked vs TOTAL ASSETS, balance n/a)\n")
    unchecked = [s for s in sheets if s["note"]]
    if unchecked:
        ex = unchecked[0]
        print(f"{len(unchecked)} sheets not checkable (n/a below). Example {ex['bank']} @ {ex['period']} "
              f"[{ex['source_file']}]:\n  {ex['note']}\n")
    print(f"BY FISCAL YEAR   n    {header}")
    for fy in sorted(by_fy):
        print(f"  FY{fy}  {len(by_fy[fy]):5d}    {' ' * 11}{_line(by_fy[fy])}")

    print(f"\nBY SOURCE FILE (chronological)   n    {header}")
    file_period = {f: min((s["period"] for s in g), key=period_key) for f, g in by_file.items()}
    for f in sorted(by_file, key=lambda f: period_key(file_period[f])):
        print(f"  {f[:45]:<45} {str(file_period[f]):<18} {len(by_file[f]):4d}  {_line(by_file[f])}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bank", "year_bs", "month_bs", "source_file", "layout", *CHECKS, "frag", "assets_gap", "note"])
        for s in sheets:
            w.writerow([s["bank"], s["period"][0], s["period"][1], s["source_file"], s["layout"],
                        *[s[c] for c in CHECKS], s["frag"], s["assets_gap"], s["note"]])
    print(f"\nper-sheet flags written to {OUT}")


if __name__ == "__main__":
    main()
