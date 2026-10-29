#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Smoke-test load_panel against the real NRB balance-sheet CSV, using ground-truth values
confirmed by hand via grep on the raw CSV on 2026-10-22 (design doc Sec 9.4): for bank NBL,
period 2076 Asar (source_file Asar_2076-Mid-Jul-2019.pdf), 'a. Interbank Lending' = 6100.0
and 'b. Interbank Borrowing' = 950.0.

Usage: uv run verify_real_csv.py [path/to/nrb_balance_sheet.csv]
"""

import sys
from pathlib import Path

try:
    from open_nrb.models.data_loader import load_panel
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from data_loader import load_panel


def main() -> None:
    default_path = "src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv"
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else default_path)

    panel = load_panel(csv_path)

    print(f"banks={len(panel.bfi_names)} periods={len(panel.periods)}")
    print(f"first period={panel.periods[0]} last period={panel.periods[-1]}")

    i = panel.bfi_names.index("NBL")
    t = panel.periods.index((2076, "Asar"))
    r_val = panel.r[i, t].item()
    c_val = panel.c[i, t].item()
    r_ok = r_val == 6100.0
    c_ok = c_val == 950.0

    print(
        f"NBL 2076 Asar: r={r_val} (expect 6100.0, {'OK' if r_ok else 'MISMATCH'}) "
        f"c={c_val} (expect 950.0, {'OK' if c_ok else 'MISMATCH'})"
    )

    if not (r_ok and c_ok):
        sys.exit(1)
    print("GROUND-TRUTH CHECK: PASS")


if __name__ == "__main__":
    main()
