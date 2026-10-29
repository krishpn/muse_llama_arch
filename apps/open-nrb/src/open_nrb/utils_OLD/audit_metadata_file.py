#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.0.0", "pypdf>=5.0.0"]
# ///



"""Full-corpus audit of extract_metadata_from_filename (Fix 2, 2026-10-25): how many real
filenames still come out with month_bs=None and/or year_bs=None after the numeric-pattern +
expanded-abbreviation rewrite. Run against either the raw PDF directory or an already-extracted
CSV's source_file column -- same "audit before trusting at scale" discipline used to catch the
'Liabilities' fallback regression earlier this session.

Usage:
  uv run audit_filename_metadata.py --raw-dir /path/to/raw_pdfs
  uv run audit_filename_metadata.py --csv /path/to/nrb_balance_sheet.csv
"""

import argparse
from pathlib import Path
import pypdf
import pandas as pd

try:
    from .pdf_extract import extract_metadata_from_filename
except ImportError:
    from pdf_extract import extract_metadata_from_filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory of *.pdf files")
    parser.add_argument("--csv", type=Path, default=None, help="CSV with a source_file column")
    args = parser.parse_args()

    if args.raw_dir:
        filenames = sorted(p.name for p in args.raw_dir.glob("*.pdf"))
    elif args.csv:
        filenames = sorted(pd.read_csv(args.csv)["source_file"].dropna().unique().tolist())
    else:
        raise SystemExit("Pass either --raw-dir or --csv")

    print(f"=== auditing extract_metadata_from_filename over {len(filenames)} distinct filenames ===\n")

    no_month, no_year, both_ok = [], [], 0
    for fn in filenames:
        meta = extract_metadata_from_filename(fn)
        if meta["month_bs"] is None:
            no_month.append(fn)
        if meta["year_bs"] is None:
            no_year.append(fn)
        if meta["month_bs"] is not None and meta["year_bs"] is not None:
            both_ok += 1

    print(f"both month_bs and year_bs resolved: {both_ok}/{len(filenames)}")
    print(f"month_bs=None: {len(no_month)}/{len(filenames)}")
    for fn in no_month:
        print(f"  {fn}")
    print(f"year_bs=None: {len(no_year)}/{len(filenames)}")
    for fn in no_year:
        print(f"  {fn}")


if __name__ == "__main__":
    main()
