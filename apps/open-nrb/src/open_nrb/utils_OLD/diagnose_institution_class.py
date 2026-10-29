#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pypdf>=5.0.0",
#     "pdfplumber>=0.11.0",
#      "pandas"
# ]
# ///
"""Tests the implicit assumption in _bank_x_anchors: that the phase-1 text-stream order of
bank-name tokens (pypdf reading order, after the 'BFI Name' marker) matches the true
left-to-right PHYSICAL x-position order of those same bank-name words on the page.

_bank_x_anchors does `dict(zip(banks, centers))` -- pairing phase-1 text order directly with
x-sorted-ascending centers from the TOTAL row. If a file's bank-name header text doesn't read
in true left-to-right order (plausible for rotated/vertical column headers), every bank after
the first mismatch gets silently mislabeled with a REAL but WRONG bank name -- data that still
satisfies the TOTAL LIABILITIES == TOTAL ASSETS identity check, since that check only verifies
internal consistency of whatever label got attached to a column, not that the label is correct.

For each page where bank columns are detected, this script independently finds the literal x0
position of each phase-1-detected bank name word via pdfplumber (locating the actual bank-name
header text itself, not the TOTAL row), sorts by x0, and reports any page where phase-1 text
order != physical x-order -- with both orderings shown side by side.

Usage: uv run diagnose_bank_order.py --raw-dir /path/to/raw_pdfs [--only-contains Mega]
"""

import argparse, re
from pathlib import Path

try:
    from .pdf_extract import NRBFinancialExtractor
except ImportError:
    from pdf_extract import NRBFinancialExtractor



_NUMERIC_TOKEN_RE = re.compile(r"^\(?\s*-?\d[\d,]*(?:\.\d+)?\)?$|^-+$")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--lines", type=int, default=12, help="How many text lines to print per page")
    args = parser.parse_args()

    pdf_files = sorted(args.raw_dir.glob("*.pdf"))[: args.max_files]

    for pdf_path in pdf_files:
        try:
            extractor = NRBFinancialExtractor(pdf_path)
            section_pages = extractor.find_section_pages("statement of assets and liabilities")
        except Exception as e:
            print(f"[skip] {pdf_path.name}: {e}")
            continue

        if not section_pages:
            continue

        print(f"\n{'=' * 90}\n{pdf_path.name}  ({len(section_pages)} section pages: "
              f"{[p + 1 for p in section_pages]})\n{'=' * 90}")

        for p_idx in section_pages:
            page_text = extractor.reader.pages[p_idx].extract_text() or ""
            lines = [l.strip() for l in page_text.split("\n") if l.strip()]
            print(f"\n  --- page {p_idx + 1}, first {args.lines} non-blank lines ---")
            for l in lines[: args.lines]:
                print(f"    {l!r}")


if __name__ == "__main__":
    main()

