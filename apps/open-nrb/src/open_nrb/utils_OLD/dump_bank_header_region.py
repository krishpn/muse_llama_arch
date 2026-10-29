#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdf>=5.0.0", "pandas"]
# ///
"""Single-file deep-dive: dump the raw extracted lines around the 'BFI Name' bank-header
region of one PDF's balance-sheet page(s), with line indices, so we can see exactly what
pypdf's extract_text() actually returns there -- vs. what extract_statement_generic's
simple marker-to-marker loop assumes. Use this to find out WHY banks like Global/Citizen/
Prime/Sunrise/NMB/Prabhu never show up as any token in regular (non-NFRS) bulletins.

Usage: uv run dump_bank_header_region.py --pdf /path/to/Asar_2080_Publish.pdf
"""

import argparse
from pathlib import Path

try:
    from .pdf_extract import NRBFinancialExtractor
except ImportError:
    from pdf_extract import NRBFinancialExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()

    extractor = NRBFinancialExtractor(args.pdf)
    section_pages = extractor.find_section_pages("statement of assets and liabilities")
    print(f"=== {args.pdf.name}: {len(section_pages)} balance-sheet page(s) found: {[p+1 for p in section_pages]} ===\n")

    for p_idx in section_pages:
        page_text = extractor.reader.pages[p_idx].extract_text() or ""
        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        print(f"--- page {p_idx + 1}: {len(lines)} non-empty lines ---")
        for i, l in enumerate(lines):
            marker = ""
            if "BFI Name" in l or "BFI name" in l:
                marker = "   <<< BFI Name marker"
            elif any(m in l for m in ["Expenses", "Income", "CAPITAL FUND", "1 CAPITAL FUND", "Total"]):
                marker = "   <<< break marker"
            print(f"  [{i:3d}] {l!r}{marker}")
        print()


if __name__ == "__main__":
    main()