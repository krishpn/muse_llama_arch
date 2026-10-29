#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdf>=5.0.0", "pandas", "pdfplumber"]
# ///
"""Full-corpus audit of the REAL column-header tokens NRB bulletins actually use in the
'BFI Name' row-of-banks region, independent of KNOWN_BFIS -- so KNOWN_BFIS mismatches (like
Everest vs EBL, and likely Mega vs MBL) can be found by comparison instead of guessed from
general banking-abbreviation knowledge.

Replicates extract_statement_generic's bank-token-collection loop (the "BFI Name" marker ->
break-marker region) but records EVERY non-empty token seen there, whether or not it's in
KNOWN_BFIS, with its frequency across the corpus and a sample of which files/pages it came from.

Usage: uv run audit_bank_tokens.py --raw-dir /path/to/raw_pdfs
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pypdf

try:
    from .pdf_extract import NRBFinancialExtractor
except ImportError:
    from pdf_extract import NRBFinancialExtractor


def tokens_in_file(pdf_path: Path) -> set[str]:
    extractor = NRBFinancialExtractor(pdf_path)
    section_pages = extractor.find_section_pages("statement of assets and liabilities")
    found: set[str] = set()
    for p_idx in section_pages:
        try:
            page_text = extractor.reader.pages[p_idx].extract_text() or ""
            lines = [l.strip() for l in page_text.split("\n") if l.strip()]
            in_banks = False
            for l in lines:
                if "BFI Name" in l or "BFI name" in l:
                    in_banks = True
                    continue
                if in_banks:
                    if any(marker in l for marker in ["Expenses", "Income", "CAPITAL FUND", "1 CAPITAL FUND", "Total"]):
                        break
                    token = l.strip()
                    if token:
                        found.add(token)
        except Exception:
            continue
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()

    pdf_files = sorted(args.raw_dir.glob("*.pdf"))
    print(f"=== auditing raw bank-column tokens over {len(pdf_files)} PDFs ===\n")

    freq: Counter[str] = Counter()
    sample_files: dict[str, str] = {}
    for pdf_path in pdf_files:
        try:
            toks = tokens_in_file(pdf_path)
        except Exception as e:
            print(f"  [skip] {pdf_path.name}: {e}")
            continue
        for t in toks:
            freq[t] += 1
            sample_files.setdefault(t, pdf_path.name)

    try:
        from pdf_extract import NRBFinancialExtractor as _E
        known = _E.KNOWN_BFIS
    except Exception:
        known = set()

    print(f"=== {len(freq)} distinct raw tokens seen, by frequency ===")
    for tok, n in freq.most_common():
        flag = "" if tok in known else "  <-- NOT in KNOWN_BFIS"
        print(f"  {n:4d}/{len(pdf_files)}  {tok!r:30s} e.g. {sample_files[tok]}{flag}")

    print("\n=== KNOWN_BFIS entries that NEVER appear as a raw token (likely noise, like 'Everest') ===")
    for tok in sorted(known):
        if tok not in freq:
            print(f"  {tok}")


if __name__ == "__main__":
    main()
