#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.0.0", "pypdf>=5.0.0","pdfplumber"]
# ///
"""Full-corpus verification of the continuation-page parsing fix (2026-10-25) in
extract_statement_generic: runs the extractor over every real PDF and reports (a) how many
records came from the new block-layout (continuation-page) path vs the original interleaved
path, by bank, and (b) the balance-sheet identity check -- TOTAL Liabilities should equal TOTAL
ASSETS for every (source_file, bfi_name) -- the strongest correctness signal available without a
hand-checked ground-truth file, since it needs no external reference.

Usage: uv run audit_label_value_alignment.py --raw-dir /path/to/raw_pdfs
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from .pdf_extract import NRBFinancialExtractor
except ImportError:
    from pdf_extract import NRBFinancialExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()

    pdf_files = sorted(args.raw_dir.glob("*.pdf"))
    print(f"=== re-extracting balance sheets from {len(pdf_files)} PDFs ===\n")

    all_rows = []
    per_file_bank_counts: list[int] = []
    failed = []
    for pdf_path in pdf_files:
        try:
            extractor = NRBFinancialExtractor(pdf_path)
            df = extractor.extract_statement_generic(
                "statement of assets and liabilities", "balance_sheet"
            )
        except Exception as e:
            failed.append((pdf_path.name, str(e)))
            continue
        if df.empty:
            per_file_bank_counts.append(0)
            continue
        per_file_bank_counts.append(df["bfi_name"].nunique())
        all_rows.append(df)

    if failed:
        print(f"=== {len(failed)} file(s) raised an exception during extraction ===")
        for name, err in failed:
            print(f"  {name}: {err}")
        print()

    if not all_rows:
        print("No records extracted at all -- something is badly broken.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    print(f"=== {len(full)} total rows, {full['bfi_name'].nunique()} distinct bfi_name values ===")
    print(f"distinct files with >=1 bank detected: {sum(1 for c in per_file_bank_counts if c > 0)}/{len(pdf_files)}")
    print(f"mean distinct banks per file: {sum(per_file_bank_counts)/len(per_file_bank_counts):.1f}")
    print(f"max distinct banks in one file: {max(per_file_bank_counts)}")
    print()

    print("=== rows per bank (top 40) ===")
    counts = Counter(full["bfi_name"])
    for bank, n in counts.most_common(40):
        print(f"  {n:6d}  {bank}")
    print()

    print("=== previously under-detected banks: row counts now ===")
    watch = ["Mega", "Global", "Citizen", "Prime", "Sunrise", "NMB", "Prabhu", "ADBNL"]
    for bank in watch:
        n_files = full.loc[full["bfi_name"] == bank, "source_file"].nunique()
        print(f"  {bank:10s} present in {n_files}/{len(pdf_files)} files, {counts.get(bank, 0)} rows")
    print()

    print("=== balance-sheet identity check: TOTAL Liabilities vs TOTAL ASSETS ===")
    tot = full[full["line_item"].str.strip().str.upper().isin(["TOTAL LIABILITIES", "TOTAL ASSETS"])]
    if tot.empty:
        print("  no 'TOTAL Liabilities' / 'TOTAL ASSETS' line items found in extracted data")
    else:
        pivot = tot.pivot_table(
            index=["source_file", "bfi_name"],
            columns=tot["line_item"].str.strip().str.upper(),
            values="value_mn_npr",
            aggfunc="first",
        )
        if "TOTAL LIABILITIES" in pivot.columns and "TOTAL ASSETS" in pivot.columns:
            pivot = pivot.dropna(subset=["TOTAL LIABILITIES", "TOTAL ASSETS"])
            diff = (pivot["TOTAL LIABILITIES"] - pivot["TOTAL ASSETS"]).abs()
            rel = diff / pivot["TOTAL ASSETS"].abs().clip(lower=1e-9)
            mismatches = (rel > 0.001).sum()
            print(f"  {len(pivot)} (file, bank) pairs checked")
            print(f"  mismatches (>0.1% relative diff): {mismatches} ({100*mismatches/max(len(pivot),1):.1f}%)")
            if mismatches:
                print("  worst 10 mismatches:")
                worst = rel.sort_values(ascending=False).head(10)
                for (fname, bank), r in worst.items():
                    print(f"    {fname} / {bank}: {r*100:.2f}% relative diff "
                          f"(L={pivot.loc[(fname, bank), 'TOTAL LIABILITIES']:.2f}, "
                          f"A={pivot.loc[(fname, bank), 'TOTAL ASSETS']:.2f})")
        else:
            print(f"  missing one side; columns found: {list(pivot.columns)}")


if __name__ == "__main__":
    main()
