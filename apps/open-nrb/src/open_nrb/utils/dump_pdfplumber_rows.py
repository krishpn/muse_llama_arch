#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pdfplumber>=0.11.0"]
# ///
"""Diagnostic: verify the hypothesis that pypdf's 'block layout' (labels-then-headers-then-
values, confirmed 2026-10-25 via dump_bank_header_region.py) is purely a text-STREAM-order
artifact, not a real geometric difference -- i.e. that pdfplumber word coordinates show each
label and its corresponding value row sharing the same 'top' (row y-position) regardless of
what order pypdf's extract_text() happened to emit them in.

Groups words into rows by clustering on 'top', then prints each row's reconstructed text (words
sorted by x0) with its top coordinate, for one page known to exhibit block layout.

Usage: uv run dump_pdfplumber_rows.py --pdf /path/to/Asar-2074.pdf --page 18
"""

import argparse
from pathlib import Path

import pdfplumber


def cluster_rows(words, tol: float = 2.5):
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows = []
    for w in words:
        if rows and abs(w["top"] - rows[-1]["top_ref"]) <= tol:
            rows[-1]["words"].append(w)
            rows[-1]["top_ref"] = (rows[-1]["top_ref"] * len(rows[-1]["words"]) + w["top"]) / (len(rows[-1]["words"]) + 1)
        else:
            rows.append({"top_ref": w["top"], "words": [w]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--page", type=int, required=True, help="1-indexed page number")
    args = parser.parse_args()

    with pdfplumber.open(str(args.pdf)) as pdf:
        page = pdf.pages[args.page - 1]
        words = page.extract_words(x_tolerance=1, y_tolerance=1)
        rows = cluster_rows(words)
        print(f"=== page {args.page}: {len(words)} words, {len(rows)} clustered rows ===\n")
        for r in rows:
            ws = sorted(r["words"], key=lambda w: w["x0"])
            text = " | ".join(f"{w['text']}@x{w['x0']:.0f}" for w in ws)
            print(f"  top={r['top_ref']:7.1f}  n={len(ws):2d}  {text}")


if __name__ == "__main__":
    main()
