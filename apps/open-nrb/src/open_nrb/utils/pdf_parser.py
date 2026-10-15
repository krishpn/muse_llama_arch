#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pypdf>=5.0.0",
#     "pdfplumber>=0.11.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///

"""
NRB Financial Statement Extractor
Extracts Balance Sheets, P&L, and Statement of Credit tables from Nepal
Rastra Bank monthly statistical bulletin PDFs.

Per-institution tables in these bulletins render the BFI name column
headers as 90-degree rotated text. pypdf's layout-mode text extraction
reconstructs row/column alignment for normal text reliably, but silently
drops rotated text. So this extractor uses two libraries for two jobs:
  - pdfplumber character coordinates to recover the rotated bank-name
    header row (via rotation-matrix detection + x0 column clustering).
  - pypdf layout-mode text to extract the actual data rows (label +
    aligned numeric columns), which pdfplumber's own word tokenizer
    corrupts in this document (overlapping/invisible space characters
    split single numbers into two words).
"""

import argparse
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Any

import pandas as pd
import pdfplumber
import pypdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nrb_extractor")

NEPALI_MONTHS = [
    "Baisakh", "Baishakh", "Jestha", "Asar", "Ashar", "Shrawan", "Sawan",
    "Bhadra", "Asoj", "Ashwin", "Aswin", "Kartik", "Mangshir", "Poush",
    "Magh", "Falgun", "Chaitra",
]

NUMERIC_TOKEN_RE = re.compile(r"^\(-?[\d,]+\.?\d*\)$|^-?[\d,]+\.?\d*$|^-+$")

# Leading numeric/alphabetic hierarchy marker on a line-item label, e.g.
# "1", "1.1", "1.1.2.1", "7.5.", "a.", "c." -- these encode the
# statement's parent/child item structure (1 -> 1.1 -> 1.1.1 -> a./b./c.)
# and are worth keeping as structured data even if the label text itself
# gets reworded across years.
HIERARCHY_RE = re.compile(r"^(\d+(?:\.\d+)*\.?|[a-zA-Z]\.)\s+")


def clean_number(val: Any) -> Optional[float]:
    """Parse raw string from NRB table into a float.

    Handles negatives in parentheses e.g. '(2,816.02)' and the accounting
    dash '-' as 0.0. A genuinely blank cell (no token at all) is returned
    as None/NaN rather than 0.0, since "not disclosed" and "zero" are
    different things for the downstream zero-inflation model.
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    if s in ["-", "–", "—", "0", "0.00", "0.0", "()"]:
        return 0.0

    is_neg = False
    if s.startswith("(") and s.endswith(")"):
        is_neg = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        is_neg = True
        s = s[1:].strip()

    s = s.replace(",", "").replace(" ", "")
    try:
        f = float(s)
        return -f if is_neg else f
    except ValueError:
        return None


def clean_line_item_label(raw_label: str) -> str:
    """Strip bracketed footnote markers from a line-item label."""
    cleaned = re.sub(r"\s*[\(\[].*?[\)\]]$", "", raw_label).strip()
    return cleaned


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Parse Nepali Bikram Sambat (BS) year/month from the filename."""
    name = Path(filename).stem
    metadata: Dict[str, Any] = {
        "source_file": filename,
        "year_bs": None,
        "month_bs": None,
        "period_label": name,
    }

    yr_match = re.search(r"20[678]\d", name)
    if yr_match:
        metadata["year_bs"] = int(yr_match.group(0))

    for m in NEPALI_MONTHS:
        if m.lower() in name.lower():
            metadata["month_bs"] = m
            break

    return metadata


def _is_rotated(char: Dict[str, Any]) -> bool:
    """True if a pdfplumber char dict is 90-degree rotated text.

    pdfplumber's own 'upright' flag is unreliable in this document (it
    reports True even for visibly rotated glyphs), so rotation is read
    directly off the text matrix: normal text has matrix ~ (1,0,0,1,..),
    90-degree rotated text has matrix ~ (0,1,-1,0,..).
    """
    a, b = char["matrix"][0], char["matrix"][1]
    return abs(a) < 0.5 and abs(b) > 0.5


def recover_bank_names(page: "pdfplumber.page.Page", col_gap: float = 5.0, top_max: float = 500.0) -> List[str]:
    """Recover the rotated BFI-name header row for one detail page.

    Finds the 'Name' word (from the 'BFI Name' header) to anchor the
    left edge of the bank-name columns, then clusters rotated characters
    to its right by x0 (one cluster per bank column). Within each
    cluster, characters are sorted by ascending 'top' and concatenated,
    then the result is reversed -- the rotated text is printed bottom to
    top, so the first character encountered (smallest top) is actually
    the LAST letter of the name.
    """
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    name_word = next((w for w in words if w["text"] == "Name"), None)
    if name_word is None:
        return []
    x0_min = name_word["x1"] - 5

    chars = [
        c for c in page.chars
        if c["x0"] > x0_min and c["top"] < top_max and c["text"].strip() and _is_rotated(c)
    ]
    if not chars:
        return []
    chars.sort(key=lambda c: c["x0"])

    columns: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_x0: Optional[float] = None
    for c in chars:
        if last_x0 is not None and c["x0"] - last_x0 > col_gap:
            columns.append(current)
            current = []
        current.append(c)
        last_x0 = c["x0"]
    if current:
        columns.append(current)

    names = []
    for col in columns:
        col_sorted = sorted(col, key=lambda c: c["top"])
        names.append("".join(c["text"] for c in col_sorted)[::-1])
    return names


class RobustNRBExtractor:
    def __init__(self, pdf_path: Path | str):
        self.pdf_path = Path(pdf_path)
        self.metadata = extract_metadata_from_filename(self.pdf_path.name)
        self.reader = pypdf.PdfReader(str(self.pdf_path))
        self.plumber_pdf = pdfplumber.open(str(self.pdf_path))
        self.total_pages = len(self.reader.pages)

    def close(self):
        self.plumber_pdf.close()

    def find_section_pages(self, keyword: str, end_page: int = 50) -> List[int]:
        matched_pages = []
        for idx in range(min(end_page, self.total_pages)):
            try:
                txt = (self.reader.pages[idx].extract_text() or "").lower()
                txt = re.sub(r"\s+", " ", txt)
                if re.sub(r"\s+", " ", keyword.lower()) in txt:
                    matched_pages.append(idx)
            except Exception:
                continue
        return matched_pages

    def _extract_rows(self, p_idx: int, banks: List[str], statement_name: str) -> List[Dict[str, Any]]:
        k = len(banks)
        layout_text = self.reader.pages[p_idx].extract_text(extraction_mode="layout") or ""
        lines = [l for l in layout_text.split("\n") if l.strip()]

        records = []
        prev_line: Optional[str] = None

        for line in lines:
            parts = line.split()
            if len(parts) < k:
                # No room for k numeric columns -- a section header
                # (e.g. "Liabilities", "Income") or page title. Not a
                # data row; skip it. The one documented exception is a
                # label that wraps mid-parenthetical (see below).
                prev_line = line
                continue

            candidate_values = parts[-k:]
            valid = [bool(NUMERIC_TOKEN_RE.match(p)) for p in candidate_values]
            if sum(valid) < max(k - 1, 1):
                prev_line = line
                continue

            raw_label = " ".join(parts[:-k])
            if prev_line is not None and prev_line.count("(") > prev_line.count(")"):
                # e.g. "LOANS & ADVANCES (Including Bills" wraps onto the
                # next physical line as "4 Purchased)" -- a real label
                # split across two lines in the source PDF, not a parsing
                # artifact. An unmatched '(' on the immediately preceding
                # (otherwise-skipped) line is a reliable, narrow signal
                # for this specific wrap, unlike guessing from "no
                # numbers on this line" alone (which also matches page
                # titles and section headers, and previously caused
                # unrelated lines to get glued together).
                #
                # The row-numbering marker (e.g. "4") prints on the
                # SECOND physical line here, not the first, so it has to
                # be moved back to the front when the two lines are
                # stitched together -- otherwise it ends up stranded in
                # the middle of the label and gets eaten by the trailing
                # bracketed-footnote strip below.
                hier_match = HIERARCHY_RE.match(raw_label)
                if hier_match:
                    marker = hier_match.group(1)
                    rest = raw_label[hier_match.end():]
                    raw_label = f"{marker} {prev_line.strip()} {rest}"
                else:
                    raw_label = prev_line.strip() + " " + raw_label
            prev_line = None

            hier_match = HIERARCHY_RE.match(raw_label.strip())
            item_code = hier_match.group(1) if hier_match else None

            canonical_label = clean_line_item_label(raw_label)
            if not canonical_label or len(canonical_label) < 2:
                continue

            for bfi_idx, bfi_name in enumerate(banks):
                val = clean_number(candidate_values[bfi_idx])
                records.append({
                    "source_file": self.pdf_path.name,
                    "year_bs": self.metadata.get("year_bs"),
                    "month_bs": self.metadata.get("month_bs"),
                    "page": p_idx + 1,
                    "statement": statement_name,
                    "bfi_name": bfi_name,
                    "item_code": item_code,
                    "line_item": canonical_label,
                    "value_mn_npr": val,
                })

        return records

    def extract_statement(self, keyword: str, statement_name: str) -> pd.DataFrame:
        section_pages = self.find_section_pages(keyword)
        records = []
        expected_k: Optional[int] = None

        for p_idx in section_pages:
            try:
                banks = recover_bank_names(self.plumber_pdf.pages[p_idx])
                if not banks:
                    logger.warning(
                        f"{statement_name} p.{p_idx + 1}: no BFI name header recovered, skipping page"
                    )
                    continue

                if expected_k is not None and len(banks) != expected_k:
                    logger.warning(
                        f"{statement_name} p.{p_idx + 1}: detected {len(banks)} banks, "
                        f"expected {expected_k} from earlier page in this section"
                    )
                expected_k = expected_k or len(banks)

                records.extend(self._extract_rows(p_idx, banks, statement_name))
            except Exception as e:
                logger.warning(f"Error parsing {statement_name} on page {p_idx + 1}: {e}")

        return pd.DataFrame(records)

    def extract_all(self) -> Dict[str, pd.DataFrame]:
        try:
            return {
                "balance_sheet": self.extract_statement("statement of assets and liabilities", "balance_sheet"),
                "profit_and_loss": self.extract_statement("profit and loss statement", "profit_and_loss"),
                "statement_of_credit": self.extract_statement("statement of credit", "statement_of_credit"),
            }
        finally:
            self.close()


def process_all_raw_pdfs(raw_dir: Path, output_dir: Path, max_files: Optional[int] = None):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]

    logger.info(f"Discovered {len(pdf_files)} PDF reports in {raw_dir}")

    all_data: Dict[str, List[pd.DataFrame]] = {"balance_sheet": [], "profit_and_loss": [], "statement_of_credit": []}

    for idx, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"[{idx}/{len(pdf_files)}] Parsing: {pdf_path.name}...")
        try:
            extractor = RobustNRBExtractor(pdf_path)
            results = extractor.extract_all()
            for key, df in results.items():
                if not df.empty:
                    all_data[key].append(df)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    for key, dfs in all_data.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            out_path = output_dir / f"nrb_{key}.csv"
            combined.to_csv(out_path, index=False)
            logger.info(f"Saved {key}: {out_path} ({len(combined)} rows)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Financial Statements from NRB PDFs")
    parser.add_argument("--all", action="store_true", help="Process all PDFs in the category's raw_pdfs directory")
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Path to the open_nrb/data directory (containing raw_pdfs/<category> and processed/<category>)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="monthly_statistical_bfis",
        choices=[
            "annual_reports",
            "financial_stability_report",
            "key_financial_indicators",
            "monthly_statistical_bfis",
            "study_reports",
        ],
        help="Subdirectory under data/raw_pdfs and data/processed to use",
    )
    parser.add_argument("--output", type=str, default=None, help="Override output directory for processed CSV files")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of PDFs processed")
    parser.add_argument("pdf", nargs="?", help="Process a single PDF instead of a directory")

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    default_raw_dir = data_root / "src" / "open_nrb" / "data" / "raw_pdfs"  / args.category
    default_out_dir = data_root / "src" / "open_nrb" / "data" / "processed" / args.category
    out_dir = Path(args.output) if args.output else default_out_dir

    if args.all:
        process_all_raw_pdfs(default_raw_dir, out_dir, max_files=args.max_files)
    elif args.pdf:
        extractor = RobustNRBExtractor(Path(args.pdf))
        results = extractor.extract_all()
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, df in results.items():
            if not df.empty:
                df.to_csv(out_dir / f"sample_{key}.csv", index=False)
        print("Sample extraction completed successfully.")
    else:
        parser.print_help()


#  /home/krishna/Documents/personal/neupane_article/network_papers/00_llama_arch/muse_llama_arch/apps/open-nrb/raw_pdfs/monthly_statistical_bfis
# /home/krishna/Documents/personal/neupane_article/network_papers/00_llama_arch/muse_llama_arch/apps/open-nrb/src/open_nrb/data/raw_pdfs/monthly_statistical_bfis
