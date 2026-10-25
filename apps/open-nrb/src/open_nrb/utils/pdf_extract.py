#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pypdf>=5.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///

"""
NRB Financial Statement Extractor (open-nrb)
Extracts Balance Sheets, P&L, Statement of Credit, and BFI Name Lists
from Nepal Rastra Bank monthly statistical bulletin PDFs.
"""

import argparse
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pypdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nrb_extractor")

def clean_number(val: Any) -> Optional[float]:
    """Parse raw string from NRB table into a float.
    Handles negative values enclosed in parentheses e.g. '( 2,816.02)' and dashes '-' as 0.0.
    """
    if val is None:
        return None
    s = str(val).strip()
    if s in ["-", "–", "—", "0", "0.00", "0.0", ""]:
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



BS_MONTHS = [
    "Baisakh", "Jestha", "Asar", "Shrawan", "Bhadra", "Ashwin",
    "Kartik", "Mangsir", "Poush", "Magh", "Falgun", "Chaitra",
]

_MONTH_ALIASES: Dict[str, str] = {
    "Baishakh": "Baisakh",
    "Ashar": "Asar",
    "Sawan": "Shrawan", "Saun": "Shrawan",
    "Bhadau": "Bhadra",
    "Asoj": "Ashwin", "Aswin": "Ashwin",
    "Kattik": "Kartik",
    "Mangshir": "Mangsir",
    "Paush": "Poush",
    "Chait": "Chaitra",
}

_ALL_MONTH_SPELLINGS = BS_MONTHS + list(_MONTH_ALIASES)

# Two numeric filename conventions confirmed present (2026-10-25), month 1-indexed from
# Baisakh: "Monthly_Statistics-{year_bs}-{month:02d}-..." and a punctuation-swapped
# "Monthly-Statistics-_-{year_bs}_{month:02d}..." variant from a different bulletin-

_NUMERIC_PERIOD_RE = re.compile(r"Monthly_Statistics-(\d{4})-(\d{2})-")
_NUMERIC_PERIOD_RE_ALT = re.compile(r"Monthly-Statistics-_-(\d{4})_(\d{2})")


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Parse Nepali Bikram Sambat (BS) month/year from filename."""
    name = Path(filename).stem
    metadata: Dict[str, Any] = {
        "source_file": filename,
        "year_bs": None,
        "month_bs": None,
        "period_label": name,
    }

    m = _NUMERIC_PERIOD_RE.search(name) or _NUMERIC_PERIOD_RE_ALT.search(name)
    if m is not None:
        year_bs, month_num = int(m.group(1)), int(m.group(2))
        if 1 <= month_num <= 12:
            metadata["year_bs"] = year_bs
            metadata["month_bs"] = BS_MONTHS[month_num - 1]
            return metadata

    # BS year is currently in the low 2080s (2026);
    yr_match = re.search(r"20[6-9]\d", name)
    if yr_match:
        metadata["year_bs"] = int(yr_match.group(0))

    for spelling in _ALL_MONTH_SPELLINGS:
        if spelling.lower() in name.lower():
            metadata["month_bs"] = _MONTH_ALIASES.get(spelling, spelling)
            break

    return metadata


class NRBFinancialExtractor:
    """Extracts BFI-level financial panel datasets from NRB PDFs."""

    KNOWN_BFIS = {
        "ADBNL", "BOK", "CBL", "Central", "Century", "Citizen", "Citizens", "Civil", "CTBNL",
        "EBL", "Excel", "Garima", "Global", "Goodwill", "Green", "HBL", "Himalaya",
        "ICFC", "Janata", "Jyoti", "Kamana", "Karnali", "Kist", "Kumari", "Laxmi", "Lumbini",
        "Mahalaxmi", "Manjushree", "MBL", "Mega", "Miteri", "Muktinath", "NABIL", "Narayani",
        "NBB", "NBBL", "NBL", "NCC", "NIBL", "NIC", "NIFRA", "NIMB", "NMB", "NSBI", "Pokhara",
        "PRABHU", "Prabhu", "Prime", "RBB", "Sanima", "Saptakoshi", "SBL", "SCBNL",
        "Shangrila", "Shine", "Siddhartha", "Sunrise",
    }

    def __init__(self, pdf_path: Path | str):
        self.pdf_path = Path(pdf_path)
        self.metadata = extract_metadata_from_filename(self.pdf_path.name)
        self.reader = pypdf.PdfReader(str(self.pdf_path))
        self.total_pages = len(self.reader.pages)

    def find_section_pages(self, keyword: str, end_page: int = 50) -> List[int]:
        """Identify 0-indexed page numbers matching target statement title."""
        matched_pages = []
        max_search = min(end_page, self.total_pages)
        for idx in range(max_search):
            try:
                txt = (self.reader.pages[idx].extract_text() or "").lower()
                if keyword.lower() in txt:
                    matched_pages.append(idx)
            except Exception:
                continue
        return matched_pages

    def extract_bfi_name_list(self) -> pd.DataFrame:
        """Extract Annex-I Name list of Banks and Financial Institutions."""
        annex_pages = self.find_section_pages("name list of banks and financial institutions")
        if not annex_pages:
            annex_pages = self.find_section_pages("annex-i")

        records = []
        for p_idx in annex_pages:
            try:
                page_text = self.reader.pages[p_idx].extract_text() or ""
                lines = [l.strip() for l in page_text.split("\n") if l.strip()]
                for line in lines:
                    # Simple filter to catch listed institution rows
                    if any(char.isdigit() for char in line[:3]):  # Starts with serial number
                        records.append({
                            "source_file": self.pdf_path.name,
                            "year_bs": self.metadata.get("year_bs"),
                            "month_bs": self.metadata.get("month_bs"),
                            "page": p_idx + 1,
                            "raw_line": line
                        })
            except Exception as e:
                logger.warning(f"Error parsing BFI Name List on page {p_idx+1}: {e}")

        return pd.DataFrame(records)

    def extract_statement_generic(self, keyword: str, statement_name: str) -> pd.DataFrame:
        """Generic extractor for tabular statement types (Balance Sheet, P&L, Credit)."""
        section_pages = self.find_section_pages(keyword)
        records = []

        for p_idx in section_pages:
            try:
                page_text = self.reader.pages[p_idx].extract_text() or ""
                lines = [l.strip() for l in page_text.split("\n") if l.strip()]

                banks = []
                in_banks = False
                for l in lines:
                    if "BFI Name" in l or "BFI name" in l:
                        in_banks = True
                        continue
                    if in_banks:
                        if any(marker in l for marker in ["Expenses", "Income", "CAPITAL FUND", "1 CAPITAL FUND", "Total"]):
                            break
                        token = l.strip()
                        # Exact membership in KNOWN_BFIS only
                        if token in self.KNOWN_BFIS:
                            banks.append(token)

                if not banks:
                    continue

                k = len(banks)
                for line in lines:
                    parts = line.split()
                    if len(parts) >= k:
                        candidate_values = parts[-k:]
                        valid_nums = [
                            bool(re.match(r"^\(?\s*-?\d[\d,]*(?:\.\d+)?\)?$|^-+$", p))
                            for p in candidate_values
                        ]
                        if sum(valid_nums) >= k - 1:
                            raw_label = " ".join(parts[:-k])
                            if not raw_label:
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
                                    "line_item": raw_label,
                                    "value_mn_npr": val,
                                })
            except Exception as e:
                logger.warning(f"Error parsing {statement_name} on page {p_idx+1}: {e}")

        return pd.DataFrame(records)

    def extract_all(self) -> Dict[str, pd.DataFrame]:
        """Extract all target datasets into a dictionary of DataFrames."""
        return {
            "balance_sheet": self.extract_statement_generic("statement of assets and liabilities", "balance_sheet"),
            "profit_and_loss": self.extract_statement_generic("profit and loss statement", "profit_and_loss"),
            "statement_of_credit": self.extract_statement_generic("statement of credit", "statement_of_credit"),
            "bfi_list": self.extract_bfi_name_list()
        }


def process_all_raw_pdfs(raw_dir: Path, output_dir: Path, max_files: Optional[int] = None):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(raw_dir.glob("*.pdf")))
    if max_files:
        pdf_files = pdf_files[:max_files]

    logger.info(f"Discovered {len(pdf_files)} PDF reports in {raw_dir}")

    all_data = {"balance_sheet": [], "profit_and_loss": [], "statement_of_credit": [], "bfi_list": []}

    for idx, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"[{idx}/{len(pdf_files)}] Parsing: {pdf_path.name}...")
        try:
            extractor = NRBFinancialExtractor(pdf_path)
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
    parser = argparse.ArgumentParser(description="Extract Financial Statements & BFI Lists from NRB PDFs")
    parser.add_argument("--all", action="store_true", help="Process all PDFs in raw_pdfs directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for processed CSV files")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of PDFs processed")

    args = parser.parse_args()
    default_raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs" / "monthly_statistical_bfis"
    default_out_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / "monthly_statistical_bfis"
    out_dir = Path(args.output) if args.output else default_out_dir
    print(f"raw directory {default_raw_dir} ")
    print(f"output directory {out_dir}")

    if args.all:
        process_all_raw_pdfs(default_raw_dir, out_dir, max_files=args.max_files)
    else:
        sample_pdf = next(default_raw_dir.glob("*.pdf"), None)
        if sample_pdf:
            extractor = NRBFinancialExtractor(sample_pdf)
            results = extractor.extract_all()
            out_dir.mkdir(parents=True, exist_ok=True)
            for key, df in results.items():
                if not df.empty:
                    df.to_csv(out_dir / f"sample_{key}.csv", index=False)
            print("Sample extraction completed successfully.")
        else:
            print(f"No PDFs found in {default_raw_dir}")