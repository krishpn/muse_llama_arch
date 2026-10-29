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
import pdfplumber
import pypdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nrb_extractor")

# Raw PDFs excluded from process_all_raw_pdfs entirely. Confirmed 2026-10-27
# (diagnose_bank_order.py full-corpus run) that Kartik2078_NFRS_PublishV1-1.pdf's
# header-reconstruction x0-clustering breaks down (garbled single-occurrence column
# names on p18, a dropped "Global" column on p17) -- this file is an annual NFRS-format
# report, structurally different from the regular Monthly_Statistics-*.pdf bulletin
# series, and is the only file left with a diagnostic mismatch after the TOC-bounding
# and Grand/BOA placeholder fixes. Excluded rather than special-cased since it's a single
# outlier, not part of the monthly time series the model actually needs.
EXCLUDED_RAW_PDFS = {"Kartik2078_NFRS_PublishV1-1.pdf"}


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

# Alternate spellings/abbreviations confirmed present across real filenames (2026-10-25),
# mapped to one canonical BS_MONTHS entry so every consumer sees one name per month. The
# previous list (Baishakh/Jestha/Asar/Ashar/Shrawan/Sawan/Bhadra/Asoj/Ashwin/Aswin/Kartik/
# Mangshir/Poush/Magh/Falgun/Chaitra) never matched real abbreviated filenames like
# "Bhadau_2075Mid-...", "Saun-2074.pdf", "Paush_2075Mid-...", "Kattik-2074.pdf",
# "Chait-2074.pdf" -- those left month_bs blank for a large fraction of real files.
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
# Canonical spellings checked before aliases, so e.g. a filename containing the full word
# "Chaitra" matches that directly rather than first hitting the "Chait" alias as a substring.
_ALL_MONTH_SPELLINGS = BS_MONTHS + list(_MONTH_ALIASES)

# Two numeric filename conventions confirmed present (2026-10-25), month 1-indexed from
# Baisakh: "Monthly_Statistics-{year_bs}-{month:02d}-..." and a punctuation-swapped
# "Monthly-Statistics-_-{year_bs}_{month:02d}..." variant from a different bulletin-
# production era (~2076-05 .. 2078-04). Tried first since they're unambiguous; the
# name-based fallback below is only needed for filenames that use neither.
_NUMERIC_PERIOD_RE = re.compile(r"Monthly_Statistics-(\d{4})-(\d{2})-")
_NUMERIC_PERIOD_RE_ALT = re.compile(r"Monthly-Statistics-_-(\d{4})_(\d{2})")

# Leading item-code token confirmed on real label lines: bare top-level digits with an
# optional trailing dot ("1", "10", "1."), or a single-letter sub-item with a trailing dot
# ("a.", "b."), per data_loader.py's FIELD_LABELS/BORROWING_SUBITEMS examples -- reconstructed
# 2026-10-25 after the real pdf_extract.py's original item_code-splitting logic was
# overwritten with no backup; re-verify infer_sides' side-boundary output against known
# bulletins once this is run, since this regex is new, not carried over from the original.
_ITEM_CODE_RE = re.compile(r"^\d+\.?$|^[a-zA-Z]\.$")

# Shared numeric-token test for both the label+values-on-one-line layout (the first page of a
# multi-page balance-sheet table) and the label-block/value-block layout (continuation pages,
# confirmed 2026-10-25 via dump_bank_header_region.py against Asar_2080_Publish.pdf -- as the
# number of BFI columns outgrew one page width, NRB's bulletins started rendering the full label
# column as its own text block, then the bank-name header row, then every value row as pure
# numbers with no label text at all on the same line).
_NUMERIC_TOKEN_RE = re.compile(r"^\(?\s*-?\d[\d,]*(?:\.\d+)?\)?$|^-+$")

# The two balance-sheet-side section dividers that appear as a bare word on their own line/row,
# with no corresponding value -- must never be treated as a line item.
_SIDE_HEADER_TOKENS = {"Liabilities", "Assets"}

# Confirmed 2026-10-27: every bulletin's Table of Contents lists printed page numbers for
# each institution-class subsection immediately after a statement's heading, e.g.
# "a. Commercial banks 15   b. Development banks 17   c. Finance companies 22". Used to
# bound extraction to the Commercial Banks subsection only -- see
# _find_commercial_bank_section_bounds.
_COMMERCIAL_BANKS_TOC_RE = re.compile(r"Commercial\s+banks\s+(\d+)", re.IGNORECASE)
_DEVELOPMENT_BANKS_TOC_RE = re.compile(r"Development\s+banks\s+(\d+)", re.IGNORECASE)


def _looks_like_data_row(parts: List[str]) -> bool:
    """True once a line's tokens look like real tabular data (>=2 numeric tokens, numeric at the
    end) rather than plain label text."""
    if len(parts) < 2 or not _NUMERIC_TOKEN_RE.match(parts[-1]):
        return False
    return sum(1 for p in parts if _NUMERIC_TOKEN_RE.match(p)) >= 2


def _cluster_rows(words: List[dict], tol: float = 2.5) -> List[Dict[str, Any]]:
    """Group pdfplumber words into table rows by page y-position ('top'), independent of
    whatever order the PDF content stream emitted them in. Confirmed 2026-10-25
    (dump_pdfplumber_rows.py) that NRB's apparent 'continuation-page block layout' -- where
    pypdf.extract_text() prints the whole label column, then the bank-name header, then every
    value row, with no label/value sharing a physical line -- is purely a text-STREAM-order
    artifact: every label and its row of bank values share the same page y-position, so
    clustering by 'top' recovers one unified row structure on every balance-sheet page, whether
    it's the first page of the table or a continuation page.
    """
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: List[Dict[str, Any]] = []
    for w in words_sorted:
        if rows and abs(w["top"] - rows[-1]["top_ref"]) <= tol:
            grp = rows[-1]
            grp["words"].append(w)
            grp["top_ref"] = (
                grp["top_ref"] * (len(grp["words"]) - 1) + w["top"]
            ) / len(grp["words"])
        else:
            rows.append({"top_ref": w["top"], "words": [w]})
    for r in rows:
        r["words"].sort(key=lambda w: w["x0"])
    return rows


def _bank_x_anchors(rows: List[Dict[str, Any]], banks: List[str]) -> Optional[Dict[str, float]]:
    """Map each bank (in the left-to-right column order already established by the existing
    bank-name detection) to its x-center on the page, using whichever clustered row has exactly
    one numeric value per bank. Prefers a 'TOTAL ASSETS'/'TOTAL LIABILITIES' row -- confirmed
    2026-10-25 to always be fully populated for every bank column, unlike ordinary line items
    which can be blank for some banks -- so column x-positions are never inferred from a
    possibly-sparse row.
    """
    k = len(banks)
    fallback: Optional[List[dict]] = None
    for r in rows:
        numeric_words = [w for w in r["words"] if _NUMERIC_TOKEN_RE.match(w["text"])]
        if len(numeric_words) != k:
            continue
        row_text = "".join(w["text"] for w in r["words"]).upper()
        if "TOTAL" in row_text and ("ASSET" in row_text or "LIABILIT" in row_text):
            centers = sorted((w["x0"] + w["x1"]) / 2 for w in numeric_words)
            return dict(zip(banks, centers))
        if fallback is None:
            fallback = numeric_words
    if fallback is None:
        return None
    centers = sorted((w["x0"] + w["x1"]) / 2 for w in fallback)
    return dict(zip(banks, centers))


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

    # BS year is currently in the low 2080s (2026 AD); "20[6-9]\d" covers 2060-2099 BS,
    # giving headroom to ~2043 AD without colliding with any Gregorian year in these
    # filenames (those all have a third digit of 0/1/2 through 2029 AD).
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

    # Canonical BFI short-name set -- merged from data_loader.py's COMMERCIAL_BANK_PREFIXES
    # (the modeling package's own, independently-verified list) plus the non-commercial
    # entries this extractor also needs to recognize. Confirmed missing "Himalaya",
    # "Siddhartha", "Sunrise", and "NIBL" (2026-10-25) -- those names never hit the old
    # isupper()-only fallback below in every bulletin-era PDF layout, which silently dropped
    # them from `banks` and shifted the column-to-bank mapping for every bank listed after
    # them on the same page (see extract_statement_generic). Deliberately does NOT include
    # "Everest" -- Everest Bank Ltd's real column header is "EBL" in every bulletin era
    # checked (confirmed 2026-10-25, including its earliest PDF from 2012); a prior pipeline
    # run's CSV had spurious "Everest"-labeled rows with garbled truncated line items and
    # values inconsistent with EBL's real figures (not a unit/scale difference, ratios vary
    # 41x-108x across line items on the same file) -- extraction noise from some other process,
    # not a legitimate alternate name, so it must not be in this whitelist.
    KNOWN_BFIS = {
        "ADBNL", "BOK", "CBL", "Central", "Century", "Citizen", "Citizens", "Civil", "CTBNL",
        "EBL", "Excel", "Garima", "Global", "Goodwill", "Green", "HBL", "Himalaya",
        "ICFC", "Janata", "Jyoti", "Kamana", "Karnali", "Kist", "Kumari", "Laxmi", "Lumbini",
        "Mahalaxmi", "Manjushree", "MBL", "Mega", "Miteri", "Muktinath", "NABIL", "Narayani",
        "NBB", "NBBL", "NBL", "NCC", "NIBL", "NIC", "NIFRA", "NIMB", "NMB", "NSBI", "Pokhara",
        "PRABHU", "Prabhu", "Prime", "RBB", "Sanima", "Saptakoshi", "SBL", "SCBNL",
        "Shangrila", "Shine", "Siddhartha", "Sunrise",
    }

    # Real physical columns confirmed present (diagnose_bank_order.py, 2026-10-27 full-corpus
    # run) but deliberately NOT added to KNOWN_BFIS -- their identity (e.g. whether "Grand" is
    # "Grand Bank Nepal Ltd.") hasn't been verified against NRB's BFI roster. Matched here only
    # so phase-1's bank count lines up with the true physical column count on pages where they
    # appear, which prevents _bank_x_anchors's count-mismatch fallback from mis-zipping KNOWN_BFIS
    # columns onto the wrong x-position; their own values are discarded at record-emission time
    # (see extract_statement_generic) rather than attributed to a confirmed bank name.
    PLACEHOLDER_BFIS = {"Grand", "BOA"}

    def __init__(self, pdf_path: Path | str):
        self.pdf_path = Path(pdf_path)
        self.metadata = extract_metadata_from_filename(self.pdf_path.name)
        self.reader = pypdf.PdfReader(str(self.pdf_path))
        self.total_pages = len(self.reader.pages)
        # Opened lazily, not in __init__, so instantiating an extractor just to read
        # find_section_pages/extract_bfi_name_list (which only need pypdf) doesn't pay the
        # cost of a second parse of the same PDF.
        self._plumber_pdf: Optional["pdfplumber.PDF"] = None

    def _plumber_page(self, p_idx: int) -> "pdfplumber.page.Page":
        if self._plumber_pdf is None:
            self._plumber_pdf = pdfplumber.open(str(self.pdf_path))
        return self._plumber_pdf.pages[p_idx]

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

    def _find_commercial_bank_section_bounds(
        self, keyword: str, section_pages: List[int]
    ) -> Optional[Tuple[int, int]]:
        """Returns (start_0idx, end_0idx_exclusive) physical page bounds covering only the
        Commercial Banks subsection of `keyword`'s table, derived from the bulletin's own
        Table of Contents -- which lists printed page numbers for "a. Commercial banks N",
        "b. Development banks N" immediately after each statement's heading. Returns None
        if this TOC pattern isn't found (irregular/older bulletin), so the caller can fall
        back to processing section_pages unfiltered rather than silently dropping everything.

        Confirmed 2026-10-27: pages past the Commercial Banks subsection (Development Banks,
        Finance Companies, Microfinance) use entirely different BFI codes that KNOWN_BFIS was
        never built to cover -- ~90% of those pages fail _bank_x_anchors's exact column-count
        match and either drop silently or mis-zip values onto the wrong bank.
        """
        if len(section_pages) < 2:
            return None
        toc_idx = section_pages[0]
        flat = re.sub(r"\s+", " ", self.reader.pages[toc_idx].extract_text() or "")
        # "and" (not "&") distinguishes this per-institution-class breakdown (TOC item 6)
        # from the aggregate statement entry (TOC item 3, "Assets & Liabilities") sharing
        # nearly the same title.
        heading_pos = flat.lower().find(keyword.lower())
        if heading_pos == -1:
            return None
        window = flat[heading_pos: heading_pos + 300]
        m_commercial = _COMMERCIAL_BANKS_TOC_RE.search(window)
        m_development = _DEVELOPMENT_BANKS_TOC_RE.search(window)
        if not m_commercial or not m_development:
            return None

        printed_commercial = int(m_commercial.group(1))
        printed_development = int(m_development.group(1))
        # section_pages[0] is the TOC page itself; section_pages[1] is always the first real
        # table page -- i.e. the physical page where the Commercial Banks subsection starts.
        first_table_physical_1idx = section_pages[1] + 1
        offset = first_table_physical_1idx - printed_commercial
        development_start_physical_1idx = printed_development + offset
        return section_pages[1], development_start_physical_1idx - 1

    def extract_statement_generic(self, keyword: str, statement_name: str) -> pd.DataFrame:
        """Generic extractor for tabular statement types (Balance Sheet, P&L, Credit)."""
        section_pages = self.find_section_pages(keyword)
        bounds = self._find_commercial_bank_section_bounds(keyword, section_pages)
        if bounds is not None:
            start, end = bounds
            section_pages = [p for p in section_pages if start <= p < end]
        else:
            logger.warning(
                f"{self.pdf_path.name}: could not find TOC Commercial/Development banks "
                f"page numbers for '{keyword}' -- processing all {len(section_pages)} "
                f"matched pages unfiltered (may include non-commercial-bank institutions)"
            )
        records = []

        for p_idx in section_pages:
            try:
                page_text = self.reader.pages[p_idx].extract_text() or ""
                lines = [l.strip() for l in page_text.split("\n") if l.strip()]

                # Phase 1: collect bank-column tokens, in left-to-right column order, after the
                # "BFI Name" marker. pypdf's extract_text() renders the rotated/vertical
                # bank-name header back into clean reading-order words, so this text-based scan
                # is kept unchanged -- only the label/value reconstruction below (phase 2) moves
                # to coordinate-based extraction.
                banks: List[str] = []
                in_banks = False
                for l in lines:
                    if "BFI Name" in l or "BFI name" in l:
                        in_banks = True
                        continue
                    if not in_banks:
                        continue
                    token = l.strip()
                    # Exact membership in KNOWN_BFIS only -- confirmed 2026-10-25 via a
                    # full-corpus run (184 files) that an isalpha()-length fallback here
                    # catches ZERO genuine bank-name variants beyond KNOWN_BFIS but DOES catch
                    # "Liabilities" as a false positive in 158/184 files.
                    if token in self.KNOWN_BFIS or token in self.PLACEHOLDER_BFIS:
                        banks.append(token)
                        continue
                    if _looks_like_data_row(token.split()):
                        break

                if not banks:
                    continue
                k = len(banks)

                # Phase 2: reconstruct the table from pdfplumber word coordinates, not
                # text-stream order. Confirmed 2026-10-25 (dump_pdfplumber_rows.py): a label and
                # its row of bank values always share the same page y-position ("top"), even on
                # pages where pypdf's text-stream order prints the whole label column before any
                # bank value -- which previously made banks like Global/Citizen/Prime/Sunrise/
                # NMB/Prabhu/ADBNL/Mega nearly invisible. Coordinates also fix a second, worse
                # problem text-only parsing can't: once any bank's cell on a row is blank, the
                # row has fewer tokens than k and there is no way to tell from text alone which
                # bank a surviving number belongs to (e.g. a lone value could be any of the k
                # columns). x-position removes the ambiguity.
                plumber_page = self._plumber_page(p_idx)
                words = plumber_page.extract_words(x_tolerance=1, y_tolerance=1)
                rows = _cluster_rows(words)

                header_row = next(
                    (r for r in rows if any("BFI" in w["text"] for w in r["words"])
                     and any("Name" in w["text"] for w in r["words"])),
                    None,
                )
                bank_x = _bank_x_anchors(rows, banks)
                if header_row is None or bank_x is None:
                    logger.warning(
                        f"Could not locate header row / bank column x-positions on page "
                        f"{p_idx+1} of {self.pdf_path.name} -- skipping page"
                    )
                    continue

                anchors = sorted(bank_x.items(), key=lambda kv: kv[1])
                anchor_xs = [x for _, x in anchors]
                gaps = [b - a for a, b in zip(anchor_xs[:-1], anchor_xs[1:])]
                threshold = (min(gaps) / 2) if gaps else 15.0
                left_margin = anchor_xs[0] - threshold

                def _row_text_upper(r: Dict[str, Any]) -> str:
                    return "".join(w["text"] for w in r["words"]).upper()

                total_row_top = max(
                    (r["top_ref"] for r in rows
                     if sum(1 for w in r["words"] if _NUMERIC_TOKEN_RE.match(w["text"])) == k
                     and "TOTAL" in _row_text_upper(r)
                     and ("ASSET" in _row_text_upper(r) or "LIABILIT" in _row_text_upper(r))),
                    default=None,
                )

                for row in rows:
                    if row["top_ref"] <= header_row["top_ref"]:
                        continue
                    if total_row_top is not None and row["top_ref"] > total_row_top + 1e-6:
                        continue

                    # Position alone (left of the leftmost bank column, with the threshold
                    # buffer already subtracted into left_margin) is enough to identify label
                    # tokens -- do NOT also exclude numeric-looking words here, or bare
                    # serial/item codes ("1", "4", ...), which match _NUMERIC_TOKEN_RE just like
                    # a value does, get silently dropped from the label and item_code ends up
                    # empty for every integer-coded top-level line item.
                    label_words = [w for w in row["words"] if w["x1"] < left_margin]
                    raw_label = " ".join(w["text"] for w in label_words).strip()
                    if not raw_label or raw_label in _SIDE_HEADER_TOKENS:
                        continue

                    value_words = [w for w in row["words"] if _NUMERIC_TOKEN_RE.match(w["text"])]
                    if not value_words:
                        continue

                    row_values: Dict[str, str] = {}
                    for w in value_words:
                        center = (w["x0"] + w["x1"]) / 2
                        nearest_bank, nearest_x = min(anchors, key=lambda kv: abs(kv[1] - center))
                        if abs(nearest_x - center) <= threshold:
                            row_values[nearest_bank] = w["text"]

                    if not row_values:
                        continue

                    label_tokens = raw_label.split()
                    item_code = label_tokens[0] if _ITEM_CODE_RE.match(label_tokens[0]) else ""

                    for bfi_name, raw_val in row_values.items():
                        if bfi_name in self.PLACEHOLDER_BFIS:
                            continue
                        records.append({
                            "source_file": self.pdf_path.name,
                            "year_bs": self.metadata.get("year_bs"),
                            "month_bs": self.metadata.get("month_bs"),
                            "page": p_idx + 1,
                            "statement": statement_name,
                            "bfi_name": bfi_name,
                            "item_code": item_code,
                            "line_item": raw_label,
                            "value_mn_npr": clean_number(raw_val),
                        })
            except Exception as e:
                logger.warning(f"Error parsing {statement_name} on page {p_idx+1}: {e}")

        return pd.DataFrame(records)

    def extract_all(self) -> Dict[str, pd.DataFrame]:
        """Extract all target datasets into a dictionary of DataFrames."""
        return {
            "balance_sheet": self.extract_statement_generic(
                "statement of assets and liabilities", "balance_sheet"
            ),
            "profit_and_loss": self.extract_statement_generic(
                "profit and loss statement", "profit_and_loss"
            ),
            "statement_of_credit": self.extract_statement_generic(
                "statement of credit", "statement_of_credit"
            ),
            "bfi_list": self.extract_bfi_name_list(),
        }


def process_all_raw_pdfs(raw_dir: Path, output_dir: Path, max_files: Optional[int] = None):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(raw_dir.glob("*.pdf")))
    pdf_files = [p for p in pdf_files if p.name not in EXCLUDED_RAW_PDFS]
    if max_files:
        pdf_files = pdf_files[:max_files]

    logger.info(f"Discovered {len(pdf_files)} PDF reports in {raw_dir}")

    all_data = {
        "balance_sheet": [], "profit_and_loss": [], "statement_of_credit": [], "bfi_list": []
    }

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
    parser = argparse.ArgumentParser(
        description="Extract Financial Statements & BFI Lists from NRB PDFs"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all PDFs in raw_pdfs directory"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory for processed CSV files"
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Limit number of PDFs processed"
    )

    args = parser.parse_args()
    default_raw_dir = (
        Path(__file__).resolve().parent.parent / "data" / "raw_pdfs" / "monthly_statistical_bfis"
    )
    default_out_dir = (
        Path(__file__).resolve().parent.parent / "data" / "processed" / "monthly_statistical_bfis"
    )
    out_dir = Path(args.output) if args.output else default_out_dir

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
