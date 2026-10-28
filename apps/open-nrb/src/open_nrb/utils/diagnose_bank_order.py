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


_NUMERIC_TOKEN_RE = re.compile(r"^\(?\s*-?\d[\d,]*(?:\.\d+)?\)?$|^-+$")


def _looks_like_data_row(parts: list[str]) -> bool:
    if len(parts) < 2 or not _NUMERIC_TOKEN_RE.match(parts[-1]):
        return False
    return sum(1 for p in parts if _NUMERIC_TOKEN_RE.match(p)) >= 2


def phase1_banks_on_page(extractor: NRBFinancialExtractor, p_idx: int) -> list[str]:
    """Exact replica of extract_statement_generic's phase-1 bank-token-collection loop."""
    page_text = extractor.reader.pages[p_idx].extract_text() or ""
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    banks: list[str] = []
    in_banks = False
    for l in lines:
        if "BFI Name" in l or "BFI name" in l:
            in_banks = True
            continue
        if not in_banks:
            continue
        token = l.strip()
        if token in extractor.KNOWN_BFIS or token in extractor.PLACEHOLDER_BFIS:
            banks.append(token)
            continue
        if _looks_like_data_row(token.split()):
            break
    return banks


def reconstruct_header_columns(
    extractor: NRBFinancialExtractor, p_idx: int
) -> list[tuple[float, str]] | None:
    """Finds the 'BFI Name' marker word, then reconstructs every physical bank-name column
    in the header band by clustering single-character words by x0 and joining each cluster
    top-down (highest 'top' first) -- see module docstring for why this is necessary instead
    of exact word matching. Returns [(x0, name), ...] sorted left-to-right, or None if no
    'BFI Name' marker word is found on this page.
    """
    plumber_page = extractor._plumber_page(p_idx)
    words = plumber_page.extract_words(x_tolerance=1, y_tolerance=1)

    bfi_top = None
    for w in words:
        if w["text"] in ("BFI", "BFI Name", "BFI name"):
            bfi_top = w["top"]
            break
    if bfi_top is None:
        return None

    # The bank-name header is built from single-character words stacked vertically (see
    # module docstring), so its vertical span grows with name length -- a fixed-height window
    # either clips long names or bleeds into the first data row. Confirmed 2026-10-27 via
    # --dump-page on Asar_2080_Publish.pdf p17: the old band_hi=bfi_top+20 captured the first
    # data row's serial-number digit ('1', x0=19.9) and its value's split-off leading digit
    # ('3', x0=123.6) as spurious single-char columns, while band_lo=bfi_top-5 clipped
    # NABIL/NIMB/Kumari/Laxmi down to their first 2-3 letters (full "Kumari" spans top
    # 45.3-58.4, but band_lo=50.7 cut everything above 50.7). Bound band_hi by the page's own
    # "Liabilities"/"Assets" side-header word -- which always immediately follows the
    # bank-name header, before any data row -- instead of a fixed offset.
    side_header_tops = [w["top"] for w in words
                         if w["text"] in ("Liabilities", "Assets") and w["top"] > bfi_top]
    band_hi = min(side_header_tops) - 1 if side_header_tops else bfi_top + 10
    # -15 was enough for 5-6 letter names (NABIL, Kumari, Laxmi) but clipped the tail of longer
    # 8-9 letter names (Mahalaxmi, Shangrila, Narayani -> truncated to Mahalax/Shangril/Narayan)
    # confirmed via full-corpus run 2026-10-27. There's a wide safe margin above the header
    # before the page title line (top~14 on sampled pages), so widen generously.
    band_lo = bfi_top - 30
    single_chars = [w for w in words if len(w["text"]) == 1 and band_lo <= w["top"] <= band_hi]
    if not single_chars:
        return []

    single_chars.sort(key=lambda w: w["x0"])
    clusters: list[list[dict]] = []
    for w in single_chars:
        if clusters and abs(w["x0"] - clusters[-1][-1]["x0"]) <= 2.0:
            clusters[-1].append(w)
        else:
            clusters.append([w])

    columns: list[tuple[float, str]] = []
    for cluster in clusters:
        cluster_sorted = sorted(cluster, key=lambda w: -w["top"])
        x0 = sum(w["x0"] for w in cluster) / len(cluster)
        name = "".join(w["text"] for w in cluster_sorted)
        columns.append((x0, name))
    return columns


def dump_page_words(extractor: NRBFinancialExtractor, p_idx: int, banks: list[str]) -> None:
    """Verbose debug: dump every pdfplumber word on the page (text, x0, x1, top, upright),
    plus every raw char's 'upright' flag.
    """
    plumber_page = extractor._plumber_page(p_idx)
    words = plumber_page.extract_words(x_tolerance=1, y_tolerance=1)
    print(f"  -- {len(words)} words on page (phase-1 banks expected: {banks}) --")
    for w in words:
        flag = "" if w.get("upright", True) else "  <<< ROTATED"
        print(f"     text={w['text']!r:30s} x0={w['x0']:.1f} x1={w['x1']:.1f} top={w['top']:.1f}{flag}")

    non_upright_chars = [c for c in plumber_page.chars if not c.get("upright", True)]
    print(f"  -- {len(plumber_page.chars)} raw chars total, {len(non_upright_chars)} non-upright (rotated) --")
    if non_upright_chars:
        sample = non_upright_chars[:40]
        joined_preview = "".join(c["text"] for c in sample)
        print(f"     first {len(sample)} rotated chars (raw order): {joined_preview!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--only-contains", type=str, default=None,
                         help="Only check files whose extracted bank list contains this name (e.g. Mega)")
    parser.add_argument("--dump-first-failures", type=int, default=0,
                         help="For the first N pages with no reconstructable header, dump raw "
                              "pdfplumber words/chars for that page")
    parser.add_argument("--dump-page", type=str, default=None,
                         help="Dump raw pdfplumber words/chars for one specific page, format "
                              "'filename.pdf:PAGE' (PAGE is 1-indexed). Runs instead of the "
                              "full-corpus scan.")
    args = parser.parse_args()

    if args.dump_page:
        fname, _, page_str = args.dump_page.rpartition(":")
        p_idx = int(page_str) - 1
        pdf_path = args.raw_dir / fname
        extractor = NRBFinancialExtractor(pdf_path)
        try:
            banks = phase1_banks_on_page(extractor, p_idx)
        except Exception as e:
            banks = [f"<phase1 error: {e}>"]
        dump_page_words(extractor, p_idx, banks)
        return


    pdf_files = sorted(args.raw_dir.glob("*.pdf"))
    print(f"=== checking phase-1 text order vs reconstructed physical header order over {len(pdf_files)} PDFs ===\n")

    checked = 0
    no_header_found = 0
    order_mismatches = []
    count_mismatches = []
    total_section_pages = 0
    empty_banks_pages = 0
    phase1_exception_count = 0
    phase1_exceptions: list[str] = []
    recon_exception_count = 0
    recon_exceptions: list[str] = []
    files_with_no_section_pages = 0
    unknown_column_counts: dict[str, int] = {}
    dumped_so_far = 0

    for pdf_path in pdf_files:
        keyword = "statement of assets and liabilities"
        try:
            extractor = NRBFinancialExtractor(pdf_path)
            section_pages = extractor.find_section_pages(keyword)
        except Exception as e:
            print(f"  [skip] {pdf_path.name}: {e}")
            continue

        if not section_pages:
            files_with_no_section_pages += 1
            continue

        # Mirrors the 2026-10-27 fix in pdf_extract.py: bound to the Commercial Banks
        # subsection only, using the bulletin's own TOC page numbers, so this diagnostic
        # measures the SAME page set the production extractor now processes.
        bounds = extractor._find_commercial_bank_section_bounds(keyword, section_pages)
        if bounds is not None:
            start, end = bounds
            section_pages = [p for p in section_pages if start <= p < end]

        for p_idx in section_pages:
            total_section_pages += 1
            try:
                banks = phase1_banks_on_page(extractor, p_idx)
            except Exception as e:
                phase1_exception_count += 1
                if len(phase1_exceptions) < 5:
                    phase1_exceptions.append(f"{pdf_path.name} p{p_idx+1}: {type(e).__name__}: {e}")
                continue
            if not banks:
                empty_banks_pages += 1
                continue
            if args.only_contains and args.only_contains not in banks:
                continue

            try:
                columns = reconstruct_header_columns(extractor, p_idx)
            except Exception as e:
                recon_exception_count += 1
                if len(recon_exceptions) < 5:
                    recon_exceptions.append(f"{pdf_path.name} p{p_idx+1}: {type(e).__name__}: {e}")
                continue
            if not columns:
                no_header_found += 1
                if dumped_so_far < args.dump_first_failures:
                    print(f"\n=== DUMP [{pdf_path.name} p{p_idx+1}] phase-1 banks: {banks} ===")
                    dump_page_words(extractor, p_idx, banks)
                    dumped_so_far += 1
                continue

            checked += 1
            known_set = extractor.KNOWN_BFIS
            # Order comparison must count placeholder columns (Grand/BOA) too, since phase-1
            # `banks` now includes them (matches extract_statement_generic's real alignment
            # behavior) -- otherwise every placeholder-containing page would spuriously show
            # as an order mismatch just from the length difference.
            alignment_set = extractor.KNOWN_BFIS | extractor.PLACEHOLDER_BFIS
            physical_known_order = [name for _, name in columns if name in alignment_set]
            unknown_names = [name for _, name in columns if name not in known_set]
            for name in unknown_names:
                unknown_column_counts[name] = unknown_column_counts.get(name, 0) + 1

            if len(columns) != len(banks):
                count_mismatches.append((pdf_path.name, p_idx + 1, banks, columns))
            elif physical_known_order != banks:
                order_mismatches.append((pdf_path.name, p_idx + 1, banks, physical_known_order))

    print(f"files with zero section pages matched at all: {files_with_no_section_pages}/{len(pdf_files)}")
    print(f"total section pages visited: {total_section_pages}")
    print(f"pages where phase-1 bank list came back empty: {empty_banks_pages}")
    print(f"pages where phase-1 scan raised an exception: {phase1_exception_count} (first 5 shown below)")
    for msg in phase1_exceptions:
        print(f"    {msg}")
    print(f"pages where header reconstruction raised an exception: {recon_exception_count} (first 5 shown below)")
    for msg in recon_exceptions:
        print(f"    {msg}")
    print(f"pages with no reconstructable header found: {no_header_found}")
    print(f"pages checked (header reconstructed): {checked}")
    if unknown_column_counts:
        print("  physical columns found that are NOT in KNOWN_BFIS (name -> # pages seen on):")
        for name, cnt in sorted(unknown_column_counts.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    {name!r}: {cnt}")
    print(f"COUNT MISMATCHES (physical column count != phase-1 bank count): {len(count_mismatches)}")
    print(f"ORDER MISMATCHES (counts equal but order differs): {len(order_mismatches)}\n")

    print("-- count mismatches (these are the dangerous ones -- _bank_x_anchors falls back or drops the page) --")
    for fname, page, banks, columns in count_mismatches[:30]:
        print(f"  {fname} p{page}")
        print(f"    phase-1 banks ({len(banks)}):      {banks}")
        print(f"    physical columns ({len(columns)}): {[name for _, name in columns]}")
        print()
    if len(count_mismatches) > 30:
        print(f"  ... and {len(count_mismatches) - 30} more")

    print("\n-- order mismatches (same count, different sequence -- would mean a real mislabeling bug) --")
    for fname, page, banks, physical_order in order_mismatches[:30]:
        print(f"  {fname} p{page}")
        print(f"    phase-1 (text) order:     {banks}")
        print(f"    physical (x0)     order:  {physical_order}")
        print()
    if len(order_mismatches) > 30:
        print(f"  ... and {len(order_mismatches) - 30} more")


if __name__ == "__main__":
    main()


