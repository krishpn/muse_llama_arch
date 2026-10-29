#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Load the long-format NRB balance-sheet CSV into model tensors (design doc Sec 8).

Produces, per bfi x period: X in R^5 (size, capital ratio, liquidity ratio,
loan-to-deposit, NRB-dependence), the two interbank marginals r/c, and the raw
capital fund K (for the single-obligor cap kappa*K). See Sec 8e for the
equation-symbol <-> raw-field map this mirrors field-for-field.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import torch

BS_MONTHS = [
    "Baisakh", "Jestha", "Asar", "Shrawan", "Bhadra", "Ashwin",
    "Kartik", "Mangsir", "Poush", "Magh", "Falgun", "Chaitra",
]
_MONTH_INDEX = {name: i for i, name in enumerate(BS_MONTHS)}

# Alternate English transliterations of month_bs actually confirmed present in the real CSV
# (counted via its month_bs column, 2026-10-23) -- different bulletin extractions spell the
# same month differently. "Mangshir" is the ONLY spelling seen in real data for that month;
# BS_MONTHS' own "Mangsir" never appears raw. Mapped to the canonical BS_MONTHS entry rather
# than adding more canonical spellings, so every month_bs consumer still sees one name/month.
_MONTH_ALIASES: dict[str, str] = {
    "Ashar": "Asar",
    "Baishakh": "Baisakh",
    "Mangshir": "Mangsir",
    "Asoj": "Ashwin",
    "Aswin": "Ashwin",
}


def _normalize_month(month_bs: str) -> str:
    return _MONTH_ALIASES.get(month_bs, month_bs)


_TOTAL_LIABILITIES_RE = re.compile(r"total\s+liabilit", re.IGNORECASE)
_TOTAL_ASSETS_RE = re.compile(r"total\s+assets?", re.IGNORECASE)
_TOP_LEVEL_CODE_RE = re.compile(r"^\d+$")

# Matching keys off descriptive line_item text only -- item_code drifts across bulletin eras
# (e.g. LOANS & ADVANCES is coded "4" some years, "4.1" others; Sec 9.3) and is never used here.
# Matching also tolerates mid-word truncation seen in some extractions, e.g. "4 LOANS &" for
# "4 LOANS & ADVANCES" (Sec 9.5), via mutual-prefix comparison in _lookup_field.
#
# Lettered sub-items (a./b./c...) are reused across unrelated sections in the raw bulletin
# (Sec 9.4) -- and their descriptive suffix text is NOT always globally unique either: "e.
# Others" appears under both "3 DEPOSITS" and "8 OTHER ASSETS" (confirmed 2026-10-22). None of
# the 8 canonical labels below collide with each other, so matching each on full normalized
# text alone is still safe for them specifically. Re-verify if a future field's text collides.
FIELD_LABELS: dict[str, tuple[str, str]] = {
    "capital_fund": ("liabilities", "1 CAPITAL FUND"),
    "liquid_funds": ("assets", "1 LIQUID FUNDS"),
    "loans_advances": ("assets", "4 LOANS & ADVANCES"),
    "deposits": ("liabilities", "3 DEPOSITS"),
    "borrowings_total": ("liabilities", "2 BORROWINGS"),
    "borrowings_nrb": ("liabilities", "a. NRB"),
    "borrowings_interbank": ("liabilities", "b. Interbank Borrowing"),
    "interbank_lending_r": ("assets", "a. Interbank Lending"),
}

# The 10 real top-level asset rows (Sec 8b). "a. Interbank Lending" is deliberately NOT
# included -- it's a sub-item already folded into "3 SHARE & OTHER INVESTMENT"'s own reported
# total, so adding it again here would double-count it (Sec 9.4 fix).
ASSET_TOP_LEVEL_LABELS = [
    "1 LIQUID FUNDS",
    "2 INVESTMENT IN SECURITIES",
    "3 SHARE & OTHER INVESTMENT",
    "4 LOANS & ADVANCES",
    "5 BILL PURCHASED",
    "6 LOANS AGAINST COLLECTED BILLS",
    "7 FIXED ASSETS",
    "8 OTHER ASSETS",
    "9 EXPENSES NOT WRITTEN OFF",
    "10 NON BANKING ASSETS",
]

# Confirmed real sub-item text under "2 BORROWINGS" (Sec 8b/9.4), used as the fallback
# denominator when the "2 BORROWINGS" subtotal row itself isn't printed. "f."/"g." sub-items
# this doc once assumed existed are not confirmed in real data -- omitted rather than guessed,
# so the fallback may undercount for banks/periods that report them.
BORROWING_SUBITEMS = [
    "a. NRB",
    "b. Interbank Borrowing",
    "c. Foreign Banks and Fin. Ins.",
    "d. Other Financial Ins.",
    "e. Bonds and Securities",
]

COVARIATE_ORDER = ["size", "capital_ratio", "liquidity_ratio", "ldr", "nrb_dependence"]


def _normalize_label(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).upper()


# Letter reuse across unrelated sections isn't always safe to ignore after all -- "e. Others"
# appears both under "3 DEPOSITS" (liabilities) and "8 OTHER ASSETS" (assets) in real data
# (confirmed Asar-2074.pdf/NBL, 2026-10-22). infer_sides therefore only enforces side
# consistency for the labels actually read by _compute_fields, not every label in the
# bulletin -- collisions among unwatched labels are real but irrelevant to us.
_WATCHED_LABELS = {_normalize_label(lbl) for _, lbl in FIELD_LABELS.values()}


@dataclass
class _Row:
    source_file: str
    year_bs: int
    month_bs: str
    bfi_name: str
    item_code: str
    line_item: str
    value: float
    order: int


_SOURCE_FILE_PERIOD_RE = re.compile(r"Monthly_Statistics-(\d{4})-(\d{2})-")
# A second, punctuation-swapped numeric convention confirmed 2026-10-25 in a ~24-month
# contiguous run (2076-05 .. 2078-04, roughly Sep 2019 - Aug 2021 -- a different bulletin
# production era): "Monthly-Statistics-_-{year_bs}_{month_num:02d}..." (dash after "Monthly",
# "_-" before the year, underscore between year/month, vs. the primary pattern's underscore
# after "Monthly" and dash everywhere else). Same 1-indexed-from-Baisakh month numbering.
_SOURCE_FILE_PERIOD_RE_ALT = re.compile(r"Monthly-Statistics-_-(\d{4})_(\d{2})")
# A third convention, confirmed 2026-10-25 in the residual unrecoverable sample, spells the
# month as a Nepali-month-name abbreviation instead of a number, e.g. "Bhadau_2075Mid-Sep-
# 2018.pdf", "Saun-2074.pdf", "Chait-2074.pdf" -- never seen in the CSV's own month_bs column
# (that uses _MONTH_ALIASES' spellings instead), so kept as its own table rather than folded
# into _MONTH_ALIASES.
_SOURCE_FILE_MONTH_NAME_RE = re.compile(r"^([A-Za-z]+)[_-](\d{4})")
_SOURCE_FILE_MONTH_NAMES: dict[str, str] = {
    "Baisakh": "Baisakh", "Jestha": "Jestha", "Asar": "Asar", "Saun": "Shrawan",
    "Bhadau": "Bhadra", "Ashwin": "Ashwin", "Kattik": "Kartik", "Mangsir": "Mangsir",
    "Paush": "Poush", "Magh": "Magh", "Falgun": "Falgun", "Chait": "Chaitra",
}
_NEW_SUFFIX_RE = re.compile(r"^(.*)-NEW(\.\w+)$")


def _recover_period_from_source_file(source_file: str) -> tuple[int, str] | None:
    """Recovers (year_bs, month_bs) from source_file's own naming convention(s) -- confirmed
    2026-10-25 against Everest/Himalaya/Siddhartha/NIBL, whose rows are 94-98% blank-month_bs
    in the CSV column yet those banks are known continuously-operating commercial banks, not
    real entry/exit. Primary numeric convention's month numbering confirmed via the 2069-12 ->
    2070-01 (Chaitra -> Baisakh) BS-new-year rollover across consecutive real filenames. Three
    conventions are tried in order (see the regexes above); used only as a fallback when the
    CSV's own month_bs column is blank -- the column is trusted first since it's unambiguous
    when present.
    """
    m = _SOURCE_FILE_PERIOD_RE.search(source_file)
    if m is not None:
        year_bs, month_num = int(m.group(1)), int(m.group(2))
        if 1 <= month_num <= 12:
            return year_bs, BS_MONTHS[month_num - 1]

    m = _SOURCE_FILE_PERIOD_RE_ALT.search(source_file)
    if m is not None:
        year_bs, month_num = int(m.group(1)), int(m.group(2))
        if 1 <= month_num <= 12:
            return year_bs, BS_MONTHS[month_num - 1]

    m = _SOURCE_FILE_MONTH_NAME_RE.match(source_file)
    if m is not None:
        month_name, year_bs = m.group(1), int(m.group(2))
        canonical = _SOURCE_FILE_MONTH_NAMES.get(month_name)
        if canonical is not None:
            return year_bs, canonical

    return None


def _superseded_source_files(csv_path: Path) -> set[str]:
    """A '-NEW' bulletin filename (e.g. '...-Oct-2014-NEW.pdf') is a full resubmission of its
    non-NEW counterpart ('...-Oct-2014.pdf') for the same year/month -- confirmed both present
    in real data (2026-10-25). Treated as superseding the whole file, not a per-row conflict:
    every row from the non-NEW counterpart is dropped wherever both exist, so the panel never
    mixes rows from two versions of the same bulletin.
    """
    seen: set[str] = set()
    with open(csv_path, newline="") as f:
        for rec in csv.DictReader(f):
            seen.add(rec["source_file"])
    superseded: set[str] = set()
    for name in seen:
        m = _NEW_SUFFIX_RE.match(name)
        if m is not None:
            base = m.group(1) + m.group(2)
            if base in seen:
                superseded.add(base)
    return superseded


def read_rows(csv_path: Path) -> list[_Row]:
    superseded = _superseded_source_files(csv_path)
    rows: list[_Row] = []
    blank_month_count = 0
    recovered_count = 0
    superseded_count = 0
    total_count = 0
    with open(csv_path, newline="") as f:
        for i, rec in enumerate(csv.DictReader(f)):
            raw_value = rec["value_mn_npr"].strip()
            if raw_value == "":
                # Blank value means the row wasn't reported for this bank/period (seen on
                # granular leaf breakdown rows in the real data) -- skip it like any other
                # absent row, not a parse error. `i` is kept as the enumerate index over
                # the unfiltered reader so extraction order (used by infer_sides) is intact.
                continue
            source_file = rec["source_file"]
            if source_file in superseded:
                superseded_count += 1
                continue
            total_count += 1
            month_bs = _normalize_month(rec["month_bs"].strip())
            if month_bs == "":
                # Confirmed at large volume in real data (2026-10-25: ~472942/859996 rows
                # with a non-blank value have blank month_bs) -- mostly recoverable from
                # source_file's own naming convention rather than a genuine monthless row
                # (see _recover_period_from_source_file).
                recovered = _recover_period_from_source_file(source_file)
                if recovered is None:
                    blank_month_count += 1
                    continue
                recovered_count += 1
                year_bs, month_bs = recovered
            else:
                year_bs = int(rec["year_bs"])
            rows.append(
                _Row(
                    source_file=source_file,
                    year_bs=year_bs,
                    month_bs=month_bs,
                    bfi_name=rec["bfi_name"],
                    item_code=rec["item_code"],
                    line_item=rec["line_item"],
                    value=float(raw_value),
                    order=i,
                )
            )
    if superseded_count:
        warnings.warn(
            f"{superseded_count} rows dropped: belong to a source_file superseded by a "
            "'-NEW' resubmission of the same bulletin",
            stacklevel=2,
        )
    if recovered_count:
        warnings.warn(
            f"{recovered_count}/{total_count} rows had blank month_bs in the CSV but were "
            "recovered via source_file's own year/month naming convention",
            stacklevel=2,
        )
    if blank_month_count:
        warnings.warn(
            f"{blank_month_count}/{total_count} rows with a non-blank value have blank "
            "month_bs (after alias normalization) AND an unparseable source_file -- "
            "investigate before trusting panel coverage",
            stacklevel=2,
        )
    return rows


def infer_sides(rows: list[_Row]) -> dict[str, str]:
    """Map each _WATCHED_LABELS entry (normalized line_item text actually read by
    _compute_fields) to 'liabilities' / 'assets'. Unwatched labels are walked for
    their state-machine cues (markers, item_code resets) but never recorded or
    checked -- lettered sub-item text isn't reliably unique across sections after
    all ("e. Others" appears under both DEPOSITS and OTHER ASSETS, Sec 9.4 update).

    item_code restarts at 1 independently on each side of the balance sheet (Sec 9
    caveat 1), so it can't disambiguate alone. Side is instead recovered by
    replaying each (source_file, bfi_name) bank-block in original extraction order
    and flipping a liabilities->assets state machine across the 'TOTAL LIABILITIES'
    / 'TOTAL ASSETS' marker rows -- the same cue a human reader uses on the page.

    Grouping is by bank, not just by file: a single bulletin source_file holds
    every bank's balance sheet extracted row-major by line item (bank A's row for
    item N immediately followed by bank B's row for item N, not bank A's full
    sheet followed by bank B's full sheet), so TOTAL LIABILITIES appears once per
    bank back-to-back. A single state machine over the raw file order would flip
    to 'assets' on bank A's marker and then misread bank B's very next marker row
    as a side conflict. Each bank's own rows, re-sorted by their original `order`,
    still recover that bank's clean top-to-bottom cycle regardless of how banks are
    interleaved in the raw file.
    Pre-2075-era bulletins (confirmed 2026-10-22 against Asar-2074.pdf) extract with
    no 'TOTAL Liabilities' marker row at all -- the liabilities section just ends and
    the assets section starts back at item_code "1" with no boundary cue. For that
    case there's a fallback signal already named in the item_code-restart caveat
    above: top-level numeric item_codes (bare digits, e.g. "1".."10", as opposed to
    lettered/dotted sub-item codes like "a." or "1.") strictly increase within one
    side, so a drop back to a lower code while still in 'liabilities' state is itself
    the side boundary, used only when no explicit marker has fired yet.
    """
    by_bank: dict[tuple[str, str], list[_Row]] = defaultdict(list)
    for r in rows:
        by_bank[(r.source_file, r.bfi_name)].append(r)

    merged: dict[str, str] = {}
    for file_rows in by_bank.values():
        file_rows.sort(key=lambda r: r.order)
        side = "liabilities"
        last_top_level_code: int | None = None
        for r in file_rows:
            code = r.item_code.strip()
            if side == "liabilities" and _TOP_LEVEL_CODE_RE.match(code):
                code_num = int(code)
                if last_top_level_code is not None and code_num < last_top_level_code:
                    side = "assets"
                last_top_level_code = code_num

            label = _normalize_label(r.line_item)
            if label in _WATCHED_LABELS:
                if side != "done" and label not in merged:
                    merged[label] = side
                elif side != "done" and merged.get(label) not in (side, None):
                    raise ValueError(
                        f"inconsistent side for line_item {label!r}: "
                        f"{merged[label]!r} vs {side!r} in {r.source_file} ({r.bfi_name})"
                    )
            if _TOTAL_LIABILITIES_RE.search(r.line_item):
                side = "assets"
            elif _TOTAL_ASSETS_RE.search(r.line_item):
                side = "done"
    return merged


def _period_key(year_bs: int, month_bs: str) -> tuple[int, int]:
    return (year_bs, _MONTH_INDEX[month_bs])


@dataclass
class Panel:
    bfi_names: list[str]
    periods: list[tuple[int, str]]  # (year_bs, month_bs), chronological
    X: torch.Tensor      # (N, T, 5), covariate order = COVARIATE_ORDER
    r: torch.Tensor      # (N, T) interbank lending (row marginal)
    c: torch.Tensor      # (N, T) interbank borrowing (column marginal)
    K: torch.Tensor      # (N, T) raw capital fund, for kappa * K cap


# Confirmed (2026-10-25) commercial ("A" class) bank name/code prefixes actually present in
# bfi_name, via `awk -F, '{print $N}' nrb_balance_sheet.csv | grep -E '^(...)' | sort -u` against
# the real CSV -- NRB's monthly bulletin format is shared across commercial banks, development
# banks, finance companies, and microfinance institutions, and bfi_name alone carries no explicit
# BFI-class column, so prefix matching is the only way to scope the panel to commercial banks only
# (the standard population for an interbank-network/systemic-risk study, Sec 8/11).
#
# "Nepal" was in the original candidate pattern but only ever matched "NepalAawas" in the real
# data -- almost certainly Nepal Aawas Bikas Bank, historically a "B" class development bank, not
# commercial. Dropped from this list rather than silently including a bank that only matched by
# prefix accident; revisit if that bank's actual class is confirmed otherwise.
#
# Added 2026-10-25 after reviewing the full non-commercial dropped-bank list (banks total: 33,
# too low): "NSBI" (Nepal SBI Bank), "Lumbini" (Lumbini Bank, a commercial bank pre-merger --
# distinct from "LumbiniDB", a separate development-bank entity also present in the raw data) and
# "Prabhu" (Prabhu Bank -- distinct from "PRABHU", already listed, and "PrabhuDB", a separate
# development-bank entity; "PRABHU"/"Prabhu" both appearing suggests the same institution is
# split across two raw name castings in different bulletin eras -- added as its own literal
# entry rather than case-folding the whole match, since case-insensitive matching would also
# sweep in "PrabhuDB", "FewaDB", "GandakiDB", etc., which are genuinely different, non-commercial
# entities in this dataset's "...DB" naming convention). "CTBNL" (possibly Commerz and Trust Bank
# Nepal Ltd) was flagged but NOT added -- confirm before including.
COMMERCIAL_BANK_PREFIXES = {
    "ADBNL", "BOK", "CBL", "Century", "Citizen", "Citizens", "Civil", "EBL", "Everest",
    "Global", "HBL", "Himalaya", "Kumari", "Laxmi", "Lumbini", "MBL", "Mega", "NABIL", "NBB",
    "NBBL", "NBL", "NCC", "NIBL", "NIC", "NIMB", "NMB", "NSBI", "PRABHU", "Prabhu", "Prime",
    "RBB", "SBL", "SCBNL", "Sanima", "Siddhartha", "Sunrise",
}


def is_commercial_bank(bfi_name: str) -> bool:
    # "...DB" suffix marks a development-bank sibling of a commercial-bank prefix (e.g.
    # "LumbiniDB", "PrabhuDB" -- distinct institutions from "Lumbini"/"Prabhu"/"PRABHU",
    # see COMMERCIAL_BANK_PREFIXES comment) -- excluded before the prefix check below,
    # otherwise plain startswith() would also match these non-commercial siblings.
    if bfi_name.endswith("DB"):
        return False
    return any(bfi_name.startswith(prefix) for prefix in COMMERCIAL_BANK_PREFIXES)


def filter_commercial_banks(panel: Panel) -> Panel:
    """Scopes panel down to commercial banks only (Sec 8/11) -- see COMMERCIAL_BANK_PREFIXES.
    Warns with the dropped names rather than silently shrinking N, same convention as
    load_panel's other diagnostic warnings.
    """
    keep = [i for i, name in enumerate(panel.bfi_names) if is_commercial_bank(name)]
    dropped = sorted(name for name in panel.bfi_names if not is_commercial_bank(name))
    if dropped:
        # Full list, not truncated: a short preview can't distinguish "correctly excluded
        # non-commercial BFI" from "real commercial bank whose name just doesn't match any
        # COMMERCIAL_BANK_PREFIXES entry" (e.g. a spelling variant, same failure mode
        # _MONTH_ALIASES exists to cover for months) -- that distinction can only be made by
        # eyeballing every dropped name, not a sample of 10.
        warnings.warn(
            f"{len(dropped)}/{len(panel.bfi_names)} banks excluded as non-commercial (no match "
            f"in COMMERCIAL_BANK_PREFIXES): {dropped}",
            stacklevel=2,
        )
    idx = torch.tensor(keep, dtype=torch.long)
    return Panel(
        bfi_names=[panel.bfi_names[i] for i in keep],
        periods=panel.periods,
        X=panel.X[idx],
        r=panel.r[idx],
        c=panel.c[idx],
        K=panel.K[idx],
    )


# Confirmed (2026-10-28, diagnose via raw-file-vs-CSV diff) that extraction produces zero
# balance_sheet rows for 25 real monthly bulletins, clustered in two contiguous blocks: BS
# 2068-03 through 2069-12 (2011-2013, 22 months) and BS 2071-12 through 2072-02 (2015, 3
# months, overlapping the Apr-2015 earthquake) -- root cause not yet diagnosed (likely an
# older-era bulletin title/TOC layout find_section_pages/_find_commercial_bank_section_bounds
# doesn't match). Rather than root-cause the pre-2070s gap, scoped the study window to start
# at BS 2072-03 (Asar), the first period with a genuinely unbroken monthly run through the
# latest available data (BS 2083) -- ~10 consecutive years, decided sufficient for now.
MIN_PERIOD = (2072, "Asar")


def filter_from_period(panel: Panel, start: tuple[int, str] = MIN_PERIOD) -> Panel:
    """Scopes panel to periods >= `start` (inclusive), dropping the T axis accordingly.
    See MIN_PERIOD comment for why BS 2072-03 is the current study-window start.
    """
    start_key = _period_key(*start)
    keep = [
        t for t, (year, month) in enumerate(panel.periods)
        if _period_key(year, month) >= start_key
    ]
    dropped = [panel.periods[t] for t in range(len(panel.periods)) if t not in keep]
    if dropped:
        warnings.warn(
            f"{len(dropped)}/{len(panel.periods)} periods excluded as before {start}: {dropped}",
            stacklevel=2,
        )
    idx = torch.tensor(keep, dtype=torch.long)
    return Panel(
        bfi_names=panel.bfi_names,
        periods=[panel.periods[t] for t in keep],
        X=panel.X[:, idx],
        r=panel.r[:, idx],
        c=panel.c[:, idx],
        K=panel.K[:, idx],
    )


def load_panel(csv_path: Path) -> Panel:
    rows = read_rows(csv_path)
    sides = infer_sides(rows)

    for name, (expected_side, label) in FIELD_LABELS.items():
        actual_side = sides.get(_normalize_label(label))
        if actual_side is not None and actual_side != expected_side:
            raise ValueError(
                f"field {name!r} label {label!r} expected side {expected_side!r} "
                f"but inferred side {actual_side!r} -- bulletin layout may have changed"
            )

    # value[(bfi_name, period_key)][normalized_label] = value
    values: dict[tuple[str, tuple[int, int]], dict[str, float]] = defaultdict(dict)
    bfi_names: set[str] = set()
    for r in rows:
        pk = _period_key(r.year_bs, r.month_bs)
        values[(r.bfi_name, pk)][_normalize_label(r.line_item)] = r.value
        bfi_names.add(r.bfi_name)

    periods_seen = sorted({pk for (_, pk) in values.keys()})
    period_labels = _labels_for(periods_seen)
    names_sorted = sorted(bfi_names)

    N, T = len(names_sorted), len(periods_seen)
    X = torch.full((N, T, len(COVARIATE_ORDER)), float("nan"))
    r_t = torch.full((N, T), float("nan"))
    c_t = torch.full((N, T), float("nan"))
    K_t = torch.full((N, T), float("nan"))

    missing_field_count = 0
    missing_examples: list[str] = []
    zero_denominator_count = 0
    for i, name in enumerate(names_sorted):
        for t, pk in enumerate(periods_seen):
            rec = values.get((name, pk))
            if rec is None:
                continue
            try:
                x, r_val, c_val, k_val = _compute_fields(rec)
            except KeyError as exc:
                # A required field (capital_fund/liquid_funds/loans_advances/deposits) has no
                # row at all for this bfi/period -- confirmed present as a real extraction gap
                # even among institutions later dropped by filter_commercial_banks (2026-10-25:
                # 583 such cells surfaced once the month_bs recovery fix stopped hiding most
                # periods for most banks). load_panel builds a dense panel for EVERY raw
                # bfi_name, not just whatever subset a caller filters to afterward, so one
                # messy institution's gap shouldn't block the whole panel from loading -- left
                # as NaN (X/r/c/K are already NaN-initialized above) rather than a hard failure,
                # same convention as the zero-denominator case below.
                missing_field_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(f"{name} @ {period_labels[t]}: missing {exc}")
                continue
            except ZeroDivisionError:
                # size==0/deposits==0 is a legitimate real-data case (e.g. a non-deposit-taking
                # BFI), not an extraction defect like a genuinely missing field -- treated the
                # same as a bank with zero rows that period (left NaN, not a hard failure), but
                # counted/warned since it's silent otherwise and affects downstream complete-case
                # filtering (run_real_experiment.py).
                zero_denominator_count += 1
                continue
            X[i, t] = x
            r_t[i, t] = r_val
            c_t[i, t] = c_val
            K_t[i, t] = k_val

    if zero_denominator_count:
        warnings.warn(
            f"{zero_denominator_count} (bfi, period) cells have a zero size/deposits "
            "denominator and were left as NaN",
            stacklevel=2,
        )

    if missing_field_count:
        warnings.warn(
            f"{missing_field_count} (bfi, period) cells missing a required field were left "
            "as NaN, e.g.: " + "; ".join(missing_examples),
            stacklevel=2,
        )

    return Panel(bfi_names=names_sorted, periods=period_labels, X=X, r=r_t, c=c_t, K=K_t)


def _labels_for(periods_seen: list[tuple[int, int]]) -> list[tuple[int, str]]:
    month_by_index = {v: k for k, v in _MONTH_INDEX.items()}
    return [(year, month_by_index[midx]) for year, midx in periods_seen]


_LEADING_NUMERAL_RE = re.compile(r"^\d+(\.\d+)?\s+")


def _strip_leading_numeral(s: str) -> str:
    return _LEADING_NUMERAL_RE.sub("", s, count=1)


def _lookup_field(rec: dict[str, float], canonical: str) -> float:
    """Find canonical's value among rec's normalized labels, tolerating mid-word truncation
    in either direction (Sec 9.5) -- item_code is never consulted (Sec 9.3/9.4)."""
    canonical_n = _normalize_label(canonical)
    for label, value in rec.items():
        if label == canonical_n or label.startswith(canonical_n) or canonical_n.startswith(label):
            return value
    # Confirmed 2026-10-28: starting BS 2076 Shrawan, NRB restructured the asset-side numbering
    # so "4 LOANS & ADVANCES" became sub-item "4.1 LOANS & ADVANCES" under a merged parent "4"
    # (paired with new "4.2 BILL PURCHASED") -- the leading item-number is embedded in line_item
    # TEXT itself, not just item_code (which Sec 9.3/9.4 already correctly ignore), so "4 " vs
    # "4.1 " diverge at the 2nd character and the mutual-prefix match above never fires. Retry
    # once on the descriptive text alone, after stripping a leading "<n>" or "<n>.<n>" token from
    # both sides. Same startswith-either-way tolerance as above (not strict equality): confirmed
    # 2026-10-28 that Magh/Falgun/Chaitra 2079's bulletins additionally append a trailing
    # footnote-reference digit to several labels (e.g. "4.1 LOANS & ADVANCES 1"), so the
    # leading-numeral-stripped candidate must still match as a prefix, not exactly.
    canonical_stripped = _strip_leading_numeral(canonical_n)
    for label, value in rec.items():
        label_stripped = _strip_leading_numeral(label)
        if (
            label_stripped == canonical_stripped
            or label_stripped.startswith(canonical_stripped)
            or canonical_stripped.startswith(label_stripped)
        ):
            return value
    raise KeyError(canonical)


def _lookup_field_or_zero(rec: dict[str, float], canonical: str) -> float:
    try:
        return _lookup_field(rec, canonical)
    except KeyError:
        return 0.0


def _compute_fields(rec: dict[str, float]) -> tuple[torch.Tensor, float, float, float]:
    # _or_zero: several of the 10 categories are legitimately blank for banks that don't
    # report them (e.g. smaller BFIs often have no "9 EXPENSES NOT WRITTEN OFF" row), and
    # read_rows already drops blank cells entirely (Sec 9 blank-value note) -- a strict lookup
    # here would wrongly drop the whole bfi/period cell over an absent optional zero, not an
    # actually-incomplete record.
    size = sum(_lookup_field_or_zero(rec, lbl) for lbl in ASSET_TOP_LEVEL_LABELS)
    capital_fund = _lookup_field(rec, FIELD_LABELS["capital_fund"][1])
    liquid_funds = _lookup_field(rec, FIELD_LABELS["liquid_funds"][1])
    loans_advances = _lookup_field(rec, FIELD_LABELS["loans_advances"][1])
    deposits = _lookup_field(rec, FIELD_LABELS["deposits"][1])
    # borrowings_nrb/interbank_r/interbank_c are lettered sub-items, same category as the
    # BORROWING_SUBITEMS fallback below -- confirmed live in real data (1546 cells, e.g. ADBNL
    # across many periods) that "a. NRB" is omitted entirely rather than printed as 0 for a
    # bank/period with no NRB borrowing at all, same leaf-omission convention as the
    # ASSET_TOP_LEVEL_LABELS categories above. interbank_r/interbank_c feed the r/c marginals
    # directly (not just a covariate ratio), but a bank legitimately doing zero interbank
    # lending/borrowing that period is exactly the same "optional zero, not missing data" case.
    borrowings_nrb = _lookup_field_or_zero(rec, FIELD_LABELS["borrowings_nrb"][1])
    interbank_r = _lookup_field_or_zero(rec, FIELD_LABELS["interbank_lending_r"][1])
    interbank_c = _lookup_field_or_zero(rec, FIELD_LABELS["borrowings_interbank"][1])

    try:
        borrowings_total = _lookup_field(rec, FIELD_LABELS["borrowings_total"][1])
    except KeyError:
        # An absent subtotal row means it wasn't printed this bank/period (Sec 8b fallback
        # note), not that the whole cell is missing data -- reconstruct from sub-items.
        borrowings_total = sum(_lookup_field_or_zero(rec, lbl) for lbl in BORROWING_SUBITEMS)

    # size==0 / deposits==0 (confirmed present in real data, 2026-10-23) mean this bank/period
    # cell can't produce a finite ratio at all (no asset rows reported, or a BFI category that
    # takes no deposits) -- there's no principled single value to substitute, so this is treated
    # the same as a missing required field (raised, collected by load_panel's `missing` list)
    # rather than guessed.
    if size == 0:
        raise ZeroDivisionError("size is zero (no asset rows reported)")
    if deposits == 0:
        raise ZeroDivisionError("deposits is zero (non-deposit-taking BFI/period)")

    # Unlike size/deposits above, borrowings_total==0 IS well-defined: borrowings_nrb is a
    # strict subset of borrowings_total (Sec 8b), so borrowings_total==0 implies
    # borrowings_nrb==0 too -- "no borrowing of any kind" is legitimately zero NRB dependence,
    # not missing data.
    nrb_dependence = borrowings_nrb / borrowings_total if borrowings_total > 0 else 0.0

    x = torch.tensor([
        size,
        capital_fund / size,
        liquid_funds / size,
        loans_advances / deposits,
        nrb_dependence,
    ])
    return x, interbank_r, interbank_c, capital_fund