#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Build the Sec 4a/4b/4d variable groups from an aligned commercial-bank Panel.

Builds the three variable groups design doc Sec 4a/4b/4d need, aligned to
an already-loaded commercial-bank Panel (panel_alignment.align_to_artifact).

Sec 4e (Statement of Credit, separate 2-year file) is explicitly deferred --
not built here. Sec 4c (the spillover exposure itself) lives in spillover.py,
not here, since it needs W_hat, not just the raw panel.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import warnings

import torch

import sys

try:
    from open_nrb.models.data_loader import COVARIATE_ORDER, Panel, _normalize_label, read_rows
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from models.data_loader import (  # type: ignore[no-redef]
        COVARIATE_ORDER,
        Panel,
        _normalize_label,
        read_rows,
    )

try:
    from .panel_alignment import capital_ratio
except ImportError:
    from panel_alignment import capital_ratio  # type: ignore[no-redef]


# Sec 4a: dose_i,t = capital_floor - capital_ratio_i,t (positive => below the new floor).
#
# capital_floor = 0.04, the Basel III leverage ratio minimum (NRB "Capital Adequacy
# Framework, 2015" Sec 2.7.B.a: minimum Tier 1 leverage ratio of 4% during the parallel
# run, Mid July 2016 - Mid July 2018). Chosen over the Total/CET1 Capital Ratio floors
# (Sec 2.4, Sec 1.8's 8.5%-11% phase-in table) because those are defined against
# risk-weighted exposure, which open_nrb never computes (it would require replicating
# Sec 3-5's full credit/operational/market risk-weighting rules) -- whereas Paper 1's
# `capital_ratio` covariate (CAPITAL FUND / size, size = sum of on-balance-sheet assets,
# data_loader.py:596,637) is structurally the leverage ratio's denominator, not the CAR's.
#
# Known approximation, not an exact replication -- state as a limitation in the paper:
#   1. Numerator mismatch: Paper 1's `capital_ratio` uses CAPITAL FUND (Tier 1 + Tier 2),
#      but the regulatory leverage ratio is Tier 1 only (Sec 2.7.B.I). This overstates
#      the numerator relative to the true leverage ratio for any bank holding Tier 2
#      capital, biasing `dose` downward (understating distance below the floor).
#   2. Denominator mismatch: "size" (sum of ASSET_TOP_LEVEL_LABELS) is not exactly the
#      regulatory "Exposure Measure" (Sec 2.7.B.II), which nets specific provisions/
#      valuation adjustments and adds off-balance-sheet items at a 100% CCF -- not
#      replicated here.
#   3. Period applicability: the 4% figure is only explicitly mandated for the parallel
#      run (mid-2016 to mid-2018, Sec 2.7.B.a); "the final leverage ratio requirement
#      might be revised by NRB" afterward, with no later figure given in this document.
#      Applied here as a constant across the full panel (Paper 1's panel starts 2072
#      Asar, before the mandate) since no revised post-2018 figure is available --
#      this is a modeling choice, not a verified post-2018 regulatory fact.
#
# Effective date: Mid July 2016 = Asar end 2073 BS (NRB Capital Adequacy Framework 2015,
# Sec 1.9); parallel run Poush end 2072 BS -> Asar end 2073 BS. NOT "Asar 2074 BS" -- an
# earlier background-agent draft proposed an unverified "NPR 8bn / Asar 2074 BS" figure
# (a different quantity -- paid-up capital, not a leverage/CAR ratio -- and not present
# anywhere in the actual framework document) that was explicitly rejected.
CAPITAL_FLOOR: float = 0.04


def compute_dose(panel: Panel, capital_floor: float) -> torch.Tensor:
    """Compute (N, T) dose_i,t = capital_floor - capital_ratio_i,t."""
    return capital_floor - capital_ratio(panel)


@dataclass
class OutcomePanel:
    """Hold the Sec 4b P&L outcomes per (bank, period): net profit/loss, provisions, write-offs, NIM."""

    bfi_names: list[str]
    periods: list[tuple[int, str]]
    net_profit_loss: torch.Tensor  # (N, T), net profit minus net loss
    loan_loss_provision: torch.Tensor  # (N, T), sum of 7.1.1/7.1.2/7.1.3
    loan_written_off: torch.Tensor  # (N, T)
    nim: torch.Tensor  # (N, T), interest income - interest expense


OUTCOME_FIELDS = ["net_profit_loss", "loan_loss_provision", "loan_written_off", "nim"]

# Sec 4b line items as (leading item code, required substring of the normalized
# line_item). Only "1. Interest Expense" is confirmed against a real extracted row
# (Asar-2074.pdf); the rest follow design doc Sec 4b's NRB field list and are
# verified at load time -- an outcome with no matched rows raises (and any single
# unmatched component warns) with the real labels found under that code, so a wrong
# guess fails loudly instead of silently yielding all-NaN.
PNL_FIELD_MATCHERS: dict[str, tuple[str, str | None]] = {
    "interest_expense": ("1", "INTEREST EXPENSE"),
    "interest_income": ("1", "INTEREST INCOME"),
    "loan_loss_provision_1": ("7.1.1", None),
    "loan_loss_provision_2": ("7.1.2", None),
    "loan_loss_provision_3": ("7.1.3", None),
    "loan_written_off": ("8", "WRITTEN OFF"),
    "net_profit": ("12", "NET PROFIT"),
    "net_loss": ("9", "NET LOSS"),
}


def _leading_code(normalized_label: str) -> str:
    return normalized_label.split(" ", 1)[0].rstrip(".")


def match_pnl_field(normalized_label: str) -> str | None:
    """Return the PNL_FIELD_MATCHERS field a normalized line-item label maps to, or None."""
    code = _leading_code(normalized_label)
    for field, (want_code, keyword) in PNL_FIELD_MATCHERS.items():
        if code == want_code and (keyword is None or keyword in normalized_label):
            return field
    return None


def check_pnl_csv_exists(pnl_csv_path: Path) -> None:
    """Raise FileNotFoundError with a locating hint if the P&L CSV is absent."""
    if not pnl_csv_path.exists():
        raise FileNotFoundError(
            f"{pnl_csv_path} does not exist -- expected open_nrb.utils.pdf_extract's "
            "nrb_profit_and_loss.csv, alongside nrb_balance_sheet.csv in "
            "src/open_nrb/data/processed/monthly_statistical_bfis/."
        )


def load_outcome_panel(pnl_csv_path: Path, panel: Panel) -> OutcomePanel:
    """Read the P&L CSV into an OutcomePanel keyed to panel's banks and periods.

    Uses Paper 1's own read_rows (same month-alias normalization, blank-month_bs
    recovery, and -NEW supersession as the balance sheet). Values are taken as
    reported -- if NRB's bulletin P&L is fiscal-year-to-date cumulative, that is
    NOT de-cumulated here.
    """
    check_pnl_csv_exists(pnl_csv_path)

    n, t = len(panel.bfi_names), len(panel.periods)
    values = {field: torch.full((n, t), float("nan")) for field in PNL_FIELD_MATCHERS}
    name_pos = {name: i for i, name in enumerate(panel.bfi_names)}
    period_pos = {period: t_idx for t_idx, period in enumerate(panel.periods)}

    seen: dict[tuple[str, int, int], tuple[str, float]] = {}
    labels_by_code: dict[str, set[str]] = defaultdict(set)
    conflicts: list[str] = []

    for row in read_rows(pnl_csv_path):
        label = _normalize_label(row.line_item)
        labels_by_code[_leading_code(label)].add(label)
        field = match_pnl_field(label)
        if field is None:
            continue
        i = name_pos.get(row.bfi_name)
        t_idx = period_pos.get((row.year_bs, row.month_bs))
        if i is None or t_idx is None:
            continue
        key = (field, i, t_idx)
        prev = seen.get(key)
        if prev is not None:
            if prev != (label, row.value):
                conflicts.append(
                    f"{field} {row.bfi_name} {row.year_bs}-{row.month_bs}: "
                    f"{prev} vs {(label, row.value)} ({row.source_file})"
                )
            continue
        seen[key] = (label, row.value)
        values[field][i, t_idx] = row.value

    if conflicts:
        raise ValueError(
            f"{len(conflicts)} (field, bank, period) cells matched more than one distinct "
            "P&L row -- tighten PNL_FIELD_MATCHERS. First few: " + "; ".join(conflicts[:5])
        )

    def _labels_hint(field: str) -> str:
        code = PNL_FIELD_MATCHERS[field][0]
        return (f"{field} wants {PNL_FIELD_MATCHERS[field]}, labels found under code {code!r}: "
                f"{sorted(labels_by_code.get(code, set()))[:10]}")

    unmatched = [field for field, v in values.items() if v.isnan().all()]
    if unmatched:
        warnings.warn(
            "P&L fields matched no rows for any panel bank/period: "
            + "; ".join(_labels_hint(f) for f in unmatched),
            stacklevel=2,
        )

    def _sum_present(*tensors: torch.Tensor) -> torch.Tensor:
        all_missing = torch.stack([x.isnan() for x in tensors]).all(dim=0)
        total = sum(x.nan_to_num(0.0) for x in tensors)
        return torch.where(all_missing, torch.tensor(float("nan")), total)

    outcomes = OutcomePanel(
        bfi_names=panel.bfi_names,
        periods=panel.periods,
        net_profit_loss=_sum_present(values["net_profit"], -values["net_loss"]),
        loan_loss_provision=_sum_present(
            values["loan_loss_provision_1"],
            values["loan_loss_provision_2"],
            values["loan_loss_provision_3"],
        ),
        loan_written_off=values["loan_written_off"],
        nim=values["interest_income"] - values["interest_expense"],
    )

    empty = [name for name in OUTCOME_FIELDS if getattr(outcomes, name).isnan().all()]
    if empty:
        raise ValueError(
            f"outcomes with no data for any panel bank/period: {empty} -- "
            + "; ".join(_labels_hint(f) for f in unmatched)
        )
    return outcomes


def pnl_reported(outcomes: OutcomePanel) -> torch.Tensor:
    """Return a (N, T) bool mask: the bank has at least one finite P&L outcome that month."""
    return torch.stack([getattr(outcomes, name).isfinite() for name in OUTCOME_FIELDS]).any(dim=0)


def zero_fill_blanks(outcomes: OutcomePanel, fields: list[str] | tuple[str, ...]) -> OutcomePanel:
    """Zero-fill blank outcome cells where the bank reported other P&L that month.

    Sec 5b item 11b sensitivity: a blank in `fields` becomes 0 where the bank reported some
    other P&L line that month (read as "nothing to report"), and stays NaN where it reported no
    P&L at all. Values are the extracted (possibly fiscal-YTD) figures, before any differencing.
    """
    reported = pnl_reported(outcomes)
    filled = {name: torch.where(getattr(outcomes, name).isnan() & reported, 0.0, getattr(outcomes, name))
              for name in fields}
    return OutcomePanel(bfi_names=outcomes.bfi_names, periods=outcomes.periods,
                        **{name: filled.get(name, getattr(outcomes, name)) for name in OUTCOME_FIELDS})


# Alternative dose (Sec 5b): the paid-up capital sub-item under "1 CAPITAL FUND". Not
# confirmed against a real extracted row -- matched on keywords, and load_paid_up_capital
# returns every matched label (and every label mentioning CAPITAL) so the match can be checked.
PAID_UP_KEYWORDS = ("PAID", "CAPITAL")


@dataclass
class PaidUpCapital:
    """Hold extracted paid-up capital (N, T) with its match provenance and conflict count."""

    values: torch.Tensor  # (N, T), NPR mn as extracted, NaN where no row matched
    matched_labels: dict[str, int]  # normalized label -> panel cells it filled
    capital_labels: list[str]  # every normalized label containing "CAPITAL" (match hints)
    n_conflicts: int  # panel cells where a second, different matched value was ignored


def load_paid_up_capital(csv_path: Path, panel: Panel) -> PaidUpCapital:
    """Load paid-up capital per (bank, period) of `panel` from the balance-sheet CSV.

    Via Paper 1's read_rows. First matched row per cell is kept; later different values are
    counted, not raised, so the reviewer check can report them.
    """
    n, t = len(panel.bfi_names), len(panel.periods)
    values = torch.full((n, t), float("nan"))
    name_pos = {name: i for i, name in enumerate(panel.bfi_names)}
    period_pos = {period: t_idx for t_idx, period in enumerate(panel.periods)}
    matched: dict[str, int] = defaultdict(int)
    capital_labels: set[str] = set()
    n_conflicts = 0
    for row in read_rows(csv_path):
        label = _normalize_label(row.line_item)
        if "CAPITAL" in label:
            capital_labels.add(label)
        if not all(k in label for k in PAID_UP_KEYWORDS):
            continue
        i = name_pos.get(row.bfi_name)
        t_idx = period_pos.get((row.year_bs, row.month_bs))
        if i is None or t_idx is None:
            continue
        if values[i, t_idx].isnan():
            values[i, t_idx] = row.value
            matched[label] += 1
        elif values[i, t_idx].item() != row.value:
            n_conflicts += 1
    return PaidUpCapital(values=values, matched_labels=dict(matched), capital_labels=sorted(capital_labels),
                         n_conflicts=n_conflicts)


# Sec 4d controls: liquidity ratio, loan-to-deposit, NRB-dependence already live in
# panel.X (COVARIATE_ORDER) -- reused directly rather than reloaded. Deposit
# composition, effective tax rate, wholesale access, and foreign-interbank split are
# NOT in panel.X; same "don't assume, check first" rule as PNL_FIELD_MATCHERS applies
# to whichever CSV/field eventually supplies them.
CONTROLS_IN_PANEL_X = ["liquidity_ratio", "ldr", "nrb_dependence"]
CONTROLS_NOT_YET_SOURCED = [
    "deposit_composition",
    "effective_tax_rate",
    "wholesale_access",
    "foreign_interbank_split",
]


def controls_from_panel(panel: Panel) -> dict[str, torch.Tensor]:
    """Return the Sec 4d controls already in panel.X, each (N, T), keyed by name."""
    return {name: panel.X[:, :, COVARIATE_ORDER.index(name)] for name in CONTROLS_IN_PANEL_X}
