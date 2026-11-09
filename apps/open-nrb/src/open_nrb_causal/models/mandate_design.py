#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Build the Sec 1 mandate-timing design for the treatment-intensity DiD.

Turns the per-cell dose/exposure of Eq. (A) into a treatment-intensity DiD
around the capital-floor mandate.

Why not Eq. (A)'s contemporaneous dose_i,t directly (first real run,
2026-11-03): dose_i,t = floor - CAPITAL FUND_i,t / size_i,t, and CAPITAL FUND
mechanically absorbs retained profit and is drawn down by provisions/write-
offs -- so a regression of P&L outcomes on contemporaneous dose partly
recovers an accounting identity, not the mandate's effect, and has no
pre/post-mandate contrast at all. Instead:

    dose_pre_i       = mean of dose_i,t over the pre-mandate window (frozen)
    exposure_i,t     = sum_j W_ij,t * dose_pre_j        (Eq. B, frozen dose)
    y_i,t = beta * dose_pre_i * post_t + gamma * exposure_i,t * post_t
            + rho * exposure_i,t + delta' (Z_pre_i * post_t) + alpha_i + tau_t + e_i,t

The exposure level term keeps gamma identified off the pre/post change, not
off cross-sectional connectedness to low-capital banks; Z_pre_i * post_t
lets banks with different pre-mandate characteristics trend differently
post-mandate, instead of conditioning on post-treatment controls.
Transition months (pre_window_end < t <= effective_period, the parallel run)
are dropped from the pooled DiD and get their own event-study bin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys

import torch

try:
    from open_nrb.models.data_loader import BS_MONTHS, COVARIATE_ORDER, Panel, _period_key
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from models.data_loader import BS_MONTHS, COVARIATE_ORDER, Panel, _period_key  # type: ignore[no-redef]

try:
    from .propensity import estimate_propensity
    from .spillover import compute_exposure, horvitz_thompson_weights, stabilized_ht_weights
except ImportError:
    from propensity import estimate_propensity  # type: ignore[no-redef]
    from spillover import (  # type: ignore[no-redef]
        compute_exposure,
        horvitz_thompson_weights,
        stabilized_ht_weights,
    )

Period = tuple[int, str]

_SHRAWAN = BS_MONTHS.index("Shrawan")


def period_key(period: Period) -> tuple[int, int]:
    """Return Paper 1's data_loader._period_key on a (year_bs, month_bs) tuple."""
    return _period_key(*period)


def fiscal_year(period: Period) -> int:
    """Return the Nepali fiscal year (Shrawan..Asar) a period falls in.

    Labelled by its starting BS year: Shrawan 2073 .. Asar 2074 -> 2073.
    """
    year, month = period
    return year if BS_MONTHS.index(month) >= _SHRAWAN else year - 1


def previous_month(period: Period) -> Period:
    """Return the BS calendar month immediately before `period` (wrapping the year)."""
    year, month = period
    idx = BS_MONTHS.index(month)
    return (year - 1, BS_MONTHS[-1]) if idx == 0 else (year, BS_MONTHS[idx - 1])


@dataclass
class MandateTiming:
    """Index the pre-mandate / transition / post-mandate split of a panel's periods."""

    pre_end_idx: int  # last period of the pre-mandate window (inclusive)
    effective_idx: int  # mandate effective period; post = strictly after

    def post(self, t: int) -> torch.Tensor:
        """Return a (T,) 0/1 mask of post-mandate periods (strictly after effective_idx)."""
        return (torch.arange(t) > self.effective_idx).float()

    def transition(self, t: int) -> torch.Tensor:
        """Return a (T,) 0/1 mask of transition periods (pre_end_idx < t <= effective_idx)."""
        idx = torch.arange(t)
        return ((idx > self.pre_end_idx) & (idx <= self.effective_idx)).float()


def mandate_timing(periods: list[Period], pre_window_end: Period, effective_period: Period) -> MandateTiming:
    """Locate pre_window_end and effective_period in `periods` as a MandateTiming.

    Raises if the periods aren't chronological, either boundary is absent, the
    pre-window ends after the effective period, or no post cells remain.
    """
    keys = [period_key(p) for p in periods]
    if keys != sorted(keys):
        raise ValueError("panel periods are not in chronological order")
    for name, p in [("pre_window_end", pre_window_end), ("effective_period", effective_period)]:
        if p not in periods:
            raise ValueError(f"{name} {p} is not in the panel's periods ({periods[0]} .. {periods[-1]})")
    timing = MandateTiming(periods.index(pre_window_end), periods.index(effective_period))
    if timing.pre_end_idx > timing.effective_idx:
        raise ValueError(f"pre_window_end {pre_window_end} is after effective_period {effective_period}")
    if timing.effective_idx == len(periods) - 1:
        raise ValueError(f"effective_period {effective_period} is the panel's last period -- no post cells")
    return timing


def freeze_pre_mandate(x: torch.Tensor, timing: MandateTiming) -> torch.Tensor:
    """Freeze a (N, T) series to its per-bank pre-mandate mean, (N,).

    x_pre_i = mean over t <= pre_end_idx of x_i,t (NaN cells skipped). Raises if
    a bank has no finite pre-window cell -- no principled frozen value.
    """
    pre = x[:, : timing.pre_end_idx + 1]
    out = pre.nanmean(dim=1)
    if not torch.isfinite(out).all():
        bad = torch.nonzero(~torch.isfinite(out)).flatten().tolist()
        raise ValueError(f"banks at rows {bad} have no finite value in the pre-mandate window")
    return out


def event_bins(periods: list[Period], timing: MandateTiming) -> list[int | str]:
    """Assign each period an event-time bin relative to the mandate.

    One label per period, binned by fiscal year relative to the mandate:
    pre-window cells get k = FY(t) - FY(pre_window_end) - 1 (so the pre-window
    end's own fiscal year is -1, the default reference bin); transition cells
    get "transition"; post cells get k = FY(t) - FY(first post period) >= 0.
    Fiscal-year bins match the P&L's own annual (Shrawan..Asar) reporting cycle.
    """
    fy_pre = fiscal_year(periods[timing.pre_end_idx])
    fy_post = fiscal_year(periods[timing.effective_idx + 1])
    bins: list[int | str] = []
    for t_idx, p in enumerate(periods):
        if t_idx <= timing.pre_end_idx:
            bins.append(fiscal_year(p) - fy_pre - 1)
        elif t_idx <= timing.effective_idx:
            bins.append("transition")
        else:
            bins.append(fiscal_year(p) - fy_post)
    return bins


def period_indicator(flags) -> torch.Tensor:
    """Build a (1, T) float 0/1 row from one bool per period."""
    return torch.tensor([1.0 if f else 0.0 for f in flags]).unsqueeze(0)


def bin_indicator(bins: list[int | str], b: int | str) -> torch.Tensor:
    """Build a (1, T) indicator of the event_bins label b."""
    return period_indicator(bt == b for bt in bins)


def bank_size(panel: Panel) -> torch.Tensor:
    """Return (N, T) bank size = sum of top-level on-balance-sheet assets, NPR mn.

    From data_loader._compute_fields.
    """
    return panel.X[:, :, COVARIATE_ORDER.index("size")]


def positive_size(size: torch.Tensor) -> torch.Tensor:
    """Mask non-positive sizes to NaN so they never divide or take a log.

    size is an item sum from the PDF extraction (Sec 5a of the design doc); a non-positive
    value is an extraction error -> NaN, never a divisor or log argument.
    """
    return torch.where(size > 0, size, torch.tensor(float("nan"), dtype=size.dtype))


def pre_x_post_controls(controls_pre: dict[str, torch.Tensor], post: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build Z_pre_i x post_t interaction controls, each (N, T) (post is (1, T))."""
    return {f"{name}_pre_x_post": c.unsqueeze(1) * post for name, c in controls_pre.items()}


def decumulate_fiscal_ytd(y: torch.Tensor, periods: list[Period]) -> torch.Tensor:
    """Convert fiscal-YTD cumulative (N, T) to monthly flow.

    Shrawan's value is already a single month; any other month is y_t - y_{t-1},
    NaN if the immediately preceding calendar month isn't the previous panel column.
    """
    out = torch.full_like(y, float("nan"))
    for t_idx, p in enumerate(periods):
        if BS_MONTHS.index(p[1]) == _SHRAWAN:
            out[:, t_idx] = y[:, t_idx]
        elif t_idx > 0 and periods[t_idx - 1] == previous_month(p):
            out[:, t_idx] = y[:, t_idx] - y[:, t_idx - 1]
    return out


QUARTER_END_MONTHS = ("Ashwin", "Poush", "Chaitra", "Asar")


def quarterly_from_fiscal_ytd(y: torch.Tensor, periods: list[Period]) -> torch.Tensor:
    """Convert fiscal-YTD cumulative (N, T) to quarterly flow at quarter ends.

    Quarterly flow at quarter-end months only, NaN elsewhere. Ashwin's YTD value is already
    Q1; Poush/Chaitra/Asar subtract the previous quarter end of the same fiscal year (NaN if
    that month isn't in the panel). Never reads a non-quarter-end month, for the bulletins
    whose P&L is only reliable at quarter ends.
    """
    pos = {(fiscal_year(p), p[1]): i for i, p in enumerate(periods)}
    out = torch.full_like(y, float("nan"))
    for t_idx, p in enumerate(periods):
        if p[1] not in QUARTER_END_MONTHS:
            continue
        q = QUARTER_END_MONTHS.index(p[1])
        if q == 0:
            out[:, t_idx] = y[:, t_idx]
            continue
        prev = pos.get((fiscal_year(p), QUARTER_END_MONTHS[q - 1]))
        if prev is not None:
            out[:, t_idx] = y[:, t_idx] - y[:, prev]
    return out


def quarter_end_only(y: torch.Tensor, periods: list[Period]) -> torch.Tensor:
    """Keep (N, T) y at quarter-end months, NaN elsewhere (stocks on the quarterly outcomes' cells)."""
    q = torch.tensor([p[1] in QUARTER_END_MONTHS for p in periods]).unsqueeze(0)
    return torch.where(q, y, torch.tensor(float("nan")))


def outcome_ratio(y: torch.Tensor, periods: list[Period], size: torch.Tensor, quarterly: bool,
                  ytd_cumulative: bool) -> torch.Tensor:
    """Convert an extracted P&L outcome (N, T) into the size-scaled regression outcome.

    Quarterly flows (quarter-end YTD only), else de-cumulated monthly flows if the P&L is
    fiscal-YTD, then / size (NPR mn / NPR mn of assets: ROA-style ratio).
    """
    if quarterly:
        y = quarterly_from_fiscal_ytd(y, periods)
    elif ytd_cumulative:
        y = decumulate_fiscal_ytd(y, periods)
    return y / size


def ytd_cumulation_diagnostic(y: torch.Tensor, periods: list[Period]) -> float:
    """Diagnose whether the P&L is fiscal-YTD cumulative or monthly.

    Median over (bank, year) of |y_Shrawan| / |y_previous Asar|. If the P&L is
    fiscal-YTD cumulative, Shrawan is 1 month vs Asar's 12, so this is ~1/12 ~
    0.08; for monthly flows it's ~1. Run on a positive, stable line (interest
    income) before choosing pnl_ytd_cumulative.
    """
    pos = {p: i for i, p in enumerate(periods)}
    ratios = []
    for t_idx, p in enumerate(periods):
        if BS_MONTHS.index(p[1]) == _SHRAWAN and (p[0], "Asar") in pos:
            num, den = y[:, t_idx].abs(), y[:, pos[(p[0], "Asar")]].abs()
            ok = torch.isfinite(num) & torch.isfinite(den) & (den > 0)
            ratios.append(num[ok] / den[ok])
    if not ratios or torch.cat(ratios).numel() == 0:
        return math.nan
    return torch.cat(ratios).median().item()


@dataclass
class DesignInputs:
    """Hold every regressor and HT weight the DiD/event study need for one exposure network."""

    exposure: torch.Tensor  # (N, T) exposure_i,t = sum_j W_ij,t dose_pre_j
    dose_x_post: torch.Tensor  # (N, T)
    exposure_x_post: torch.Tensor  # (N, T)
    controls: dict[str, torch.Tensor]  # exposure level + Z_pre x post, each (N, T)
    weights: torch.Tensor  # (N, T) stabilized HT weights, 0 on transition cells (pooled DiD)
    event_weights: torch.Tensor  # (N, T) same weights, transition cells kept (event study)


def did_design(
    W: torch.Tensor,
    dose_pre: torch.Tensor,
    controls_pre: dict[str, torch.Tensor],
    timing: MandateTiming,
    gps_period_effects: bool = False,
) -> DesignInputs:
    """Build every regressor and HT weight for one exposure network W (T, N, N).

    Constructs, with post_t = timing.post and dose_b = dose_pre broadcast over t,
        exposure_i,t    = sum_j W_ij,t * dose_pre_j        (Eq. B, frozen dose)
        dose_x_post     = dose_pre_i * post_t
        exposure_x_post = exposure_i,t * post_t
        controls        = {exposure_level: exposure} + Z_pre_i * post_t
        weights         = stabilized_ht_weights(horvitz_thompson_weights(exposure, GPS)),
    with the pooled-DiD weights zeroed on transition cells (event_weights keep
    them). Call once with W_hat and once with W_baseline so the RAS robustness
    run gets propensity/HT weights estimated from its OWN exposure, not W_hat's.
    GPS model: exposure_i,t | (dose_pre_i, Z_pre_i), pre-treatment covariates only;
    gps_period_effects adds period intercepts + per-period sigma (propensity.py).
    """
    n = dose_pre.shape[0]
    t = W.shape[0]
    post = timing.post(t).unsqueeze(0)
    dose_b = dose_pre.unsqueeze(1).expand(n, t)
    exposure = compute_exposure(W, dose_b)

    gps_covariates = torch.stack(
        [dose_b, *[c.unsqueeze(1).expand(n, t) for c in controls_pre.values()]], dim=-1
    )
    weights = stabilized_ht_weights(horvitz_thompson_weights(
        exposure, estimate_propensity(exposure, gps_covariates, period_effects=gps_period_effects)))

    controls = {"exposure_level": exposure} | pre_x_post_controls(controls_pre, post)
    return DesignInputs(
        exposure=exposure,
        dose_x_post=dose_b * post,
        exposure_x_post=exposure * post,
        controls=controls,
        weights=weights * (1.0 - timing.transition(t)).unsqueeze(0),
        event_weights=weights,
    )


WEIGHTINGS = ("unweighted", "ht")  # mirrored in experiment_config, which stays torch-free


def apply_weighting(design: DesignInputs, weighting: str) -> DesignInputs:
    """Select the estimator's cell weights for the given weighting scheme.

    "ht" keeps did_design's stabilised HT weights; "unweighted" puts 1 on every cell those
    weights keep (same sample, transition still 0 in the pooled DiD). Unweighted TWFE is the
    main estimator: with 16 banks the HT weights can't achieve overlap (design doc Sec 5b
    item 8).
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting {weighting!r} not in {WEIGHTINGS}")
    if weighting == "ht":
        return design
    return replace(design, weights=(design.weights > 0).to(design.weights.dtype),
                   event_weights=(design.event_weights > 0).to(design.event_weights.dtype))
