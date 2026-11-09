"""Estimate the headline dose-only total effect on quarterly outcomes.

Headline beta (user decision 2026-11-03, design doc Sec 5b item 3): the dose-only total
effect on quarterly outcomes from Baisakh 2070.

Eq. A' conditions on exposure built from the post-mandate W_hat, which the mandate can itself
move, so for beta those terms are post-treatment controls. Dropping them gives the total
effect of dose, and since no W_hat is needed the panel can start before Asar 2072: the
2070-2072 bulletins are clean (Sec 5a), which buys a pre-trend lead. Eq. A' is kept for gamma.

Same 16 banks, frozen dose_pre and Z_pre x post controls as the main run; bank + period FE;
always unweighted (the HT weights need the exposure). Pooled regressors: dose x lead (pre bins
before the reference bin -1), dose x transition, dose x post.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

try:
    from .estimator import fit_twfe
    from .mandate_design import (
        bank_size, event_bins, mandate_timing, outcome_ratio, period_indicator, period_key, positive_size,
        pre_x_post_controls, previous_month, quarter_end_only)
    from .panel_alignment import Panel, load_commercial_panel, reindex_panel
    from .variables import OUTCOME_FIELDS, load_outcome_panel, zero_fill_blanks
except ImportError:
    from estimator import fit_twfe  # type: ignore[no-redef]
    from mandate_design import (  # type: ignore[no-redef]
        bank_size, event_bins, mandate_timing, outcome_ratio, period_indicator, period_key, positive_size,
        pre_x_post_controls, previous_month, quarter_end_only)
    from panel_alignment import Panel, load_commercial_panel, reindex_panel  # type: ignore[no-redef]
    from variables import OUTCOME_FIELDS, load_outcome_panel, zero_fill_blanks  # type: ignore[no-redef]

TOTAL_EFFECT_START = (2070, "Baisakh")
REFERENCE_BIN = -1


def extended_panel(csv_path, bfi_names: list[str], last_period) -> Panel:
    """Build the extended commercial-bank panel from TOTAL_EFFECT_START to last_period.

    Commercial banks reindexed to bfi_names (panel_alignment.reindex_panel).
    Paper 1's loaders are called, not edited.
    """
    full = load_commercial_panel(csv_path, TOTAL_EFFECT_START)
    missing = [b for b in bfi_names if b not in full.bfi_names]
    if missing:
        raise ValueError(f"estimation banks absent from the extended panel: {missing}")
    end_key = period_key(last_period)
    return reindex_panel(full, bfi_names, [p for p in full.periods if period_key(p) <= end_key])


def missing_months(periods) -> list:
    """List the calendar months absent between consecutive panel periods.

    For each adjacent pair, walks back month by month from the later period and
    collects every month strictly between them (the gaps the extraction skipped).
    """
    gaps = []
    for t in range(1, len(periods)):
        run, p = [], previous_month(periods[t])
        while period_key(p) > period_key(periods[t - 1]):
            run.append(p)
            p = previous_month(p)
        gaps.extend(reversed(run))
    return gaps


@dataclass
class TotalEffectResult:
    """Hold the headline total-effect estimate: dose x post beta, pre-trend lead, and extras.

    beta_dose / se_dose / p_dose are the dose x post headline; beta_lead / se_lead /
    p_lead the pre-trend check; with pre_mean, effect_1sd, dropped regressors, and
    any extra-dose or trend-control results.
    """

    beta_dose: float  # dose x post: the headline beta
    se_dose: float
    p_dose: float | None
    beta_lead: float  # dose x lead: pre-trend check, ~0 under parallel trends
    se_lead: float
    p_lead: float | None
    n_obs: int
    n_clusters: int
    pre_mean: float  # mean quarterly outcome over the pre cells (lead + reference)
    effect_1sd: float  # beta_dose x SD(dose_pre)
    dropped_regressors: list[str] = field(default_factory=list)
    note: str = ""
    # extra_doses' post terms: name -> {"beta", "se", "p", "effect_1sd"}
    extra: dict[str, dict] = field(default_factory=dict)
    # trend_controls' x trend terms: name -> {"beta", "se"} (per year); post_centre = mean post
    # period in years since the pre-window end (where dose_post is evaluated under trend_fit="pre")
    trend: dict[str, dict] = field(default_factory=dict)
    post_centre: float = math.nan


@dataclass
class TotalEffectPanel:
    """Describe the extended estimation panel: its span, length, and missing months."""

    start: tuple
    end: tuple
    n_periods: int
    missing_months: list


def _nan_result(note: str) -> TotalEffectResult:
    return TotalEffectResult(math.nan, math.nan, None, math.nan, math.nan, None, 0, 0, math.nan, math.nan,
                             note=note)


# Extended panel + outcomes per (inputs, zero-fill), so repeated fits (leave-one-out) read the
# CSVs once.
_PANEL_CACHE: dict[tuple, tuple] = {}


def _panel_and_outcomes(cfg, inp, zero_fill: tuple[str, ...]):
    key = (str(cfg.csv_path), str(cfg.pnl_csv_path), tuple(inp.bfi_names), tuple(inp.periods[-1]), zero_fill)
    if key not in _PANEL_CACHE:
        panel = extended_panel(cfg.csv_path, inp.bfi_names, inp.periods[-1])
        outcomes = load_outcome_panel(cfg.pnl_csv_path, panel)
        if zero_fill:
            outcomes = zero_fill_blanks(outcomes, zero_fill)
        _PANEL_CACHE[key] = (panel, outcomes)
    return _PANEL_CACHE[key]


def years_since_pre_end(periods, timing) -> torch.Tensor:
    """Compute (1, T) time in years since the pre-window end (negative before it).

    For each period, ((year - y0) * 12 + (month - m0)) / 12 -- months since the
    pre-window-end period (y0, m0), expressed in years.
    """
    y0, m0 = period_key(periods[timing.pre_end_idx])
    return torch.tensor([((y - y0) * 12 + (m - m0)) / 12 for y, m in map(period_key, periods)]).unsqueeze(0)


def total_effect_series(cfg, inp, zero_fill: tuple[str, ...] | None = None, fields: list[str] | None = None,
                        denominator: torch.Tensor | None = None,
                        stock_outcomes: dict[str, torch.Tensor] | None = None):
    """Build the (panel, timing, outcomes) fit_total_effect regresses, for reviewer checks.

    Returns (panel, timing, {name: (N, T) outcome}) exactly as fit_total_effect
    regresses them, so a reviewer check can run its own regression on the same
    cells. Each outcome is outcome_ratio(y, periods, size, quarterly=True,
    ytd_cumulative=True) = quarterly_from_fiscal_ytd(y) / size (the quarter-end
    YTD flow scaled by bank size); stock_outcomes are instead kept at quarter
    ends only (quarter_end_only), undifferenced and unscaled. Arguments as in
    fit_total_effect.
    """
    if not cfg.pnl_ytd_cumulative:
        raise ValueError("pnl_ytd_cumulative is false; quarterly differencing assumes fiscal-YTD figures")
    zero_fill = tuple(cfg.blank_as_zero) if zero_fill is None else tuple(zero_fill)
    panel, outcomes = _panel_and_outcomes(cfg, inp, zero_fill)
    timing = mandate_timing(panel.periods, cfg.pre_window_end, cfg.effective_period)
    periods, t = panel.periods, len(panel.periods)
    if stock_outcomes is not None:
        series = {}
        for name, s in stock_outcomes.items():
            if tuple(s.shape) != (inp.n, t):
                raise ValueError(f"stock outcome {name} has shape {tuple(s.shape)}, expected {(inp.n, t)}")
            series[name] = quarter_end_only(s, periods)
        return panel, timing, series
    if denominator is None:
        size = positive_size(bank_size(panel))
    else:
        size = positive_size(denominator).unsqueeze(1).expand(inp.n, t)
    return panel, timing, {name: outcome_ratio(getattr(outcomes, name), periods, size, True, True)
                           for name in fields or OUTCOME_FIELDS}


def _broken_trend_terms(trend_controls, years, timing, t, regs) -> tuple[dict[str, torch.Tensor], float]:
    """Own level and slope in transition and in post for each trend control, so its x trend
    slope is identified from the pre cells only. A level already in regs (dose_post,
    log_size_pre_x_post, ...) is not entered twice: fit_twfe's pinv would split it silently."""
    out: dict[str, torch.Tensor] = {}
    centres = {}
    for phase, ind in (("transition", timing.transition(t)), ("post", timing.post(t))):
        centres[phase] = years[0, ind > 0].mean().item()
        for name, v in trend_controls.items():
            level = v.unsqueeze(1) * ind.unsqueeze(0)
            if not any(r.shape == level.shape and torch.equal(r, level) for r in regs.values()):
                out[f"{name}_x_{phase}"] = level
            out[f"{name}_x_{phase}_slope"] = level * (years - centres[phase])
    return out, centres["post"]


def fit_total_effect(cfg, inp, n_boot: int, seed: int, zero_fill: tuple[str, ...] | None = None,
                     fields: list[str] | None = None, extra_doses: dict[str, torch.Tensor] | None = None,
                     denominator: torch.Tensor | None = None, drop_bank: int | None = None,
                     stock_outcomes: dict[str, torch.Tensor] | None = None,
                     trend_controls: dict[str, torch.Tensor] | None = None, trend_fit: str = "panel",
                     ) -> tuple[TotalEffectPanel, dict[str, TotalEffectResult]]:
    """Fit the headline dose-only total effect of dose_pre on quarterly outcomes.

    Estimates, unweighted with bank + period FE,
        y_i,t = beta_lead (dose_pre_i * lead_t) + beta_trans (dose_pre_i * transition_t)
                + beta (dose_pre_i * post_t) + delta' (Z_pre_i * post_t)
                + alpha_i + tau_t + e_i,t,
    where beta (the dose x post coefficient) is the headline total effect and
    beta_lead is the pre-trend check (lead_t marks pre bins before the reference
    bin -1). extra_doses add parallel lead/transition/post terms per extra frozen
    dose. trend_controls add a term trend_control_i * years_t (years since the
    pre-window end); trend_fit="pre" identifies each trend slope from the pre
    cells only (own level and slope in transition and post), so dose_post becomes
    the deviation from the extrapolated pre trend evaluated at post_centre = mean
    over post cells of years_since_pre_end, and the dose lead terms are dropped
    (beta_lead NaN).

    Quarterly outcomes only (the monthly P&L is unreliable off quarter ends, item 11a), so this
    needs pnl_ytd_cumulative; raises ValueError otherwise or if the extended panel can't be built.
    zero_fill: outcome fields whose blanks are read as 0 (variables.zero_fill_blanks, item 11b);
    None = cfg.blank_as_zero, () = every blank missing.
    The dose is inp.dose_pre, so a reviewer check can pass an alternative one via
    dataclasses.replace(inp, dose_pre=...).
    Reviewer-check options, default off: extra_doses = more (N,) frozen doses, each with its own
    lead / transition / post terms, fitted jointly with dose_pre (post terms in result.extra);
    denominator = (N,) per-bank divisor replacing current size (item 11c: frozen pre-window
    size); drop_bank = row index whose outcomes are set NaN (leave-one-out); stock_outcomes =
    balance-sheet stocks (N, T) on extended_panel's periods, used as given (no differencing, no
    size scaling) and kept at quarter-end months only so the cells match the P&L outcomes';
    trend_controls = (N,) per-bank vectors, each entered x a linear time trend (years since the
    pre-window end). trend_fit="panel": one slope over the whole panel, so dose_post is a shift
    beyond a dose-specific linear trend (a ramping effect is partly absorbed). trend_fit="pre":
    each trend control also gets its own level and slope in transition and in post, so the slope
    comes from the pre cells only and dose_post is the deviation from the extrapolated pre trend
    at the mean post date (result.post_centre); the dose lead terms are left out (beta_lead NaN).
    With stock_outcomes, fields and denominator are ignored."""
    if trend_fit not in ("panel", "pre"):
        raise ValueError(f"trend_fit must be 'panel' or 'pre', got {trend_fit!r}")
    panel, timing, series = total_effect_series(cfg, inp, zero_fill, fields, denominator, stock_outcomes)
    periods, t = panel.periods, len(panel.periods)
    bins = event_bins(periods, timing)
    post = timing.post(t).unsqueeze(0)
    lead = period_indicator([i <= timing.pre_end_idx and bins[i] < REFERENCE_BIN for i in range(t)])
    pre_cols = [i for i in range(t) if i <= timing.pre_end_idx and bins[i] <= REFERENCE_BIN]
    # dose_pre comes from inp: freezing on this panel would average from TOTAL_EFFECT_START instead.
    doses = {"dose": inp.dose_pre} | (extra_doses or {})
    # Under a pre-fitted trend the lead dummy would soak up the early-vs-reference contrast the
    # slope is identified from, so it is left out (beta_lead NaN): the trend replaces the lead test.
    pre_trend = bool(trend_controls) and trend_fit == "pre"
    regs: dict[str, torch.Tensor] = {}
    for dname, dvec in doses.items():
        dose_b = dvec.unsqueeze(1).expand(inp.n, t)
        if not pre_trend:
            regs[f"{dname}_lead"] = dose_b * lead
        regs |= {f"{dname}_transition": dose_b * timing.transition(t).unsqueeze(0), f"{dname}_post": dose_b * post}
    regs |= pre_x_post_controls(inp.controls_pre, post)
    years = years_since_pre_end(periods, timing)
    post_centre = years[0, timing.post(t) > 0].mean().item()
    if trend_controls:
        regs |= {f"{name}_x_trend": v.unsqueeze(1) * years for name, v in trend_controls.items()}
        if pre_trend:
            broken, post_centre = _broken_trend_terms(trend_controls, years, timing, t, regs)
            regs |= broken
    sd = {dname: dvec.std().item() for dname, dvec in doses.items()}
    boot = ([] if pre_trend else ["dose_lead"]) + ["dose_post"] + [f"{k}_post" for k in extra_doses or {}]

    results: dict[str, TotalEffectResult] = {}
    for name, y in series.items():
        if drop_bank is not None:
            y = y.clone()
            y[drop_bank] = float("nan")
        ref_cells = sum(int(y[:, i].isfinite().sum()) for i in pre_cols if lead[0, i] == 0)
        if ref_cells == 0:
            results[name] = _nan_result(f"reference bin {REFERENCE_BIN} has no finite cells")
            continue
        f = fit_twfe(y, regs, torch.ones(inp.n, t), bootstrap=boot, n_boot=n_boot, seed=seed)
        results[name] = TotalEffectResult(
            beta_dose=f.beta["dose_post"], se_dose=f.se["dose_post"], p_dose=f.p_wild.get("dose_post"),
            beta_lead=f.beta.get("dose_lead", math.nan), se_lead=f.se.get("dose_lead", math.nan),
            p_lead=f.p_wild.get("dose_lead"),
            n_obs=f.n_obs, n_clusters=f.n_clusters, pre_mean=y[:, pre_cols].nanmean().item(),
            effect_1sd=f.beta["dose_post"] * sd["dose"], dropped_regressors=list(f.dropped),
            extra={k: {"beta": f.beta[f"{k}_post"], "se": f.se[f"{k}_post"], "p": f.p_wild.get(f"{k}_post"),
                       "effect_1sd": f.beta[f"{k}_post"] * sd[k]} for k in extra_doses or {}},
            trend={k: {"beta": f.beta[f"{k}_x_trend"], "se": f.se[f"{k}_x_trend"]} for k in trend_controls or {}},
            post_centre=post_centre)
    return TotalEffectPanel(start=periods[0], end=periods[-1], n_periods=t,
                            missing_months=missing_months(periods)), results
