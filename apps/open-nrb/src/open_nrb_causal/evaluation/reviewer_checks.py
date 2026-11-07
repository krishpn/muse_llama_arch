#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
#     "scipy",
# ]
# ///
"""Run the cheap Sec 5b reviewer checks (paper2_causal_spillover.md), one section each.

All built on the same sample as run_causal_experiment (shared build_inputs). Item 1 (MDE) is
power_mde.py.

  survivorship    item 2   included vs excluded commercial banks: pre-window dose and size,
                           months with data, first/last month (exits / late entrants)
  weights         item 8   HT weight distribution, Kish effective sample size, trimmed and
                           unweighted DiD
  gps_period      item 8   remedy: GPS with period intercepts + per-period sigma, weights + DiD
  negative_flows  item 11a share of negative de-cumulated monthly flows, by FY and calendar month
  quarterly       item 11a remedy: outcomes from quarter-end YTD figures only, DiD under main,
                           unweighted and period-effects weights
  same_quarter    item 11a Ashwin (Q1) quarterly cells only, rules out a dose-related seasonal pattern
  pre_trend       item 3   dose-only quarterly event study extended back to Baisakh 2070 (clean
                           bulletins, before W_hat exists): leads, pooled lead vs post, 1-SD effect
  frozen_size     item 11c outcomes divided by the frozen pre-window size instead of current size
  leave_one_out   item 6   pooled DiD dropping one bank at a time (design and weights held fixed)
  marginal_balance Sec 8   per month sum r vs sum c and single-bank infeasibility in the 16-bank
                           closed system, against both networks' marginal deviations
  row_normalised  item 12  exposure from the row-normalised network (W_ij / sum_j W_ij)
  centrality_exposure item 12b exploratory: Katz-Bonacich exposure W~ (I - 0.5 W~)^-1 dose (direct
                           plus indirect counterparties), W~ row-normalised; Holm within the check
  written_off_zero item 11b loans written off: blank as missing vs blank as 0 where the bank
                           reported other P&L lines (main per blank_as_zero), Eq. A' and the
                           headline total effect side by side
  paid_up_dose    item 1   alternative dose from paid-up capital; pre-window distribution always,
                           estimates once paid_up_capital_requirement is set in the config, then
                           total-effect robustness: size vs paid-up term, frozen size, leave-one-out
  paid_up_path    item 1   per bank: paid-up at each Asar, first month at the requirement vs the
                           Asar 2074 deadline, share of the increase already in the pre window
  paid_up_threats item 1   threats to the paid-up result: stock event study + trend-adjusted total
                           effect; SBL / RBB mergers (data dating, drop, SBL pro-forma); floor vs
                           catch-up (bunching, raise vs shortfall, raise timing)
  paid_up_trend   item 1   dose trend fitted on the pre cells only and extrapolated: event study by
                           bin with and without it, pooled total effect (none / whole-panel /
                           pre-only trend), smoothness bound on a post slope break (breakdown M)
  label_continuity item 2  bulletin label splits (overlap, gap, size carry-over) and exits with
                           candidate absorbers; data continuity only, not merger verification
  multiple_tests  item 5   test count, expected false positives, Holm / BH over the saved run JSON,
                           joint wild-bootstrap Wald tests over the event-study bins

--quarterly swaps the monthly outcomes for the quarterly ones in every check (output file gets a
_quarterly suffix); multiple_tests still reads the saved monthly run JSON.

Read-only: re-estimates, never edits data. Wild-bootstrap p-values use --n-boot draws (default:
the config's n_bootstrap, so a check that reproduces a main-run fit gives the same p-value);
leave_one_out is CR1 only. Writes results/reviewer_checks_<config>.md.

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.evaluation.reviewer_checks
  uv run python -m open_nrb_causal.evaluation.reviewer_checks --config src/open_nrb_causal/experiments/configs/baseline.yaml --only weights frozen_size
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
import json
import math
import re
import sys
from pathlib import Path

import torch
from scipy.stats import mannwhitneyu, spearmanr
from scipy.stats import t as t_dist

try:
    from open_nrb.evaluation.report_io import captured_output
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from evaluation.report_io import captured_output  # type: ignore[no-redef]

try:
    from ..models.estimator import fit_did, fit_event_study, fit_twfe
    from ..models.experiment_config import CONFIG_DIR, load_causal_experiment_config
    from ..models.mandate_design import (
        BS_MONTHS, QUARTER_END_MONTHS, apply_weighting, bank_size, bin_indicator, did_design, event_bins,
        fiscal_year, freeze_pre_mandate, mandate_timing, outcome_ratio, period_indicator, period_key,
        positive_size, pre_x_post_controls, quarter_end_only)
    from ..models.panel_alignment import Panel, filter_commercial_banks, load_commercial_panel, load_panel
    from ..models.run_causal_experiment import RESULTS_DIR, build_inputs
    from ..models.total_effect import (
        TOTAL_EFFECT_START, extended_panel, fit_total_effect, missing_months, total_effect_series,
        years_since_pre_end)
    from ..models.variables import (
        OUTCOME_FIELDS, compute_dose, load_outcome_panel, load_paid_up_capital, pnl_reported, zero_fill_blanks)
    from .report_format import fmt_num as _num
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from estimator import fit_did, fit_event_study, fit_twfe  # type: ignore[no-redef]
    from experiment_config import CONFIG_DIR, load_causal_experiment_config  # type: ignore[no-redef]
    from mandate_design import (  # type: ignore[no-redef]
        BS_MONTHS, QUARTER_END_MONTHS, apply_weighting, bank_size, bin_indicator, did_design, event_bins,
        fiscal_year, freeze_pre_mandate, mandate_timing, outcome_ratio, period_indicator, period_key,
        positive_size, pre_x_post_controls, quarter_end_only)
    from panel_alignment import (  # type: ignore[no-redef]
        Panel, filter_commercial_banks, load_commercial_panel, load_panel)
    from run_causal_experiment import RESULTS_DIR, build_inputs  # type: ignore[no-redef]
    from total_effect import (  # type: ignore[no-redef]
        TOTAL_EFFECT_START, extended_panel, fit_total_effect, missing_months, total_effect_series,
        years_since_pre_end)
    from variables import (  # type: ignore[no-redef]
        OUTCOME_FIELDS, compute_dose, load_outcome_panel, load_paid_up_capital, pnl_reported, zero_fill_blanks)
    from report_format import fmt_num as _num  # type: ignore[no-redef]

DEFAULT_CONFIGS = [CONFIG_DIR / "main.yaml"]
CHECKS = ["survivorship", "weights", "gps_period", "negative_flows", "quarterly", "same_quarter",
          "pre_trend", "frozen_size", "leave_one_out", "marginal_balance", "row_normalised", "centrality_exposure", "written_off_zero",
          "paid_up_dose",
          "paid_up_path", "paid_up_threats", "paid_up_trend", "label_continuity", "multiple_tests"]


def _p(x) -> str:
    return _num(x, ".3f")


def _fit(y, d, weights=None, n_boot=0, seed=0):
    return fit_did(y, d.dose_x_post, d.exposure_x_post, d.controls,
                   d.weights if weights is None else weights, n_boot=n_boot, seed=seed)


_REF: dict[tuple[str, str], object] = {}


def _ref(name: str, net: str, inp, cfg, args):
    """Main-specification fit at --n-boot, computed once per (outcome, network) and config."""
    if (name, net) not in _REF:
        d = inp.design_main if net == "W_hat" else inp.design_ras
        _REF[(name, net)] = _fit(inp.y[name], d, n_boot=args.n_boot, seed=cfg.bootstrap_seed)
    return _REF[(name, net)]


def _did_table(out, rows: list[tuple[str, str, object, object]]) -> None:
    """rows: (outcome, network, reference DiDResult, alternative DiDResult)."""
    out("| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |")
    out("|---|---|---|---|---|---|---|---|---|")
    for name, net, ref, alt in rows:
        for coef in ["dose", "exposure"]:
            out(f"| {name} | {net} | {coef} | {_num(getattr(ref, f'beta_{coef}'), '.4g')} | "
                f"{_p(getattr(ref, f'p_{coef}'))} | {_num(getattr(alt, f'beta_{coef}'), '.4g')} | "
                f"{_num(getattr(alt, f'se_{coef}'), '.3g')} | {_p(getattr(alt, f'p_{coef}'))} | {alt.n_obs} |")


# --- item 2 ------------------------------------------------------------------
def check_survivorship(out, cfg, inp, args) -> None:
    """Compare included vs excluded commercial banks (item 2)."""
    out("## Survivorship (item 2): included vs excluded commercial banks")
    full = load_commercial_panel(cfg.csv_path)
    try:
        timing = mandate_timing(full.periods, cfg.pre_window_end, cfg.effective_period)
    except ValueError as e:
        out(f"not computed: {e}")
        return
    pre = slice(0, timing.pre_end_idx + 1)
    size = positive_size(bank_size(full))
    dose = compute_dose(full, cfg.capital_floor)
    included = set(inp.bfi_names)
    complete_cells = torch.isfinite(full.X).all(dim=-1)  # (N, T): every covariate finite
    out(f"{len(full.bfi_names)} commercial banks from {full.periods[0]}; {len(included)} in the "
        f"estimation panel (Paper 1 complete-case filter). Pre window = {full.periods[0]}..{full.periods[timing.pre_end_idx]}.")
    out("")
    out("| bank | in panel | months with complete data | first | last | dose_pre | pre size (NPR mn) |")
    out("|---|---|---|---|---|---|---|")
    groups: dict[bool, dict[str, list[float]]] = {True: {"dose": [], "size": []}, False: {"dose": [], "size": []}}
    for i, name in enumerate(full.bfi_names):
        has = torch.nonzero(complete_cells[i]).flatten().tolist()
        first = full.periods[has[0]] if has else None
        last = full.periods[has[-1]] if has else None
        d_pre = dose[i, pre].nanmean().item()
        s_pre = size[i, pre].nanmean().item()
        inc = name in included
        if math.isfinite(d_pre):
            groups[inc]["dose"].append(d_pre)
        if math.isfinite(s_pre):
            groups[inc]["size"].append(s_pre)
        out(f"| {name} | {'yes' if inc else 'no'} | {len(has)}/{len(full.periods)} | "
            f"{'-' if first is None else f'{first[1]} {first[0]}'} | {'-' if last is None else f'{last[1]} {last[0]}'} | "
            f"{_num(d_pre, '.4f')} | {_num(s_pre, ',.0f')} |")
    out("")
    for key, label in [("dose", "dose_pre"), ("size", "pre-window size")]:
        a, b = groups[True][key], groups[False][key]
        if len(a) >= 1 and len(b) >= 1:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            med = lambda v: float(torch.tensor(v).median())  # noqa: E731
            out(f"{label}: included median {med(a):.4g} (n={len(a)}) vs excluded median {med(b):.4g} "
                f"(n={len(b)}), Mann-Whitney p={p:.3f}")
        else:
            out(f"{label}: not comparable (included n={len(a)}, excluded n={len(b)} with a finite pre-window value)")
    out("Excluded banks whose last complete month falls inside the post window are candidate "
        "merger exits; check names/dates against NRB merger records before writing it up.")


# --- item 8 ------------------------------------------------------------------
def _weight_summary(out, inp, designs: list[tuple[str, object]]) -> None:
    out("| network | cells w>0 | mean | p99 / mean | max / mean | Kish ESS | ESS / cells |")
    out("|---|---|---|---|---|---|---|")
    for net, d in designs:
        w = d.weights[(d.weights > 0) & d.weights.isfinite()].to(torch.float64)
        if w.numel() == 0:
            out(f"| {net} | 0 | n/a | n/a | n/a | n/a | n/a |")
            continue
        m = w.mean().item()
        ess = (w.sum() ** 2 / (w ** 2).sum()).item()
        out(f"| {net} | {w.numel()} | {m:.3g} | {torch.quantile(w, 0.99).item() / m:.2f} | "
            f"{w.max().item() / m:.2f} | {ess:.0f} | {ess / w.numel():.2f} |")
    out("")
    out("Where the weight sits: share of total HT weight per bank (top 5), and per period block.")
    for net, d in designs:
        w = torch.where((d.weights > 0) & d.weights.isfinite(), d.weights, 0.0).to(torch.float64)
        total = w.sum().item()
        if total <= 0:
            continue
        share = w.sum(dim=1) / total
        top = sorted(range(inp.n), key=lambda i: -share[i].item())[:5]
        pre_w = w[:, : inp.timing.pre_end_idx + 1].sum().item() / total
        out(f"- {net}: " + ", ".join(f"{inp.bfi_names[i]} {share[i].item():.2f}" for i in top)
            + f"; pre-window cells {pre_w:.2f}, post cells {1 - pre_w:.2f}")


def check_weights(out, cfg, inp, args) -> None:
    """Summarise the HT weight distribution and reweighted DiD (item 8)."""
    out("## HT/GPS weights (item 8)")
    ht = [("W_hat", inp.design_main_ht), ("RAS", inp.design_ras_ht)]
    _weight_summary(out, inp, ht)
    out("")
    out("Trimmed HT weights (cap at the q-th percentile of positive weights, a standard IPW fix) vs main:")
    for q in (0.95, 0.90):
        out("")
        out(f"q = {q}:")
        rows = []
        for name, y in inp.y.items():
            for net, d in ht:
                pos = d.weights[(d.weights > 0) & d.weights.isfinite()]
                cap = torch.quantile(pos.to(torch.float64), q).item() if pos.numel() else math.inf
                rows.append((name, net, _ref(name, net, inp, cfg, args),
                             _fit(y, d, weights=d.weights.clamp(max=cap), n_boot=args.n_boot,
                                  seed=cfg.bootstrap_seed)))
        _did_table(out, rows)
    other = "ht" if cfg.weighting == "unweighted" else "unweighted"
    out("")
    out(f"The other weighting ({other}) vs main ({cfg.weighting}), same cells:")
    rows = []
    for name, y in inp.y.items():
        for net, d in ht:
            rows.append((name, net, _ref(name, net, inp, cfg, args),
                         _fit(y, apply_weighting(d, other), n_boot=args.n_boot, seed=cfg.bootstrap_seed)))
    _did_table(out, rows)


# --- item 8, remedy ----------------------------------------------------------
_PERIOD_DESIGNS: dict[str, object] = {}


def _period_design(net: str, inp):
    """did_design with the period-effects GPS (alpha_t + sigma_t), cached per config."""
    if net not in _PERIOD_DESIGNS:
        W = inp.W_main if net == "W_hat" else inp.W_ras
        _PERIOD_DESIGNS[net] = did_design(W, inp.dose_pre, inp.controls_pre, inp.timing, gps_period_effects=True)
    return _PERIOD_DESIGNS[net]


def check_gps_period(out, cfg, inp, args) -> None:
    """Refit the GPS weights with period intercepts and per-period sigma (item 8 remedy)."""
    out("## GPS with period intercepts + per-period sigma (item 8 remedy)")
    out("The main GPS conditions only on frozen pre-mandate covariates, so its fitted mean is "
        "constant over time while exposure trends with W's nominal level; whole periods land in "
        "the density tails. Same regressors here, only the HT weights change.")
    _weight_summary(out, inp, [(net, _period_design(net, inp)) for net in ("W_hat", "RAS")])
    out("")
    rows = []
    for name, y in inp.y.items():
        for net in ("W_hat", "RAS"):
            rows.append((name, net, _ref(name, net, inp, cfg, args),
                         _fit(y, _period_design(net, inp), n_boot=args.n_boot, seed=cfg.bootstrap_seed)))
    _did_table(out, rows)


# --- item 11a, remedy --------------------------------------------------------
def _quarterly_y(inp) -> dict[str, torch.Tensor]:
    return {name: outcome_ratio(y, inp.periods, inp.size, True, True) for name, y in inp.y_raw.items()}


def check_quarterly(out, cfg, inp, args) -> None:
    """Re-estimate on quarterly outcomes from quarter-end YTD figures only (item 11a remedy)."""
    out("## Quarterly outcomes from quarter-end YTD only (item 11a remedy)")
    if not cfg.pnl_ytd_cumulative:
        out("skipped: pnl_ytd_cumulative is false, quarterly differencing assumes YTD figures")
        return
    out(f"Flow at {', '.join(QUARTER_END_MONTHS)} = YTD minus the previous quarter end of the "
        "same fiscal year (Ashwin = YTD itself), divided by quarter-end size. Non-quarter-end "
        "months are never read. Regressors unchanged; non-quarter cells drop out as NaN outcomes.")
    yq = _quarterly_y(inp)
    pre = slice(0, inp.timing.pre_end_idx + 1)
    out("")
    out("| outcome | finite quarter cells | in pre window | negative share |")
    out("|---|---|---|---|")
    for name, y in yq.items():
        ok = y.isfinite()
        n_ok = int(ok.sum())
        out(f"| {name} | {n_ok} | {int(ok[:, pre].sum())} | "
            f"{_num(int((ok & (y < 0)).sum()) / n_ok if n_ok else math.nan, '.3f')} |")
    for label, weights_for in [
        ("pooled-GPS HT weights", lambda d, net: d.weights),
        ("unweighted", lambda d, net: (d.weights > 0).to(d.weights.dtype)),
        ("period-effects GPS weights", lambda d, net: _period_design(net, inp).weights),
    ]:
        out("")
        out(f"Quarterly DiD, {label} (main = {cfg.outcome_construction if not args.quarterly else 'quarterly'} "
            f"outcome, {cfg.weighting}):")
        rows = []
        for name, y in yq.items():
            for net, d in [("W_hat", inp.design_main_ht), ("RAS", inp.design_ras_ht)]:
                rows.append((name, net, _ref(name, net, inp, cfg, args),
                             _fit(y, d, weights=weights_for(d, net), n_boot=args.n_boot, seed=cfg.bootstrap_seed)))
        _did_table(out, rows)


def check_same_quarter(out, cfg, inp, args) -> None:
    """Re-estimate on Ashwin (Q1) quarterly cells only (item 11a remedy, seasonality)."""
    out("## Same-quarter DiD: Ashwin (Q1) quarterly cells only (item 11a remedy, seasonality)")
    if not cfg.pnl_ytd_cumulative:
        out("skipped: pnl_ytd_cumulative is false, quarterly differencing assumes YTD figures")
        return
    out("The quarterly pre window holds only Ashwin 2072 (Q1), while post cells span all four "
        "quarters, so a dose-related seasonal pattern would load onto dose x post. Keeping Q1 "
        "only compares like with like (one pre and a few post cells per bank).")
    cols = [i for i, p in enumerate(inp.periods) if p[1] == QUARTER_END_MONTHS[0]]
    y1 = {}
    for name, y in _quarterly_y(inp).items():
        keep = torch.full_like(y, float("nan"))
        keep[:, cols] = y[:, cols]
        y1[name] = keep
    out(f"Q1 columns: {', '.join(f'{inp.periods[i][1]} {inp.periods[i][0]}' for i in cols)}")
    for label, weights_for in [
        ("unweighted", lambda d, net: (d.weights > 0).to(d.weights.dtype)),
        ("period-effects GPS weights", lambda d, net: _period_design(net, inp).weights),
    ]:
        out("")
        out(f"Q1-only DiD, {label}:")
        rows = []
        for name, y in y1.items():
            for net, d in [("W_hat", inp.design_main_ht), ("RAS", inp.design_ras_ht)]:
                rows.append((name, net, _ref(name, net, inp, cfg, args),
                             _fit(y, d, weights=weights_for(d, net), n_boot=args.n_boot, seed=cfg.bootstrap_seed)))
        _did_table(out, rows)


# --- item 3 ------------------------------------------------------------------
def check_pre_trend(out, cfg, inp, args) -> None:
    """Run the dose-only quarterly pre-trend event study back to 2070 (item 3)."""
    out("## Dose-only quarterly pre-trend from the clean 2070-2072 bulletins (item 3)")
    if not cfg.pnl_ytd_cumulative:
        out("skipped: pnl_ytd_cumulative is false, quarterly differencing assumes YTD figures")
        return
    out(f"W_hat starts at Asar 2072, so exposure can't be built earlier; this is the dose-only "
        f"event study on quarterly outcomes from {TOTAL_EFFECT_START[1]} {TOTAL_EFFECT_START[0]}. Same 16 "
        "banks, same frozen dose_pre (Asar-Mangsir 2072 mean) and Z_pre x post controls as the main "
        "run; bank + period FE, unweighted. Reference bin -1 = the pre-window fiscal year.")
    try:
        panel = extended_panel(cfg.csv_path, inp.bfi_names, inp.periods[-1])
        outcomes = load_outcome_panel(cfg.pnl_csv_path, panel)
        if cfg.blank_as_zero:
            outcomes = zero_fill_blanks(outcomes, cfg.blank_as_zero)
        timing = mandate_timing(panel.periods, cfg.pre_window_end, cfg.effective_period)
    except ValueError as e:
        out(f"not computed: {e}")
        return
    periods, t = panel.periods, len(panel.periods)
    gaps = missing_months(periods)
    out(f"Extended panel {periods[0]} .. {periods[-1]}, T={t}; months absent from the extraction: "
        + (", ".join(f"{m} {y}" for y, m in gaps) if gaps else "none")
        + ". A quarter whose own or previous quarter-end month is absent is NaN.")
    size = positive_size(bank_size(panel))
    bins = event_bins(periods, timing)
    ref = -1
    post = timing.post(t).unsqueeze(0)
    controls = pre_x_post_controls(inp.controls_pre, post)
    # dose_pre comes from inp: freezing on this panel would average from TOTAL_EFFECT_START instead.
    dose_b = inp.dose_pre.unsqueeze(1).expand(inp.n, t)
    ones = torch.ones(inp.n, t)
    lead = period_indicator(i <= timing.pre_end_idx and bins[i] < ref for i in range(t))
    in_ref = [i <= timing.pre_end_idx and bins[i] == ref for i in range(t)]
    yq = {name: outcome_ratio(getattr(outcomes, name), periods, size, True, True) for name in OUTCOME_FIELDS}
    start = periods.index(inp.periods[0])
    exposure_terms: dict[str, torch.Tensor] = {}
    if periods[start:] == list(inp.periods):
        for nm, v in [("exposure_level", inp.design_main.exposure),
                      ("exposure_x_post", inp.design_main.exposure_x_post)]:
            pad = torch.full((inp.n, t), float("nan"))
            pad[:, start:] = v
            exposure_terms[nm] = pad
    else:
        out("Extended periods don't end with the main panel's periods; '+ exposure' rows skipped.")

    out("")
    out("Finite pre-mandate quarterly cells per bank (bins <= -1):")
    out("| bank | " + " | ".join(yq) + " |")
    out("|---|" + "---|" * len(yq))
    pre_cols = list(range(timing.pre_end_idx + 1))
    for i, bank in enumerate(inp.bfi_names):
        out(f"| {bank} | " + " | ".join(str(int(y[i, pre_cols].isfinite().sum())) for y in yq.values()) + " |")

    pooled_rows = []
    for name, y in yq.items():
        ok = y.isfinite()
        cells = {b: sum(int(ok[:, i].sum()) for i, bt in enumerate(bins) if bt == b) for b in dict.fromkeys(bins)}
        out("")
        out(f"### {name}")
        if cells.get(ref, 0) == 0:
            out(f"reference bin {ref} has no finite cells, not estimated")
            continue
        es = fit_event_study(y, inp.dose_pre, None, bins, controls, ones, reference_bin=ref,
                             n_boot=args.n_boot, seed=cfg.bootstrap_seed)
        out(f"n_obs={es.n_obs}, clusters={es.n_clusters}"
            + (f", unidentified: {es.dropped_regressors}" if es.dropped_regressors else ""))
        out("| bin | quarters | cells | dose beta | SE | wild p |")
        out("|---|---|---|---|---|---|")
        for b in [b for b in cells if cells[b] > 0]:
            qs = [f"{p[1][:3]} {p[0]}" for i, p in enumerate(periods) if bins[i] == b and ok[:, i].any()]
            if b == ref:
                out(f"| {b} (ref) | {', '.join(qs)} | {cells[b]} | 0 | - | - |")
                continue
            e = es.dose[str(b)]
            out(f"| {b} | {', '.join(qs)} | {cells[b]} | {_num(e['beta'], '.4g')} | "
                f"{_num(e['se'], '.3g')} | {_p(e['p'])} |")

        # Pooled: one lead (all pre bins before the reference) against post. The Q1-only rows keep
        # Ashwin cells alone, so a dose-related seasonal pattern can't load onto either coefficient.
        # The "from Asar 2072" rows separate the two differences from the main quarterly DiD: the
        # extra 2070-71 cells, and the missing exposure terms (NaN before Asar 2072, so fit_twfe
        # drops those cells; the "+ exposure" rows should reproduce the main unweighted estimate).
        y_q1 = torch.where(period_indicator(p[1] == QUARTER_END_MONTHS[0] for p in periods).bool().expand_as(y), y,
                           torch.tensor(float("nan")))
        in_win = period_indicator(i >= start and not timing.pre_end_idx < i <= timing.effective_idx
                                  for i in range(t)).bool().expand_as(y)  # main DiD: transition cells weight 0
        cut = lambda v: torch.where(in_win, v, torch.tensor(float("nan")))  # noqa: E731
        samples = [("all quarters", y, {}), ("Ashwin (Q1) only", y_q1, {}),
                   ("from Asar 2072, all quarters", cut(y), {}), ("from Asar 2072, Q1 only", cut(y_q1), {})]
        if exposure_terms:
            samples += [("from Asar 2072, all quarters, + exposure", cut(y), exposure_terms),
                        ("from Asar 2072, Q1 only, + exposure", cut(y_q1), exposure_terms)]
        for label, y_use, extra in samples:
            regs = {"dose_lead": dose_b * lead, "dose_transition": dose_b * timing.transition(t).unsqueeze(0),
                    "dose_post": dose_b * post} | controls | extra
            f = fit_twfe(y_use, regs, ones, bootstrap=["dose_lead", "dose_post"],
                         n_boot=args.n_boot, seed=cfg.bootstrap_seed)
            ref_mean = y_use[:, [i for i in range(t) if in_ref[i] or lead[0, i] > 0]].nanmean().item()
            effect = f.beta["dose_post"] * inp.dose_pre.std().item()
            pooled_rows.append((name, label, f, ref_mean, effect))

    out("")
    out("### Pooled: lead (bins < -1) and post, reference bin -1")
    out("1-SD effect = post beta x SD(dose_pre); pre mean = mean quarterly outcome over pre cells.")
    out("| outcome | sample | lead beta | lead SE | lead p | post beta | post SE | post p | 1-SD effect | pre mean | effect / pre mean | n_obs |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, label, f, ref_mean, effect in pooled_rows:
        rel = effect / ref_mean if math.isfinite(ref_mean) and ref_mean != 0 else math.nan
        out(f"| {name} | {label} | {_num(f.beta['dose_lead'], '.4g')} | {_num(f.se['dose_lead'], '.3g')} | "
            f"{_p(f.p_wild.get('dose_lead'))} | {_num(f.beta['dose_post'], '.4g')} | {_num(f.se['dose_post'], '.3g')} | "
            f"{_p(f.p_wild.get('dose_post'))} | {_num(effect, '.3g')} | {_num(ref_mean, '.3g')} | {_num(rel, '.2f')} | {f.n_obs} |")
    out(f"SD(dose_pre) = {inp.dose_pre.std().item():.4f}. A lead near 0 next to a post effect of "
        "similar size to the main run supports parallel pre-trends; a lead of the post's size and "
        "sign points to a pre-existing trend or mean reversion.")


# --- item 11a ----------------------------------------------------------------
def check_negative_flows(out, cfg, inp, args) -> None:
    """Report the share of negative de-cumulated monthly flows (item 11a)."""
    out("## Negative de-cumulated monthly flows (item 11a)")
    out(f"pnl_ytd_cumulative={cfg.pnl_ytd_cumulative}. NIM and loans written off should be "
        "non-negative; provisions can be legitimately negative (write-backs), net profit can be either.")
    out("| outcome | finite cells | negative | share | worst FY (share) |")
    out("|---|---|---|---|---|")
    fys = [fiscal_year(p) for p in inp.periods]
    for name, y in inp.y.items():
        ok = y.isfinite()
        neg = ok & (y < 0)
        by_fy = {}
        for fy in sorted(set(fys)):
            cols = [i for i, f in enumerate(fys) if f == fy]
            n_ok = int(ok[:, cols].sum())
            if n_ok:
                by_fy[fy] = int(neg[:, cols].sum()) / n_ok
        worst = max(by_fy, key=by_fy.get) if by_fy else None
        n_ok = int(ok.sum())
        out(f"| {name} | {n_ok} | {int(neg.sum())} | {_num(int(neg.sum()) / n_ok if n_ok else math.nan, '.3f')} | "
            f"{'n/a' if worst is None else f'FY{worst} ({by_fy[worst]:.2f})'} |")
    # A YTD restatement (a bulletin revising earlier months down) shows up as negative flows
    # concentrated in particular calendar months; a uniform spread points elsewhere.
    present = {p[1] for p in inp.periods}
    k = BS_MONTHS.index("Shrawan")
    months = [m for m in BS_MONTHS[k:] + BS_MONTHS[:k] if m in present]
    out("")
    out("Negative share by calendar month (Shrawan values are single months, never differenced):")
    out("| outcome | " + " | ".join(months) + " |")
    out("|---|" + "---|" * len(months))
    for name, y in inp.y.items():
        cells = []
        for mth in months:
            cols = [i for i, p in enumerate(inp.periods) if p[1] == mth]
            ok = y[:, cols].isfinite()
            n_ok = int(ok.sum())
            cells.append(_num(int((ok & (y[:, cols] < 0)).sum()) / n_ok if n_ok else math.nan, ".2f"))
        out(f"| {name} | " + " | ".join(cells) + " |")


# --- item 11c ----------------------------------------------------------------
def check_frozen_size(out, cfg, inp, args) -> None:
    """Re-estimate with the frozen pre-window size as the denominator (item 11c)."""
    out("## Frozen pre-window size as denominator (item 11c)")
    pre = inp.size[:, : inp.timing.pre_end_idx + 1].nanmean(dim=1, keepdim=True)
    rows = []
    for name, y in inp.y.items():
        y_frozen = y * inp.size / pre  # undo / size, apply / size_pre; NaN wherever size was bad
        for net, d in [("W_hat", inp.design_main), ("RAS", inp.design_ras)]:
            rows.append((name, net, _ref(name, net, inp, cfg, args),
                         _fit(y_frozen, d, n_boot=args.n_boot, seed=cfg.bootstrap_seed)))
    _did_table(out, rows)


# --- item 6 ------------------------------------------------------------------
def check_leave_one_out(out, cfg, inp, args) -> None:
    """Re-estimate the pooled DiD dropping one bank at a time (item 6)."""
    out("## Leave-one-bank-out (item 6), CR1 only; design and weights held fixed")
    out("| outcome | network | coef | full beta | LOO min | LOO max | sign flips | max abs t | most influential bank |")
    out("|---|---|---|---|---|---|---|---|---|")
    for name, y in inp.y.items():
        for net, d in [("W_hat", inp.design_main), ("RAS", inp.design_ras)]:
            full = _fit(y, d)
            loo = []
            for i, bank in enumerate(inp.bfi_names):
                y_i = y.clone()
                y_i[i] = float("nan")
                loo.append((bank, _fit(y_i, d)))
            for coef in ["dose", "exposure"]:
                b0 = getattr(full, f"beta_{coef}")
                vals = [(bank, getattr(r, f"beta_{coef}"), getattr(r, f"se_{coef}")) for bank, r in loo]
                vals = [v for v in vals if math.isfinite(v[1])]
                if not vals or not math.isfinite(b0):
                    out(f"| {name} | {net} | {coef} | {_num(b0, '.4g')} | n/a | n/a | n/a | n/a | n/a |")
                    continue
                flips = sum(1 for _, b, _ in vals if b * b0 < 0)
                ts = [abs(b / s) for _, b, s in vals if math.isfinite(s) and s > 0]
                infl = max(vals, key=lambda v: abs(v[1] - b0))
                out(f"| {name} | {net} | {coef} | {b0:.4g} | {min(v[1] for v in vals):.4g} | "
                    f"{max(v[1] for v in vals):.4g} | {flips}/{len(vals)} | {_num(max(ts) if ts else math.nan, '.2f')} | "
                    f"{infl[0]} ({infl[1]:.4g}) |")


# --- item 12 -----------------------------------------------------------------
def _row_normalise(W: torch.Tensor, zero_rows: torch.Tensor | None = None) -> torch.Tensor:
    """zero_rows (T, N): rows forced to 0 before normalising. Without it, a reconstructed row that is
    ~0 (a bank reporting no interbank lending) is scaled up to sum 1 from numerical noise."""
    if zero_rows is not None:
        W = W * (~zero_rows).unsqueeze(-1)
    s = W.sum(dim=-1, keepdim=True)
    return torch.where(s > 0, W / torch.where(s > 0, s, torch.ones_like(s)), torch.zeros_like(W))


def _no_lending_rows(inp) -> torch.Tensor:
    """(T, N) True where the published row marginal r (stored in the Paper 1 artefact) is <= 0."""
    if inp.artifact.r is None:
        raise SystemExit("the artefact has no stored r marginal: cannot tell which banks report no lending")
    return (inp.artifact.r[:, : inp.t] <= 0).T


def _zero_row_diagnostic(out, inp) -> None:
    """Row and column sums of both networks against the published marginals r, c, so the
    zero-row rule and the orientation of each matrix are checked, not assumed."""
    art = inp.artifact
    if art.r is None or art.c is None:
        out("- marginal diagnostic skipped: the artefact has no stored r / c")
        return
    r, c = art.r[:, : inp.t].T, art.c[:, : inp.t].T  # (T, N)
    out(f"- published marginals: r <= 0 on {int((r <= 0).sum())}, c <= 0 on {int((c <= 0).sum())} "
        f"of {r.numel()} bank-months; median r {_num(r.median().item(), '.4g')}, median c {_num(c.median().item(), '.4g')}")
    out("| matrix | sum over | zero sums | max abs dev vs r | max abs dev vs c |")
    out("|---|---|---|---|---|")
    for net, W in [("W_hat", inp.W_main), ("RAS", inp.W_ras)]:
        for label, s in [("rows (j)", W.sum(dim=-1)), ("cols (i)", W.sum(dim=-2))]:
            out(f"| {net} | {label} | {int((s <= 0).sum())} | {_num((s - r).abs().max().item(), '.4g')} | "
                f"{_num((s - c).abs().max().item(), '.4g')} |")
    z_ras = inp.W_ras.sum(dim=-1) <= 0
    if z_ras.any():
        out(f"- on the {int(z_ras.sum())} bank-months where the RAS row is 0: median r "
            f"{_num(r[z_ras].median().item(), '.4g')}, median c {_num(c[z_ras].median().item(), '.4g')}, "
            f"median W_hat row sum {_num(inp.W_main.sum(dim=-1)[z_ras].median().item(), '.4g')}")
    out(f"- rows set to 0 before row-normalising: those with r <= 0 ({int((r <= 0).sum())})")


def check_marginal_balance(out, cfg, inp, args) -> None:
    """Test whether any matrix can match both published marginals (Sec 8 follow-up).

    In the 16-bank closed system a zero-diagonal matrix can match r (lending) and
    c (borrowing) only if sum_i r_i = sum_j c_j in that month and no bank lends
    more than all others borrow (r_i <= sum_{j != i} c_j) or borrows more than all
    others lend (c_i <= sum_{j != i} r_j). Flags months failing either condition
    and relates both networks' marginal deviations to the gap.
    """
    out("## Marginal balance (Sec 8 follow-up): can any matrix match both published marginals?")
    out("Within the 16-bank closed system a matrix with zero diagonal can match r (lending) and c "
        "(borrowing) only if sum_i r_i = sum_j c_j in that month and no bank lends more than all "
        "others borrow (r_i <= sum_{j != i} c_j) or borrows more than all others lend. Where either "
        "fails, no reconstruction can be exact and a projection returns a compromise. The per-entry "
        "cap is a further constraint, not checked here.")
    art = inp.artifact
    if art.r is None or art.c is None:
        out("skipped: the artefact has no stored r / c")
        return
    r, c = art.r[:, : inp.t].T.to(torch.float64), art.c[:, : inp.t].T.to(torch.float64)  # (T, N)
    sr, sc = r.sum(dim=1), c.sum(dim=1)
    gap = sr - sc
    rel = gap.abs() / torch.maximum(sr, sc).clamp(min=1e-12)
    over_r = r > (sc.unsqueeze(1) - c)  # lends more than the other banks borrow
    over_c = c > (sr.unsqueeze(1) - r)
    infeasible = (rel > 0.01) | over_r.any(dim=1) | over_c.any(dim=1)
    out(f"- months: {inp.t}; |sum r - sum c| / max > 1%: {int((rel > 0.01).sum())}, > 10%: {int((rel > 0.10).sum())}; "
        f"median relative gap {_num(rel.median().item(), '.3f')}, max {_num(rel.max().item(), '.3f')}")
    out(f"- sum r > sum c in {int((gap > 0).sum())} months, < in {int((gap < 0).sum())}; "
        f"max |gap| {_num(gap.abs().max().item(), ',.0f')} (marginal units)")
    out(f"- bank-months with r_i > sum_(j!=i) c_j: {int(over_r.sum())}; with c_i > sum_(j!=i) r_j: {int(over_c.sum())}")
    out(f"- months failing at least one condition: {int(infeasible.sum())} of {inp.t}")

    dev = {}
    for net, W in [("W_hat", inp.W_main), ("RAS", inp.W_ras)]:
        dev[net] = ((W.sum(dim=-1) - r).abs().max(dim=1).values, (W.sum(dim=-2) - c).abs().max(dim=1).values)
    for net in dev:
        d = torch.maximum(*dev[net])
        bad, ok = d[infeasible], d[~infeasible]
        out(f"- {net}: max |marginal dev| per month, median in failing months "
            f"{_num(bad.median().item() if bad.numel() else math.nan, ',.4g')} (n={bad.numel()}), in the others "
            f"{_num(ok.median().item() if ok.numel() else math.nan, ',.4g')} (n={ok.numel()}); "
            f"Spearman(|gap|, dev) = {_num(spearmanr(gap.abs().numpy(), d.numpy())[0], '.3f')}")
    out("")
    out("Ten months with the largest |gap|:")
    out("| month | sum r | sum c | gap | rel | banks over | W_hat row dev | W_hat col dev | RAS row dev | RAS col dev |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(range(inp.t), key=lambda k: -gap[k].abs().item())[:10]:
        p = inp.periods[t]
        out(f"| {p[1]} {p[0]} | {sr[t].item():,.0f} | {sc[t].item():,.0f} | {gap[t].item():+,.0f} | "
            f"{rel[t].item():.3f} | {int(over_r[t].sum() + over_c[t].sum())} | "
            f"{dev['W_hat'][0][t].item():,.4g} | {dev['W_hat'][1][t].item():,.4g} | "
            f"{dev['RAS'][0][t].item():,.4g} | {dev['RAS'][1][t].item():,.4g} |")
    out("If the deviations sit in the failing months and track |gap|, the misfit is the data (an "
        "unbalanced closed system), not the projection; if not, look at the cap or the solver.")


def check_row_normalised(out, cfg, inp, args) -> None:
    """Re-estimate with exposure from the row-normalised network (item 12).

    exposure_i = sum_j (W_ij / sum_k W_ik) dose_pre_j; banks with no interbank row
    in a month get exposure 0.
    """
    out("## Row-normalised exposure mapping (item 12): exposure_i = sum_j (W_ij / sum_k W_ik) dose_pre_j")
    out(f"Weights: {cfg.weighting} (HT ones re-estimated from the row-normalised exposure); banks "
        "with no interbank row in a month get exposure 0.")
    _zero_row_diagnostic(out, inp)
    rows = []
    for net, W in [("W_hat", inp.W_main), ("RAS", inp.W_ras)]:
        d = apply_weighting(did_design(_row_normalise(W, _no_lending_rows(inp)), inp.dose_pre, inp.controls_pre,
                                       inp.timing), cfg.weighting)
        for name, y in inp.y.items():
            rows.append((name, net, _ref(name, net, inp, cfg, args),
                         _fit(y, d, n_boot=args.n_boot, seed=cfg.bootstrap_seed)))
    _did_table(out, rows)


# --- item 12b ----------------------------------------------------------------
# Fixed before any centrality result was seen; not tuned. Row-normalised W has spectral radius <= 1,
# so (I - alpha W~)^-1 exists for alpha < 1.
CENTRALITY_ALPHA = 0.5


def _katz_bonacich(W: torch.Tensor, zero_rows: torch.Tensor, alpha: float = CENTRALITY_ALPHA) -> torch.Tensor:
    """Return the Katz-Bonacich exposure matrix M = W~ (I - alpha W~)^-1.

    M_t = W~_t (I - alpha W~_t)^-1 = sum_k alpha^(k-1) W~_t^k, W~ row-normalised:
    exposure M d counts the dose of direct counterparties plus counterparties'
    counterparties, discounted by alpha per hop (Katz-Bonacich centrality with
    the dose as the source term).
    """
    Wn = _row_normalise(W, zero_rows)
    eye = torch.eye(W.shape[-1], dtype=W.dtype).expand_as(Wn)
    return Wn @ torch.linalg.inv(eye - alpha * Wn)


def check_centrality_exposure(out, cfg, inp, args) -> None:
    """Re-estimate with Katz-Bonacich centrality exposure (item 12b, exploratory).

    exposure_i = [W~ (I - 0.5 W~)^-1 dose_pre]_i with W~ row-normalised (direct
    plus indirect counterparties); Holm correction within this check's exposure
    coefficients only.
    """
    out(f"## Centrality-based exposure (item 12b, exploratory): exposure_i = [W~ (I - {CENTRALITY_ALPHA} W~)^-1 "
        "dose_pre]_i, W~ row-normalised")
    out(f"Katz-Bonacich exposure, alpha = {CENTRALITY_ALPHA} fixed in advance. Leverage dose, Eq. (2), "
        f"weights {cfg.weighting}. Not part of any main-run test family; Holm below is within this "
        "check's exposure coefficients only.")
    _zero_row_diagnostic(out, inp)
    zero_rows = _no_lending_rows(inp)
    rows = []
    for net, W in [("W_hat", inp.W_main), ("RAS", inp.W_ras)]:
        M = _katz_bonacich(W, zero_rows)
        d = apply_weighting(did_design(M, inp.dose_pre, inp.controls_pre, inp.timing), cfg.weighting)
        d_row = did_design(_row_normalise(W, zero_rows), inp.dose_pre, inp.controls_pre, inp.timing)
        base = (inp.design_main if net == "W_hat" else inp.design_ras).exposure_x_post
        post_cells = inp.timing.post(inp.t).bool().unsqueeze(0).expand_as(base)
        for label, other in [("direct exposure", base), ("row-normalised direct exposure", d_row.exposure_x_post)]:
            a, b = d.exposure_x_post[post_cells], other[post_cells]
            out(f"- {net}: corr(centrality exposure, {label}) on post cells = "
                f"{_num(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), '.3f')} ({int(post_cells.sum())} cells)")
        for name, y in inp.y.items():
            rows.append((name, net, _ref(name, net, inp, cfg, args),
                         _fit(y, d, n_boot=args.n_boot, seed=cfg.bootstrap_seed)))
    out("")
    _did_table(out, rows)
    ps = [r[3].p_exposure for r in rows]
    finite = [p for p in ps if math.isfinite(p)]
    holm = iter(_holm(finite))
    out("")
    out("| outcome | network | exposure wild p | Holm p |")
    out("|---|---|---|---|")
    for (name, net, _, _), p in zip(rows, ps):
        out(f"| {name} | {net} | {_p(p)} | {_p(next(holm)) if math.isfinite(p) else 'n/a'} |")


# --- item 11b ----------------------------------------------------------------
def _main_outcomes(cfg, inp):
    """The run's own panel and P&L outcomes (as build_inputs builds them), before slicing."""
    return inp.panel, inp.outcomes_raw


def _construct(y: torch.Tensor, cfg, inp, quarterly: bool) -> torch.Tensor:
    """Extracted outcome -> regression outcome, same steps as build_inputs."""
    return outcome_ratio(y, inp.periods, inp.size, quarterly, cfg.pnl_ytd_cumulative)


def _total_effect_table(out, rows: list[tuple[str, object]]) -> None:
    out("| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, te in rows:
        out(f"| {label} | {_num(te.beta_dose, '.4g')} | {_num(te.se_dose, '.3g')} | {_p(te.p_dose)} | "
            f"{_num(te.beta_lead, '.4g')} | {_p(te.p_lead)} | {_num(te.effect_1sd, '.3g')} | "
            f"{_num(te.pre_mean, '.3g')} | {te.n_obs} | {te.n_clusters} | {te.note or '-'} |")


def check_written_off_zero(out, cfg, inp, args) -> None:
    """Compare loans written off with blank read as missing vs as 0 (item 11b)."""
    name = "loan_written_off"
    main_zero = name in cfg.blank_as_zero
    main_label = "blank = 0" if main_zero else "blank = missing"
    out("## Loans written off: blank read as missing vs blank read as 0 (item 11b)")
    out("The bulletin often prints no written-off row. 'Blank = missing' leaves it NaN; 'blank = 0' "
        "reads it as 0 wherever the bank reported some other P&L line that month (nothing written "
        "off yet this fiscal year); with no P&L row at all it stays missing. Zero-filling happens on "
        "the extracted (fiscal-YTD) figure, before differencing; a 0 after a positive YTD in the same "
        f"fiscal year then shows up as a negative flow. This config's main version (blank_as_zero): "
        f"{main_label} (user decision 2026-11-04).")
    try:
        _, outcomes = _main_outcomes(cfg, inp)
    except ValueError as e:
        out(f"not computed: {e}")
        return
    raw = outcomes.loan_written_off[:, : inp.t]
    filled = zero_fill_blanks(outcomes, [name]).loan_written_off[:, : inp.t]
    reported = pnl_reported(outcomes)[:, : inp.t]
    blank = raw.isnan()
    out("")
    out("| bank | written-off reported | of which exactly 0 | blank, other P&L present (-> 0) | no P&L row (stays missing) |")
    out("|---|---|---|---|---|")
    for i, bank in enumerate(inp.bfi_names):
        out(f"| {bank} | {int((~blank[i]).sum())} | {int((raw[i] == 0).sum())} | "
            f"{int((blank[i] & reported[i]).sum())} | {int((blank[i] & ~reported[i]).sum())} |")
    out(f"| total | {int((~blank).sum())} | {int((raw == 0).sum())} | {int((blank & reported).sum())} | "
        f"{int((blank & ~reported).sum())} |")
    out("Explicit zeros in the extraction would suggest a blank means 'not reported', not 0.")

    quarterly = args.quarterly or cfg.outcome_construction == "quarterly"
    y_miss = _construct(raw, cfg, inp, quarterly)
    y_zero = _construct(filled, cfg, inp, quarterly)
    out("")
    out(f"Regression outcome ({'quarterly' if quarterly else 'monthly'}, / size):")
    out("| version | finite cells | banks with any | negative share |")
    out("|---|---|---|---|")
    for label, y in [("blank = missing", y_miss), ("blank = 0", y_zero)]:
        ok = y.isfinite()
        n_ok = int(ok.sum())
        out(f"| {label} | {n_ok} | {int(ok.any(dim=1).sum())} | "
            f"{_num(int((ok & (y < 0)).sum()) / n_ok if n_ok else math.nan, '.3f')} |")

    out("")
    out(f"Eq. A' ({cfg.weighting}); 'main' columns = blank as missing, 'alt' = blank as 0:")
    rows, clusters = [], []
    for net, d in [("W_hat", inp.design_main), ("RAS", inp.design_ras)]:
        ref = _fit(y_miss, d, n_boot=args.n_boot, seed=cfg.bootstrap_seed)
        alt = _fit(y_zero, d, n_boot=args.n_boot, seed=cfg.bootstrap_seed)
        rows.append((name, net, ref, alt))
        clusters.append(f"{net} {ref.n_clusters} -> {alt.n_clusters}")
    _did_table(out, rows)
    out("Clusters (missing -> 0): " + ", ".join(clusters))

    out("")
    out(f"Headline total effect (dose-only, quarterly, from {TOTAL_EFFECT_START[1]} {TOTAL_EFFECT_START[0]}):")
    try:
        te_rows = []
        for label, zf in [("blank = missing", ()), ("blank = 0", (name,))]:
            _, te = fit_total_effect(cfg, inp, n_boot=args.n_boot, seed=cfg.bootstrap_seed, zero_fill=zf, fields=[name])
            te_rows.append((label, te[name]))
    except ValueError as e:
        out(f"not computed: {e}")
        return
    _total_effect_table(out, te_rows)


# --- alternative dose: paid-up capital ---------------------------------------
def check_paid_up_dose(out, cfg, inp, args) -> None:
    """Estimate with paid-up capital as an alternative dose (item 1)."""
    out("## Paid-up capital as an alternative dose")
    out("dose_pu = (required paid-up capital - paid-up capital) / size, frozen at its pre-window mean "
        "like the main dose; same pre window, transition and post as the leverage dose (if the "
        "paid-up directive has its own dates, that needs its own timing). The balance-sheet row is "
        "matched on keywords " + " + ".join(f"'{k}'" for k in ("PAID", "CAPITAL")) + ", not confirmed "
        "against a printed bulletin: check the matched labels below.")
    pu = load_paid_up_capital(cfg.csv_path, inp.panel)
    out("")
    if not pu.matched_labels:
        out("No balance-sheet row matched. Labels containing CAPITAL: " + "; ".join(pu.capital_labels[:30]))
        return
    out("Matched labels (panel cells filled): " + "; ".join(f"'{k}' ({v})" for k, v in
                                                          sorted(pu.matched_labels.items(), key=lambda kv: -kv[1])))
    out(f"Cells with a second, different matched value (ignored, first kept): {pu.n_conflicts}")
    other = [lbl for lbl in pu.capital_labels if lbl not in pu.matched_labels]
    if other:
        out("Other labels containing CAPITAL (not used): " + "; ".join(other[:20]))

    vals = pu.values[:, : inp.t]
    pre = slice(0, inp.timing.pre_end_idx + 1)
    pre_pu = vals[:, pre].nanmean(dim=1)
    pre_size = inp.size[:, pre].nanmean(dim=1)
    last_pu = torch.tensor([v[v.isfinite()][-1].item() if v.isfinite().any() else math.nan for v in vals])
    out("")
    out(f"Per bank, pre window {inp.periods[0][1]} {inp.periods[0][0]} .. "
        f"{inp.periods[inp.timing.pre_end_idx][1]} {inp.periods[inp.timing.pre_end_idx][0]} (NPR mn):")
    out("| bank | pre cells | paid-up (pre mean) | size (pre mean) | paid-up / size | last paid-up / pre | main dose_pre |")
    out("|---|---|---|---|---|---|---|")
    for i, bank in enumerate(inp.bfi_names):
        out(f"| {bank} | {int(vals[i, pre].isfinite().sum())} | {_num(pre_pu[i].item(), ',.0f')} | "
            f"{_num(pre_size[i].item(), ',.0f')} | {_num((pre_pu[i] / pre_size[i]).item(), '.4f')} | "
            f"{_num((last_pu[i] / pre_pu[i]).item(), '.2f')} | {inp.dose_pre[i].item():.4f} |")
    ok = pre_pu.isfinite()
    if int(ok.sum()) >= 3:
        out(f"Pre-window paid-up capital: median {pre_pu[ok].median().item():,.0f}, "
            f"min {pre_pu[ok].min().item():,.0f}, max {pre_pu[ok].max().item():,.0f} (n={int(ok.sum())}); "
            f"corr(paid-up / size, main dose_pre) = "
            f"{torch.corrcoef(torch.stack([(pre_pu / pre_size)[ok], inp.dose_pre[ok]]))[0, 1].item():.3f}")
    out(f"'last paid-up / pre' uses the last month in this run's window ({inp.periods[-1][1]} {inp.periods[-1][0]}).")

    req = cfg.paid_up_capital_requirement
    out("")
    if req is None:
        out("paid_up_capital_requirement is null in the config: fill it in (NPR mn) from the policy text, "
            "then re-run this check for the estimates.")
        return
    try:
        dose_pu_pre = paid_up_dose_pre(cfg, inp, req)
    except ValueError as e:
        out(f"not computed: {e} (no paid-up row in the pre window)")
        return
    out(f"Requirement {req:,.0f} NPR mn: {int((dose_pu_pre > 0).sum())}/{inp.n} banks below it pre-mandate, "
        f"dose_pu_pre range [{dose_pu_pre.min().item():.4f}, {dose_pu_pre.max().item():.4f}], "
        f"SD {dose_pu_pre.std().item():.4f} (main dose SD {inp.dose_pre.std().item():.4f}). Betas are per unit of "
        "each dose; compare the 1-SD effects.")
    out("")
    out(f"Eq. A' ({cfg.weighting}); main = leverage dose, alt = paid-up dose (exposure rebuilt from it):")
    rows = []
    for net, W in [("W_hat", inp.W_main), ("RAS", inp.W_ras)]:
        d = apply_weighting(did_design(W, dose_pu_pre, inp.controls_pre, inp.timing), cfg.weighting)
        for name, y in inp.y.items():
            rows.append((name, net, _ref(name, net, inp, cfg, args),
                         _fit(y, d, n_boot=args.n_boot, seed=cfg.bootstrap_seed)))
    _did_table(out, rows)
    out("")
    out("Headline total effect (dose-only, quarterly), leverage dose vs paid-up dose:")
    try:
        _, te_main = fit_total_effect(cfg, inp, n_boot=args.n_boot, seed=cfg.bootstrap_seed)
        _, te_pu = fit_total_effect(cfg, replace(inp, dose_pre=dose_pu_pre), n_boot=args.n_boot,
                                    seed=cfg.bootstrap_seed)
    except ValueError as e:
        out(f"not computed: {e}")
        return
    for name in OUTCOME_FIELDS:
        out("")
        out(f"{name}:")
        _total_effect_table(out, [("leverage dose", te_main[name]), ("paid-up dose", te_pu[name])])
    _paid_up_robustness(out, cfg, inp, args, vals, req, dose_pu_pre, te_pu)


def _paid_up_robustness(out, cfg, inp, args, vals, req: float, dose_pu_pre, te_pu) -> None:
    """Total-effect robustness for the paid-up dose (Sec 5b item 1, re-run 2026-11-04): is it the
    paid-up shortfall or bank size? log size_pre x post is already a control in every fit."""
    size_term = freeze_pre_mandate(req / inp.size, inp.timing)
    pu_term = -freeze_pre_mandate(vals / inp.size, inp.timing)
    corr = lambda a, b: torch.corrcoef(torch.stack([a, b]))[0, 1].item()  # noqa: E731
    out("")
    out("### Paid-up dose robustness (total effect)")
    out(f"dose_pu_pre = size term (req / size, frozen) + paid-up term (-paid-up / size, frozen); "
        f"max |sum - dose_pu_pre| = {(size_term + pu_term - dose_pu_pre).abs().max().item():.2e}. "
        f"Size term range [{size_term.min().item():.4f}, {size_term.max().item():.4f}], paid-up term "
        f"[{pu_term.min().item():.4f}, {pu_term.max().item():.4f}]; corr(size term, dose_pu) = "
        f"{corr(size_term, dose_pu_pre):.3f}, corr(paid-up term, dose_pu) = {corr(pu_term, dose_pu_pre):.3f}, "
        f"corr(size term, log size_pre) = {corr(size_term, inp.controls_pre['log_size']):.3f}. "
        "Every fit below already has log size_pre x post among its controls.")
    try:
        _, te_split = fit_total_effect(cfg, replace(inp, dose_pre=size_term), n_boot=args.n_boot,
                                       seed=cfg.bootstrap_seed, extra_doses={"paid_up_term": pu_term})
        pre_size = inp.size[:, : inp.timing.pre_end_idx + 1].nanmean(dim=1)
        _, te_frozen = fit_total_effect(cfg, replace(inp, dose_pre=dose_pu_pre), n_boot=args.n_boot,
                                        seed=cfg.bootstrap_seed, denominator=pre_size)
        _, te_full_cr1 = fit_total_effect(cfg, replace(inp, dose_pre=dose_pu_pre), n_boot=0, seed=cfg.bootstrap_seed)
        loo = [fit_total_effect(cfg, replace(inp, dose_pre=dose_pu_pre), n_boot=0, seed=cfg.bootstrap_seed,
                                drop_bank=i)[1] for i in range(inp.n)]
        ext = extended_panel(cfg.csv_path, inp.bfi_names, inp.periods[-1])
        stocks = _paid_up_stocks(load_paid_up_capital(cfg.csv_path, ext).values, positive_size(bank_size(ext)),
                                 pre_size)
        _, te_stock = fit_total_effect(cfg, replace(inp, dose_pre=dose_pu_pre), n_boot=args.n_boot,
                                       seed=cfg.bootstrap_seed, stock_outcomes=stocks)
        _, te_race = fit_total_effect(cfg, replace(inp, dose_pre=dose_pu_pre), n_boot=args.n_boot,
                                      seed=cfg.bootstrap_seed, extra_doses={"leverage_dose": inp.dose_pre})
    except ValueError as e:
        out(f"not computed: {e}")
        return

    out("")
    out("(a) Decomposition: both terms entered together (each with its own lead / transition / post). "
        "A binding paid-up floor predicts the same post beta on both (same units); a pure size "
        "effect puts it on the size term only. The size term is close to a function of log size_pre, "
        "so expect it to be imprecise.")
    out("| outcome | combined beta (p) | size term beta | SE | wild p | 1-SD | paid-up term beta | SE | wild p | 1-SD |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for name in OUTCOME_FIELDS:
        te, c = te_split[name], te_pu[name]
        x = te.extra.get("paid_up_term", {})
        out(f"| {name} | {_num(c.beta_dose, '.4g')} ({_p(c.p_dose)}) | {_num(te.beta_dose, '.4g')} | "
            f"{_num(te.se_dose, '.3g')} | {_p(te.p_dose)} | {_num(te.effect_1sd, '.3g')} | "
            f"{_num(x.get('beta', math.nan), '.4g')} | {_num(x.get('se', math.nan), '.3g')} | "
            f"{_p(x.get('p'))} | {_num(x.get('effect_1sd', math.nan), '.3g')} |")

    out("")
    out("(c) Outcomes divided by the frozen pre-window size (item 11c) instead of current size, so "
        "asset growth from capital raising can't shrink the ratio:")
    for name in OUTCOME_FIELDS:
        out("")
        out(f"{name}:")
        _total_effect_table(out, [("current size", te_pu[name]), ("frozen pre size", te_frozen[name])])

    out("")
    out("(d) Leave-one-bank-out, CR1 only: post beta (t) with that bank's outcomes dropped.")
    out("| dropped | " + " | ".join(OUTCOME_FIELDS) + " |")
    out("|---|" + "---|" * len(OUTCOME_FIELDS))
    _t = lambda r: r.beta_dose / r.se_dose if math.isfinite(r.se_dose) and r.se_dose > 0 else math.nan  # noqa: E731
    out("| none | " + " | ".join(f"{_num(te_full_cr1[n].beta_dose, '.4g')} ({_num(_t(te_full_cr1[n]), '.2f')})"
                                 for n in OUTCOME_FIELDS) + " |")
    for i, bank in enumerate(inp.bfi_names):
        out(f"| {bank} | " + " | ".join(f"{_num(loo[i][n].beta_dose, '.4g')} ({_num(_t(loo[i][n]), '.2f')})"
                                        for n in OUTCOME_FIELDS) + " |")
    for name in OUTCOME_FIELDS:
        b0 = te_full_cr1[name].beta_dose
        bs = [(bank, loo[i][name].beta_dose) for i, bank in enumerate(inp.bfi_names)
              if math.isfinite(loo[i][name].beta_dose)]
        if not bs or not math.isfinite(b0):
            continue
        infl = max(bs, key=lambda v: abs(v[1] - b0))
        out(f"{name}: LOO range [{min(b for _, b in bs):.4g}, {max(b for _, b in bs):.4g}], "
            f"sign flips {sum(1 for _, b in bs if b * b0 < 0)}/{len(bs)}, most influential {infl[0]} ({infl[1]:.4g})")

    out("")
    out("(f) First stage and (g) asset growth: the same total-effect regression on balance-sheet "
        "stocks at quarter-end months (no differencing, no size scaling). 'paid_up / size_pre' is "
        "in the dose's units: beta = post-period paid-up raised per unit of pre-window shortfall, "
        "averaged over all post quarters (so below 1 even under full compliance, since post starts "
        "a year before the deadline). 'log size' tests the denominator channel behind (c).")
    _total_effect_table(out, [(name, te) for name, te in te_stock.items()])

    out("")
    out("(h) Horse race: paid-up dose and leverage dose entered together (each with its own lead / "
        "transition / post).")
    out("| outcome | paid-up beta | SE | wild p | 1-SD | leverage beta | SE | wild p | 1-SD |")
    out("|---|---|---|---|---|---|---|---|---|")
    for name in OUTCOME_FIELDS:
        te = te_race[name]
        x = te.extra.get("leverage_dose", {})
        out(f"| {name} | {_num(te.beta_dose, '.4g')} | {_num(te.se_dose, '.3g')} | {_p(te.p_dose)} | "
            f"{_num(te.effect_1sd, '.3g')} | {_num(x.get('beta', math.nan), '.4g')} | "
            f"{_num(x.get('se', math.nan), '.3g')} | {_p(x.get('p'))} | {_num(x.get('effect_1sd', math.nan), '.3g')} |")
    out(f"corr(dose_pu_pre, leverage dose_pre) = {torch.corrcoef(torch.stack([dose_pu_pre, inp.dose_pre]))[0, 1].item():.3f}")


# NRB Unified Directive 2073, dir. 23 pt. 4(2): Class "ka" banks meet the paid-up minimum by Asar
# end 2074 (design doc Sec 4a).
PAID_UP_DEADLINE = (2074, "Asar")


def _months_between(a, b) -> int:
    (ya, ma), (yb, mb) = period_key(a), period_key(b)
    return (yb - ya) * 12 + (mb - ma)


def check_paid_up_path(out, cfg, inp, args) -> None:
    """Trace each bank's paid-up capital path to the requirement (item 1)."""
    req = cfg.paid_up_capital_requirement
    out("## Paid-up capital path: when each bank reaches the requirement")
    if req is None:
        out("paid_up_capital_requirement is null in the config: nothing to compare against.")
        return
    try:
        panel = extended_panel(cfg.csv_path, inp.bfi_names, inp.periods[-1])
    except ValueError as e:
        out(f"not computed: {e}")
        return
    pu = load_paid_up_capital(cfg.csv_path, panel).values
    periods = panel.periods
    keys = [period_key(p) for p in periods]
    deadline_key = period_key(PAID_UP_DEADLINE)
    at = lambda i, p: _at(pu, periods, i, p)  # noqa: E731

    snaps = [(y, "Asar") for y in range(periods[0][0], periods[-1][0] + 1)
             if (y, "Asar") in periods] + [tuple(cfg.pre_window_end), periods[-1]]
    snaps = sorted(set(snaps), key=period_key)
    out(f"Requirement {req:,.0f} NPR mn, deadline {PAID_UP_DEADLINE[1]} end {PAID_UP_DEADLINE[0]} (Unified "
        f"Directive 2073, dir. 23 pt. 4). Panel {periods[0]} .. {periods[-1]} (paid-up after Falgun 2075 "
        "is in the corrupted bulletins, Sec 5a). 'first month >=' is the first bulletin month at or "
        "above the requirement; 'vs deadline' = months after the deadline (negative = early); "
        "'months below after' counts later months back under it.")
    out("")
    out("| bank | " + " | ".join(f"{m} {y}" for y, m in snaps) + " | first month >= | vs deadline | months below after |")
    out("|---|" + "---|" * (len(snaps) + 3))
    n_by_deadline, n_ever = 0, 0
    for i, bank in enumerate(inp.bfi_names):
        row = pu[i]
        hit = [t for t in range(len(periods)) if math.isfinite(row[t].item()) and row[t].item() >= req]
        if hit:
            first = periods[hit[0]]
            n_ever += 1
            n_by_deadline += int(keys[hit[0]] <= deadline_key)
            below = sum(1 for t in range(hit[0], len(periods)) if math.isfinite(row[t].item()) and row[t].item() < req)
            tail = f"{first[1]} {first[0]} | {_months_between(PAID_UP_DEADLINE, first):+d} | {below} |"
        else:
            tail = "never | n/a | n/a |"
        out(f"| {bank} | " + " | ".join(_num(at(i, s), ',.0f') for s in snaps) + f" | {tail}")
    out(f"Reached {req:,.0f} by the deadline: {n_by_deadline}/{inp.n}; within the panel at all: {n_ever}/{inp.n}.")
    pre_end = tuple(cfg.pre_window_end)
    start = inp.periods[0]  # pre-window start: earlier raises predate the FY 2072/73 announcement
    grow = [(at(i, pre_end) - at(i, start), at(i, periods[-1]) - at(i, start)) for i in range(inp.n)]
    shares = [a / b for a, b in grow if math.isfinite(a) and math.isfinite(b) and b > 0]
    if shares:
        shares.sort()
        raised = [bank for i, bank in enumerate(inp.bfi_names) if math.isfinite(grow[i][0]) and grow[i][0] > 0]
        out(f"Anticipation (Sec 4a timing caveat): share of each bank's {start[1]} {start[0]} .. {periods[-1][1]} "
            f"{periods[-1][0]} paid-up increase already in place by {pre_end[1]} {pre_end[0]}: median "
            f"{shares[len(shares) // 2]:.2f}, range [{shares[0]:.2f}, {shares[-1]:.2f}] (n={len(shares)}); "
            f"banks with any increase inside the pre window: {', '.join(raised) or 'none'}.")


# --- item 1, threats to the paid-up result -------------------------------------
# Approximate dates: the '*' joint-operation dates in NRB's "List of BFIs" (user-pasted
# 2026-11-04, '*' = joint operation after merger/acquisition), hand-converted AD -> BS. Not
# facts about the mergers: the data check below dates them from the bulletins.
MERGER_HINTS = [("SBL", "lumbini", (2073, "Asar")), ("RBB", None, (2075, "Baisakh"))]
MERGER_WINDOW = 4  # months either side of the hinted date
CLEAN_END = (2075, "Falgun")  # last audited-clean bulletin (Sec 5a)
POST_DEADLINE = (2075, "Asar")  # one year after the paid-up deadline, still clean
BUNCH_THRESHOLDS = (4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000)
N_PERM = 9999


def _at(pu: torch.Tensor, periods, i: int, p) -> float:
    """Value at the last month <= p with a finite cell (bulletin months can be absent)."""
    k = period_key(p)
    cand = [t for t, q in enumerate(periods) if period_key(q) <= k and math.isfinite(pu[i, t].item())]
    return pu[i, cand[-1]].item() if cand else math.nan


def _paid_up_stocks(pu: torch.Tensor, size: torch.Tensor, pre_size: torch.Tensor) -> dict[str, torch.Tensor]:
    nan = torch.tensor(float("nan"))
    return {"paid_up / size_pre": pu / pre_size.unsqueeze(1),
            "log paid_up": torch.where(pu > 0, pu.clamp(min=1e-12).log(), nan),
            "log size": torch.where(size > 0, size.clamp(min=1e-12).log(), nan)}


def paid_up_dose_pre(cfg, inp, req):
    """Build the paid-up dose dose_pu_pre = (req - paid-up) / size, frozen pre-window.

    Frozen at its pre-window mean like the main dose; the one construction
    check_paid_up_dose, _paid_up_setup and power_mde share. Raises ValueError
    like the loaders do.
    """
    vals = load_paid_up_capital(cfg.csv_path, inp.panel).values[:, : inp.t]
    return freeze_pre_mandate((req - vals) / inp.size, inp.timing)


def _paid_up_setup(cfg, inp, req):
    """dose_pu_pre exactly as check_paid_up_dose builds it, plus the extended-panel stocks.
    Raises ValueError like the loaders do."""
    dose = paid_up_dose_pre(cfg, inp, req)
    ext = extended_panel(cfg.csv_path, inp.bfi_names, inp.periods[-1])
    timing = mandate_timing(ext.periods, cfg.pre_window_end, cfg.effective_period)
    pu_ext = load_paid_up_capital(cfg.csv_path, ext).values
    size_ext = positive_size(bank_size(ext))
    pre_size = inp.size[:, : inp.timing.pre_end_idx + 1].nanmean(dim=1)
    return dose, ext, timing, pu_ext, size_ext, pre_size, _paid_up_stocks(pu_ext, size_ext, pre_size)


def check_paid_up_threats(out, cfg, inp, args) -> None:
    """Test threats to the paid-up result: pre-trend, mergers, catch-up (item 1)."""
    out("## Threats to the paid-up result (Sec 5b item 1): pre-trend, mergers, catch-up")
    req = cfg.paid_up_capital_requirement
    if req is None:
        out("paid_up_capital_requirement is null in the config: nothing to test.")
        return
    try:
        dose, ext, timing, pu_ext, size_ext, pre_size, stocks = _paid_up_setup(cfg, inp, req)
    except ValueError as e:
        out(f"not computed: {e}")
        return
    _threat_trend(out, cfg, inp, args, dose, ext, timing, stocks)
    _threat_mergers(out, cfg, inp, args, req, dose, ext, pu_ext, size_ext, pre_size, stocks)
    _threat_catch_up(out, cfg, inp, args, req, ext, pu_ext)


def _threat_trend(out, cfg, inp, args, dose, ext, timing, stocks) -> None:
    out("")
    out("### 1. Pre-trend in the balance-sheet stocks")
    out("dose_pu x fiscal-year bin on quarter-end stocks; reference bin -1 = the pre-window fiscal "
        "year (its quarter-ends up to Mangsir 2072); Z_pre x post controls, bank + period FE, "
        "unweighted. A negative pre bin means high-dose banks were lower then relative to the "
        "reference, i.e. they were growing faster into it. Cells: beta (SE, wild p).")
    periods, t = ext.periods, len(ext.periods)
    bins = event_bins(periods, timing)
    post = timing.post(t).unsqueeze(0)
    controls = pre_x_post_controls(inp.controls_pre, post)
    ones = torch.ones(inp.n, t)
    order = list(dict.fromkeys(bins))
    fits: dict[str, tuple | None] = {}
    for name, s in stocks.items():
        y = quarter_end_only(s, periods)
        ok = y.isfinite()
        cells = {b: sum(int(ok[:, i].sum()) for i, bt in enumerate(bins) if bt == b) for b in order}
        if cells.get(-1, 0) == 0:
            fits[name] = None
            continue
        fits[name] = (cells, fit_event_study(y, dose, None, bins, controls, ones, reference_bin=-1,
                                             n_boot=args.n_boot, seed=cfg.bootstrap_seed))
    out("| bin | months | " + " | ".join(stocks) + " |")
    out("|---|---|" + "---|" * len(stocks))
    for b in order:
        cols = [i for i, bt in enumerate(bins) if bt == b]
        span = f"{periods[cols[0]][1][:3]} {periods[cols[0]][0]} - {periods[cols[-1]][1][:3]} {periods[cols[-1]][0]}"
        row = []
        for name in stocks:
            if fits[name] is None:
                row.append("n/a (empty reference bin)")
                continue
            cells, es = fits[name]
            if b == -1:
                row.append(f"0 (ref, {cells[b]} cells)")
            elif cells[b] == 0:
                row.append("no cells")
            else:
                e = es.dose[str(b)]
                row.append(f"{_num(e['beta'], '.3g')} ({_num(e['se'], '.2g')}, p {_p(e['p'])})")
        out(f"| {b} | {span} | " + " | ".join(row) + " |")

    out("")
    out("Pooled total effect, without and with linear trends: dose_pu x trend and log size_pre x "
        "trend (years since the pre-window end, whole panel). With trends, post beta is the shift "
        "beyond each bank's dose- and size-predicted linear path. Conservative: a trend fitted on "
        "the whole panel also soaks up an effect that ramps up towards the deadline.")
    trends = {"dose": dose, "log_size_pre": inp.controls_pre["log_size"]}
    pl = ["nim", "net_profit_loss"]
    try:
        inp_pu = replace(inp, dose_pre=dose)
        _, s0 = fit_total_effect(cfg, inp_pu, n_boot=args.n_boot, seed=cfg.bootstrap_seed, stock_outcomes=stocks)
        _, s1 = fit_total_effect(cfg, inp_pu, n_boot=args.n_boot, seed=cfg.bootstrap_seed, stock_outcomes=stocks,
                                 trend_controls=trends)
        _, p0 = fit_total_effect(cfg, inp_pu, n_boot=args.n_boot, seed=cfg.bootstrap_seed, fields=pl)
        _, p1 = fit_total_effect(cfg, inp_pu, n_boot=args.n_boot, seed=cfg.bootstrap_seed, fields=pl,
                                 trend_controls=trends)
    except ValueError as e:
        out(f"not computed: {e}")
        return
    rows = []
    for name in stocks:
        rows += [(f"{name}", s0[name]), (f"{name}, + trends", s1[name])]
    for name in pl:
        rows += [(f"{name}", p0[name]), (f"{name}, + trends", p1[name])]
    _total_effect_table(out, rows)
    out("Not done here: a Rambachan-Roth sensitivity bound (how large a trend violation would explain "
        "the post beta); with about 7 pre quarters per bank the event study above is the input it needs.")


def _threat_mergers(out, cfg, inp, args, req, dose, ext, pu_ext, size_ext, pre_size, stocks) -> None:
    out("")
    out("### 2. Mergers inside the window")
    out("Hinted dates are approximate (NRB 'List of BFIs' '*' joint-operation dates, hand-converted "
        "AD -> BS). The tables date them from the bulletins: a merger shows as a size and paid-up "
        f"jump beyond the 99th percentile of month-to-month |log size change| in the clean bulletins "
        f"(up to {CLEAN_END[1]} {CLEAN_END[0]}, adjacent panel months only).")
    full = filter_commercial_banks(load_panel(cfg.csv_path))
    names, fper = list(full.bfi_names), full.periods
    fsize = positive_size(bank_size(full).to(torch.float64))
    fpu = load_paid_up_capital(cfg.csv_path, full).values.to(torch.float64)
    logs = fsize.log()
    clean_cols = [t for t, p in enumerate(fper) if period_key(p) <= period_key(CLEAN_END)]
    adjacent = [b for b in clean_cols[1:] if _months_between(fper[b - 1], fper[b]) == 1]
    changes = [abs((logs[i, b] - logs[i, b - 1]).item()) for i in range(len(names)) for b in adjacent
               if math.isfinite((logs[i, b] - logs[i, b - 1]).item())]
    if not changes:
        out("no adjacent clean months with finite size; nothing to compare")
        return
    p99 = torch.quantile(torch.tensor(changes), 0.99).item()
    out(f"Clean-month |log size change|: median {torch.tensor(changes).median().item():.3f}, "
        f"99th pct {p99:.3f} (n={len(changes)}).")

    absorbed: dict[str, str] = {}
    for bank, sub, date in MERGER_HINTS:
        out("")
        if bank not in names:
            out(f"{bank}: not among the commercial-bank labels")
            continue
        i = names.index(bank)
        partners = [n for n in names if sub and sub in n.lower()]
        j = names.index(partners[0]) if partners else None
        if sub and not partners:
            out(f"{bank}: no label containing '{sub}' among the commercial banks")
        if len(partners) > 1:
            out(f"{bank}: several labels contain '{sub}': {partners}; using {partners[0]}")
        if j is not None:
            absorbed[bank] = names[j]
        win = [t for t, p in enumerate(fper) if abs(_months_between(date, p)) <= MERGER_WINDOW]
        out(f"{bank}, hinted {date[1]} {date[0]}" + (f", absorbing {names[j]}" if j is not None else "") + ":")
        head = "| month | size | dlog size | paid-up | d paid-up |"
        if j is not None:
            head += f" {names[j]} size | {names[j]} paid-up |"
        out(head)
        out("|---|---|---|---|---|" + ("---|---|" if j is not None else ""))
        jumps = []
        for t in win:
            dl = (logs[i, t] - logs[i, t - 1]).item() if t > 0 else math.nan
            dp = (fpu[i, t] - fpu[i, t - 1]).item() if t > 0 else math.nan
            flag = " (> p99)" if math.isfinite(dl) and abs(dl) > p99 else ""
            if flag:
                jumps.append((fper[t], dl))
            line = (f"| {fper[t][1]} {fper[t][0]} | {_num(fsize[i, t].item(), ',.0f')} | {_num(dl, '+.3f')}{flag} | "
                    f"{_num(fpu[i, t].item(), ',.0f')} | {_num(dp, '+,.0f')} |")
            if j is not None:
                line += f" {_num(fsize[j, t].item(), ',.0f')} | {_num(fpu[j, t].item(), ',.0f')} |"
            out(line)
        out(f"{bank}: size jumps > p99 in the window: "
            + (", ".join(f"{m} {y} ({d:+.3f})" for (y, m), d in jumps) or "none"))
        if j is not None:
            last = [t for t in range(len(fper)) if math.isfinite(fsize[j, t].item())]
            out(f"{names[j]}: last month with a finite size: " + (f"{fper[last[-1]][1]} {fper[last[-1]][0]}" if last else "never"))

    # Robustness: drop each hinted panel bank; SBL pro-forma (absorbed bank added before the merger).
    out("")
    out("Total effect of dose_pu on the stocks and the NIM / net-profit horse race (paid-up and "
        "leverage dose jointly), all banks vs dropping a merging bank vs pro-forma. Pro-forma: "
        "before the absorbed bank's last bulletin month, its paid-up and size are added to the "
        "acquirer's (a cell where it is missing stays missing); the acquirer's dose_pu, log size "
        "control and size_pre are rebuilt from the sums. Its P&L outcomes and leverage dose are "
        "left as reported (the absorbed bank's P&L is not added).")
    variants: list[tuple[str, object, dict, int | None]] = [("all banks", inp, stocks, None)]
    for bank, _, _ in MERGER_HINTS:
        if bank in inp.bfi_names:
            variants.append((f"drop {bank}", inp, stocks, inp.bfi_names.index(bank)))
    for bank, partner in absorbed.items():
        if bank not in inp.bfi_names:
            continue
        try:
            pf = _pro_forma(cfg, inp, req, dose, ext, pu_ext, size_ext, pre_size, bank, partner)
        except ValueError as e:
            out(f"pro-forma {bank} + {partner}: not computed: {e}")
            continue
        out(pf[0])
        variants.append((f"{bank} pro-forma", pf[1], pf[2], None))
    rows, race = [], []
    for label, v_inp, v_stocks, drop in variants:
        v_inp = v_inp if label.endswith("pro-forma") else replace(inp, dose_pre=dose)
        try:
            _, s = fit_total_effect(cfg, v_inp, n_boot=args.n_boot, seed=cfg.bootstrap_seed,
                                    stock_outcomes=v_stocks, drop_bank=drop)
            _, h = fit_total_effect(cfg, v_inp, n_boot=args.n_boot, seed=cfg.bootstrap_seed,
                                    fields=["nim", "net_profit_loss"], drop_bank=drop,
                                    extra_doses={"leverage_dose": inp.dose_pre})
        except ValueError as e:
            out(f"{label}: not computed: {e}")
            continue
        rows += [(f"{label}: {name}", s[name]) for name in v_stocks]
        race += [(label, name, h[name]) for name in ("nim", "net_profit_loss")]
    out("")
    _total_effect_table(out, rows)
    out("")
    out("| version | outcome | paid-up beta | SE | wild p | leverage beta | SE | wild p |")
    out("|---|---|---|---|---|---|---|---|")
    for label, name, te in race:
        x = te.extra.get("leverage_dose", {})
        out(f"| {label} | {name} | {_num(te.beta_dose, '.4g')} | {_num(te.se_dose, '.3g')} | {_p(te.p_dose)} | "
            f"{_num(x.get('beta', math.nan), '.4g')} | {_num(x.get('se', math.nan), '.3g')} | {_p(x.get('p'))} |")
    out("A merger is also a way to comply with the floor, so dropping or pro-forma-ing the bank "
        "removes part of the effect as well as the mechanical jump; report both next to the main fit.")


def _pro_forma(cfg, inp, req, dose, ext, pu_ext, size_ext, pre_size, bank, partner):
    ext_p = extended_panel(cfg.csv_path, [partner], inp.periods[-1])
    if ext_p.periods != ext.periods:
        raise ValueError("partner's extended panel has different periods")
    pu_p = load_paid_up_capital(cfg.csv_path, ext_p).values[0]
    size_p = positive_size(bank_size(ext_p)[0])
    alive = [t for t in range(len(ext.periods)) if math.isfinite(size_p[t].item())]
    if not alive:
        raise ValueError(f"{partner} has no finite size in the extended panel")
    last = alive[-1]
    i = inp.bfi_names.index(bank)
    pu_pf, size_pf = pu_ext.clone(), size_ext.clone()
    pu_pf[i, : last + 1] = pu_ext[i, : last + 1] + pu_p[: last + 1]
    size_pf[i, : last + 1] = size_ext[i, : last + 1] + size_p[: last + 1]
    start = ext.periods.index(inp.periods[0])
    pre = slice(start, start + inp.timing.pre_end_idx + 1)
    dose_pf, pre_size_pf = dose.clone(), pre_size.clone()
    dose_pf[i] = torch.nanmean((req - pu_pf[i, pre]) / size_pf[i, pre])
    pre_size_pf[i] = torch.nanmean(size_pf[i, pre])
    controls_pf = dict(inp.controls_pre)
    controls_pf["log_size"] = inp.controls_pre["log_size"].clone()
    controls_pf["log_size"][i] = torch.nanmean(size_pf[i, pre].log())
    if not (math.isfinite(dose_pf[i].item()) and math.isfinite(pre_size_pf[i].item())):
        raise ValueError(f"no pre-window cell where both {bank} and {partner} are reported")
    note = (f"pro-forma {bank} + {partner}: added through {ext.periods[last][1]} {ext.periods[last][0]}; "
            f"{bank} dose_pu_pre {dose[i].item():.4f} -> {dose_pf[i].item():.4f}, size_pre "
            f"{pre_size[i].item():,.0f} -> {pre_size_pf[i].item():,.0f}; acquirer cells made missing by a "
            f"missing partner cell: {int((size_pf[i, : last + 1].isnan() & size_ext[i, : last + 1].isfinite()).sum())}")
    v_inp = replace(inp, dose_pre=dose_pf, controls_pre=controls_pf)
    return note, v_inp, _paid_up_stocks(pu_pf, size_pf, pre_size_pf)


def _threat_catch_up(out, cfg, inp, args, req, ext, pu_ext) -> None:
    out("")
    out("### 3. Floor or catch-up?")
    out("Plain catch-up (low-capital banks raising anyway) and an absolute floor both make high-dose "
        "banks raise more. They differ in where the banks end up: a floor piles them up just above "
        f"{req:,.0f} whatever their start, catch-up keeps their order and spread.")
    periods = ext.periods
    pre_end = tuple(cfg.pre_window_end)
    snaps = [pre_end, PAID_UP_DEADLINE, POST_DEADLINE]
    snaps = [s for s in snaps if period_key(s) <= period_key(periods[-1])]
    lvl = {s: torch.tensor([_at(pu_ext, periods, i, s) for i in range(inp.n)], dtype=torch.float64) for s in snaps}

    out("")
    out("(a) Bunching: banks with paid-up in [T, 1.05 T) / [T, 1.10 T), at each snapshot. A floor "
        f"predicts a spike at T = {req:,.0f} after the deadline and none at the placebo thresholds.")
    out("| T | " + " | ".join(f"{m} {y}" for y, m in snaps) + " |")
    out("|---|" + "---|" * len(snaps))
    for thr in sorted(set(BUNCH_THRESHOLDS) | {int(req)}):
        cells = []
        for s in snaps:
            v = lvl[s][lvl[s].isfinite()]
            cells.append(f"{int(((v >= thr) & (v < 1.05 * thr)).sum())} / {int(((v >= thr) & (v < 1.10 * thr)).sum())}")
        out(f"| {thr:,}{' (requirement)' if thr == int(req) else ''} | " + " | ".join(cells) + " |")

    if POST_DEADLINE not in snaps:
        out(f"{POST_DEADLINE} is past the panel end; (b)-(c) skipped")
        return
    before, after = lvl[pre_end], lvl[POST_DEADLINE]
    ok = before.isfinite() & after.isfinite()
    below = ok & (before < req)
    out("")
    out(f"(b) Where the banks below {req:,.0f} at {pre_end[1]} {pre_end[0]} end up at "
        f"{POST_DEADLINE[1]} {POST_DEADLINE[0]} (n={int(below.sum())}). Floor: rank correlation start vs end "
        "near 0 and a much smaller spread at the end; catch-up: rank correlation > 0.")
    if int(below.sum()) >= 4:
        b, a = before[below], after[below]
        rho, p_rho = spearmanr(b.numpy(), a.numpy())
        cv = lambda v: (v.std() / v.mean()).item()  # noqa: E731
        out(f"Spearman(start, end) = {rho:.3f} (p {p_rho:.3f}); coefficient of variation {cv(b):.3f} -> {cv(a):.3f}; "
            f"end level median {a.median().item():,.0f}, range [{a.min().item():,.0f}, {a.max().item():,.0f}]")

    out("")
    out(f"(c) Raise vs shortfall, cross-section: raise = paid-up {POST_DEADLINE[1]} {POST_DEADLINE[0]} - "
        f"{pre_end[1]} {pre_end[0]}; shortfall = max(0, {req:,.0f} - paid-up at {pre_end[1]} {pre_end[0]}). "
        "A floor predicts slope near 1 and intercept near 0 (raise exactly to the floor, plus whatever "
        f"overshoot); the slope's p is a permutation test ({N_PERM} draws), not a t-test.")
    out("| bank | paid-up start | shortfall | raise | raise / shortfall | end - requirement |")
    out("|---|---|---|---|---|---|")
    short = (req - before).clamp(min=0)
    raise_ = after - before
    for i, bank in enumerate(inp.bfi_names):
        rs = (raise_[i] / short[i]).item() if short[i] > 0 else math.nan
        out(f"| {bank} | {_num(before[i].item(), ',.0f')} | {_num(short[i].item(), ',.0f')} | "
            f"{_num(raise_[i].item(), ',.0f')} | {_num(rs, '.2f')} | {_num((after[i] - req).item(), '+,.0f')} |")
    x, y = short[ok], raise_[ok]
    if len(x) >= 4 and x.std() > 0:
        def slope(yv):
            """Return the OLS slope of yv on the shortfall x (both mean-centred)."""
            xc = x - x.mean()
            return ((xc * (yv - yv.mean())).sum() / (xc * xc).sum()).item()
        b1 = slope(y)
        b0 = (y.mean() - b1 * x.mean()).item()
        resid = y - b0 - b1 * x
        r2 = 1 - (resid.pow(2).sum() / (y - y.mean()).pow(2).sum()).item()
        g = torch.Generator().manual_seed(cfg.bootstrap_seed)
        perm = [slope(y[torch.randperm(len(y), generator=g)]) for _ in range(N_PERM)]
        p_perm = (1 + sum(abs(s) >= abs(b1) for s in perm)) / (N_PERM + 1)
        out(f"OLS (n={len(x)}): raise = {b0:,.0f} + {b1:.3f} x shortfall, R^2 {r2:.3f}, permutation p (slope) {p_perm:.4f}")

    out("")
    out("(d) Timing: paid-up increases by fiscal year (adjacent bulletin months only; bonus-share "
        "issues are annual, so every year has some). A floor concentrates them before and just after "
        "the deadline.")
    out("| fiscal year | banks raising | total increase (NPR mn) |")
    out("|---|---|---|")
    by_fy: dict[int, list] = {}
    for i in range(inp.n):
        for t in range(1, len(periods)):
            if _months_between(periods[t - 1], periods[t]) != 1:
                continue
            d = (pu_ext[i, t] - pu_ext[i, t - 1]).item()
            if math.isfinite(d) and d > 0:
                by_fy.setdefault(fiscal_year(periods[t]), []).append((i, d))
    for fy in sorted(by_fy):
        ev = by_fy[fy]
        out(f"| {fy}/{(fy + 1) % 100:02d} | {len({i for i, _ in ev})} | {sum(d for _, d in ev):,.0f} |")
    out(f"Panel {periods[0][1]} {periods[0][0]} .. {periods[-1][1]} {periods[-1][0]}, so the first and last "
        "fiscal years are partial.")


# Trends for the paid-up dose: the dose itself and log size_pre (dose_pu is mostly inverse size).
TREND_PL = ["nim", "net_profit_loss", "loan_loss_provision"]
SMOOTHNESS_MULTIPLES = (0.25, 0.5, 1.0)  # M as a multiple of |estimated pre slope|


def _bin_span(periods, bins, b) -> str:
    cols = [i for i, bt in enumerate(bins) if bt == b]
    return f"{periods[cols[0]][1][:3]} {periods[cols[0]][0]} - {periods[cols[-1]][1][:3]} {periods[cols[-1]][0]}"


def paid_up_trend_series(cfg, inp, req):
    """Build the total-effect cells paid_up_trend and plot_paid_up_figures fit.

    Returns (pl, dose_pu, stocks, inp_pu, panel, timing, series), stocks first
    then TREND_PL. Raises ValueError like the loaders.
    """
    pl = [f for f in TREND_PL if f in OUTCOME_FIELDS]
    dose, _, _, _, _, _, stocks = _paid_up_setup(cfg, inp, req)
    stocks = {k: v for k, v in stocks.items() if k in ("paid_up / size_pre", "log size")}
    inp_pu = replace(inp, dose_pre=dose)
    panel, timing, series = total_effect_series(cfg, inp_pu, stock_outcomes=stocks)
    series |= total_effect_series(cfg, inp_pu, fields=pl)[2]
    return pl, dose, stocks, inp_pu, panel, timing, series


def no_trend_event_study(y, dose, inp, periods, timing, n_boot: int, seed: int):
    """Fit the paid-up dose event study by fiscal-year bin, no trend, reference bin -1.

    The 'no trend' column of paid_up_trend, and the figure. Returns
    (bins, order, fit_twfe result).
    """
    t = len(periods)
    bins = event_bins(periods, timing)
    order = list(dict.fromkeys(bins))
    dose_b = dose.unsqueeze(1).expand(inp.n, t)
    terms = ({f"dose[{b}]": dose_b * bin_indicator(bins, b) for b in order if b != -1}
             | pre_x_post_controls(inp.controls_pre, timing.post(t).unsqueeze(0)))
    fit = fit_twfe(y, terms, torch.ones(inp.n, t), bootstrap=[k for k in terms if k.startswith("dose[")],
                   n_boot=n_boot, seed=seed)
    return bins, order, fit


def check_paid_up_trend(out, cfg, inp, args) -> None:
    """Extrapolate a pre-period trend and bound its smoothness for the paid-up dose (item 1)."""
    out("## Paid-up dose: pre-period trend extrapolation and a smoothness bound (Sec 5b item 1)")
    req = cfg.paid_up_capital_requirement
    if req is None:
        out("paid_up_capital_requirement is null in the config: nothing to test.")
        return
    try:
        pl, dose, stocks, inp_pu, panel, timing, series = paid_up_trend_series(cfg, inp, req)
    except ValueError as e:
        out(f"not computed: {e}")
        return
    out("whole-panel trend (paid_up_threats) = one dose slope over every quarter, so a post effect "
        "that ramps up also tilts the slope. Pre-only trend = the dose x trend and log size_pre x "
        "trend slopes are identified from the pre quarters (Baisakh 2070 - Mangsir 2072) only: the "
        "transition and post periods get their own level and slope, and dose_post is the deviation "
        "from the extrapolated pre line at the mean post date. Same cells, controls and FE as the "
        "headline total effect; unweighted.")
    periods, t = panel.periods, len(panel.periods)
    bins = event_bins(periods, timing)
    order = list(dict.fromkeys(bins))
    nonpre = list(dict.fromkeys(bins[timing.pre_end_idx + 1:]))
    years = years_since_pre_end(periods, timing)
    post = timing.post(t).unsqueeze(0)
    ind = lambda b: bin_indicator(bins, b)  # noqa: E731
    dose_b = dose.unsqueeze(1).expand(inp.n, t)
    ls_b = inp.controls_pre["log_size"].unsqueeze(1).expand(inp.n, t)
    controls = pre_x_post_controls(inp.controls_pre, post)
    # log size_pre's post level is saturated by size[b], so its pooled x post control goes. Each
    # later bin also gets a within-bin slope (centred), so no post quarter feeds the trend slopes;
    # a one-quarter bin's slope column is all zero and fit_twfe drops it.
    def centred(b):
        """Return bin b's indicator times its within-bin mean-centred year."""
        cols = [i for i, bt in enumerate(bins) if bt == b]
        return ind(b) * (years - years[0, cols].mean())
    pre_trend = ({f"dose[{b}]": dose_b * ind(b) for b in nonpre} | {f"size[{b}]": ls_b * ind(b) for b in nonpre}
                 | {f"dose[{b}]_slope": dose_b * centred(b) for b in nonpre}
                 | {f"size[{b}]_slope": ls_b * centred(b) for b in nonpre}
                 | {"dose_x_trend": dose_b * years, "size_x_trend": ls_b * years}
                 | {k: v for k, v in controls.items() if k != "log_size_pre_x_post"})
    pre_cols = list(range(timing.pre_end_idx + 1))

    out("")
    out("### Event study by fiscal-year bin, dose_pu coefficient: beta (SE, wild p)")
    out("No trend: reference bin -1. Pre-only trend: the pre bins are the trend fit (no dummies); "
        "each later bin is that year's deviation from the extrapolated pre line at the bin's mid-date.")
    for name, y in series.items():
        if sum(int(y[:, i].isfinite().sum()) for i in pre_cols) == 0:
            out(f"{name}: no finite pre cells, skipped")
            continue
        _, _, f0 = no_trend_event_study(y, dose, inp, periods, timing, args.n_boot, cfg.bootstrap_seed)
        f1 = fit_twfe(y, pre_trend, torch.ones(inp.n, t),
                      bootstrap=[k for k in pre_trend if k.startswith("dose[") and not k.endswith("_slope")],
                      n_boot=args.n_boot, seed=cfg.bootstrap_seed)
        cell = lambda f, nm: (f"{_num(f.beta[nm], '.3g')} ({_num(f.se[nm], '.2g')}, p {_p(f.p_wild.get(nm))})"  # noqa: E731
                              if nm in f.beta else "-")
        out("")
        out(f"{name} (n_obs {f0.n_obs} / {f1.n_obs}):")
        out("| bin | months | no trend | pre-only trend |")
        out("|---|---|---|---|")
        for b in order:
            nm = f"dose[{b}]"
            left = "0 (ref)" if b == -1 else cell(f0, nm)
            right = "trend fit" if b not in nonpre else cell(f1, nm)
            out(f"| {b} | {_bin_span(periods, bins, b)} | {left} | {right} |")
        out(f"pre slope, dose x trend: {_num(f1.beta['dose_x_trend'], '.3g')} per year "
            f"(SE {_num(f1.se['dose_x_trend'], '.2g')}); dropped: {', '.join(f1.dropped) or 'none'}")

    out("")
    out("### Pooled total effect: no trend vs whole-panel trend vs pre-only trend")
    trends = {"dose": dose, "log_size_pre": inp.controls_pre["log_size"]}
    kw = dict(n_boot=args.n_boot, seed=cfg.bootstrap_seed, trend_controls=trends)
    try:
        fits = {}
        for src, extra in (("stocks", {"stock_outcomes": stocks}), ("pl", {"fields": pl})):
            fits[src] = {
                "no trend": fit_total_effect(cfg, inp_pu, n_boot=args.n_boot, seed=cfg.bootstrap_seed, **extra)[1],
                "whole-panel trend": fit_total_effect(cfg, inp_pu, trend_fit="panel", **kw, **extra)[1],
                "pre-only trend": fit_total_effect(cfg, inp_pu, trend_fit="pre", **kw, **extra)[1]}
    except ValueError as e:
        out(f"not computed: {e}")
        return
    merged = {v: fits["stocks"][v] | fits["pl"][v] for v in fits["stocks"]}
    rows = [(f"{name}, {v}", merged[v][name]) for name in series for v in merged]
    _total_effect_table(out, rows)
    out("lead = n/a under the pre-only trend by construction (the trend replaces the lead test).")

    out("")
    out("### Smoothness bound on the pre-only-trend estimate")
    out("Suppose the dose slope changes by M per year right at the pre-window end (a trend break the "
        "pre data can't see). Post has its own level and slope, so dose_post, evaluated at the mean "
        "post date tau, is biased by exactly M x tau. CI(M) = beta +/- (c x SE + M x tau), c = "
        "t(0.975, G-1), CR1 SE. Breakdown M* = the smallest M whose CI includes 0, also shown as a "
        "multiple of the estimated pre slope |s|. A simplified version of Rambachan-Roth's smoothness "
        "restriction (one slope break, CR1), not their optimal FLCI.")
    head = " | ".join(f"CI at M = {m:g}|s|" for m in SMOOTHNESS_MULTIPLES)
    out(f"| outcome | beta | SE | pre slope s (SE) | tau (yr) | CI at M = 0 | {head} | M* | M* / |s| |")
    out("|---|---|---|---|---|---|" + "---|" * len(SMOOTHNESS_MULTIPLES) + "---|---|")
    for name in series:
        te = merged["pre-only trend"][name]
        s = te.trend.get("dose", {})
        b, se, tau = te.beta_dose, te.se_dose, te.post_centre
        s_b, s_se = s.get("beta", math.nan), s.get("se", math.nan)
        if not all(math.isfinite(v) for v in (b, se, tau, s_b)) or te.n_clusters < 2:
            out(f"| {name} | {_num(b, '.4g')} | {_num(se, '.3g')} | n/a | | | | | | | |")
            continue
        c = t_dist.ppf(0.975, te.n_clusters - 1)
        ci = lambda m: f"[{b - c * se - m * tau:.4g}, {b + c * se + m * tau:.4g}]"  # noqa: E731
        m_star = max(0.0, abs(b) - c * se) / tau
        rel = m_star / abs(s_b) if s_b != 0 else math.nan
        out(f"| {name} | {b:.4g} | {se:.3g} | {s_b:.3g} ({_num(s_se, '.2g')}) | {tau:.2f} | {ci(0)} | "
            + " | ".join(ci(m * abs(s_b)) for m in SMOOTHNESS_MULTIPLES) + f" | {m_star:.3g} | {_num(rel, '.2f')} |")
    out("M* = 0: the pre-only-trend CI already includes 0. M* / |s| > 1: only a slope change larger "
        "than the whole pre-period dose slope would explain the estimate away.")


# --- item 2 follow-up --------------------------------------------------------
# Pairs whose names share no prefix, so the automatic match below misses them (survivorship run
# 2026-11-03). Data-based hints only, not facts about the banks.
LABEL_PAIR_HINTS = [("Civil", "CBL")]


def _name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def check_label_continuity(out, cfg, inp, args) -> None:
    """Check bulletin label splits and exits in the data only (item 2 follow-up)."""
    out("## Bank label splits and exits (item 2 follow-up): continuity in the data only")
    out("Checks whether two labels look like one bank renamed in the bulletin (never both present, "
        "short gap, size and capital fund carried over), and which banks' size jumps when another "
        "exits. This is NOT a verification of mergers or renames: that needs NRB's records.")
    full = filter_commercial_banks(load_panel(cfg.csv_path))
    names, periods = list(full.bfi_names), full.periods
    present = torch.isfinite(full.X).all(dim=-1)
    size = bank_size(full).to(torch.float64)
    cap = full.K.to(torch.float64)
    logs = positive_size(size).log()
    cols = [torch.nonzero(present[i]).flatten().tolist() for i in range(len(names))]
    changes = [abs(logs[i, b] - logs[i, a]).item() for i, c in enumerate(cols) for a, b in zip(c, c[1:])
               if math.isfinite((logs[i, b] - logs[i, a]).item())]
    if not changes:
        out("no bank has two consecutive complete months; nothing to compare")
        return
    ref = torch.tensor(changes)
    p99 = torch.quantile(ref, 0.99).item()
    out(f"{len(names)} commercial-bank labels, {periods[0][1]} {periods[0][0]} .. {periods[-1][1]} {periods[-1][0]}. "
        f"Reference: |log size change| between a bank's consecutive complete months, median "
        f"{ref.median().item():.3f}, 99th pct {p99:.3f} (n={len(changes)}).")
    fmt = lambda t_idx: f"{periods[t_idx][1]} {periods[t_idx][0]}"  # noqa: E731

    keys = {nm: _name_key(nm) for nm in names}
    pairs = [(a, b) for a, b in combinations(names, 2)
             if keys[a] != keys[b] and min(len(keys[a]), len(keys[b])) >= 3
             and (keys[a].startswith(keys[b]) or keys[b].startswith(keys[a]))]
    pairs += [(a, b) for a, b in combinations(names, 2) if keys[a] == keys[b]]
    lower = {nm.lower(): nm for nm in names}
    for a, b in LABEL_PAIR_HINTS:
        if a.lower() in lower and b.lower() in lower:
            pairs.append((lower[a.lower()], lower[b.lower()]))
        else:
            out(f"hint pair {a} / {b}: not both in the panel")
    out("")
    out("| old label | new label | in estimation panel | old last | new first | overlap months | gap (panel cols) | size new/old | |log ratio| vs p99 | capital fund new/old |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    paired = set()
    for a, b in pairs:
        ia, ib = names.index(a), names.index(b)
        if not cols[ia] or not cols[ib]:
            continue
        old, new = (ia, ib) if cols[ia][0] <= cols[ib][0] else (ib, ia)
        paired |= {old, new}
        overlap = int((present[old] & present[new]).sum())
        lo, fn = cols[old][-1], cols[new][0]
        s_ratio = (size[new, fn] / size[old, lo]).item()
        c_ratio = (cap[new, fn] / cap[old, lo]).item() if cap[old, lo] != 0 else math.nan
        lr = abs(math.log(s_ratio)) if s_ratio > 0 and math.isfinite(s_ratio) else math.nan
        in_panel = "/".join("yes" if names[i] in inp.bfi_names else "no" for i in (old, new))
        out(f"| {names[old]} | {names[new]} | {in_panel} | {fmt(lo)} | {fmt(fn)} | {overlap} | {fn - lo} | "
            f"{_num(s_ratio, '.3f')} | {_num(lr, '.3f')} {'<=' if lr <= p99 else '>'} {p99:.3f} | {_num(c_ratio, '.3f')} |")
    out("Consistent with a relabel: 0 overlap months, gap 1, size and capital-fund ratios near 1.")

    last_col = len(periods) - 1
    exits = [i for i, c in enumerate(cols) if c and c[-1] < last_col and i not in paired]
    entrants = [i for i, c in enumerate(cols) if c and c[0] > 0 and i not in paired]
    out("")
    out("Exits (last complete month before the panel end, not part of a label pair) and the banks whose "
        "size jumps most at the next month (|log change| > p99), as candidate absorbers:")
    out("| bank | in estimation panel | last month | size then | next month | size jumps > p99 (bank: +NPR mn) |")
    out("|---|---|---|---|---|---|")
    for i in exits:
        lo = cols[i][-1]
        nxt = lo + 1
        jumps = []
        for j in range(len(names)):
            if j == i or not (present[j, lo] and present[j, nxt]):
                continue
            ch = (logs[j, nxt] - logs[j, lo]).item()
            if math.isfinite(ch) and abs(ch) > p99:
                jumps.append((names[j], (size[j, nxt] - size[j, lo]).item()))
        jumps.sort(key=lambda kv: -abs(kv[1]))
        out(f"| {names[i]} | {'yes' if names[i] in inp.bfi_names else 'no'} | {fmt(lo)} | "
            f"{_num(size[i, lo].item(), ',.0f')} | {fmt(nxt)} | "
            + (", ".join(f"{nm}: {d:+,.0f}" for nm, d in jumps[:4]) or "none") + " |")
    out("")
    out("Entrants (first complete month after the panel start, not part of a label pair):")
    out("| bank | first month | size then |")
    out("|---|---|---|")
    for i in entrants:
        fn = cols[i][0]
        out(f"| {names[i]} | {fmt(fn)} | {_num(size[i, fn].item(), ',.0f')} |")
    out("Corrupted bulletins (Chaitra 2075 onward, Sec 5a) also move size, so a jump there is weak evidence.")


# --- item 5 ------------------------------------------------------------------
def _holm(ps: list[float]) -> list[float]:
    """Return Holm-Bonferroni step-down adjusted p-values.

    With the p-values sorted ascending p_(1) <= ... <= p_(m), the adjusted value
    at rank k is max_{j <= k} min(1, (m - j + 1) * p_(j)) (monotone, capped at 1).
    """
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def _bh(ps: list[float]) -> list[float]:
    """Return Benjamini-Hochberg (FDR) step-up adjusted p-values.

    With the p-values sorted ascending p_(1) <= ... <= p_(m), the adjusted value
    at rank r is min_{s >= r} (p_(s) * m / s) (monotone non-decreasing, capped at 1).
    """
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    adj, running = [0.0] * m, 1.0
    for k, i in enumerate(order):
        running = min(running, ps[i] * m / (m - k))
        adj[i] = running
    return adj


def check_multiple_tests(out, cfg, inp, args, config_path: Path) -> None:
    """Apply Holm and BH multiple-testing corrections over the saved run JSON (item 5)."""
    out("## Multiple testing (item 5), over the saved run JSON")
    path = RESULTS_DIR / f"{config_path.stem}_seed{inp.artifact.seed}.json"
    if not path.exists():
        out(f"{path} not found -- run run_causal_experiment for this config first")
        return
    run = json.loads(path.read_text())
    families: dict[str, list[tuple[str, float]]] = {"pooled DiD (W_hat + RAS)": [], "event study (W_hat)": [],
                                                    "event study joint, all post bins = 0 (W_hat)": [],
                                                    "headline total effect (dose-only)": []}
    joint_rows = []
    for name, res in run["outcomes"].items():
        for key, net in [("did", "W_hat"), ("did_ras_baseline", "RAS")]:
            for coef in ["dose", "exposure"]:
                families["pooled DiD (W_hat + RAS)"].append((f"{name} {net} {coef}", res[key][f"p_{coef}"]))
        es = res["event_study"]
        for coef in ["dose", "exposure"]:
            for b, v in es.get(coef, {}).items():
                families["event study (W_hat)"].append((f"{name} {coef} bin {b}", v["p"]))
        for fam, j in es.get("joint", {}).items():
            joint_rows.append((name, fam, j))
            if fam.endswith("_post"):
                families["event study joint, all post bins = 0 (W_hat)"].append((f"{name} {fam}", j["p"]))
        te = res.get("total_effect")
        if te:
            families["headline total effect (dose-only)"].append((f"{name} dose", te["p_dose"]))
    out("| family | tests | p<0.05 | expected under H0 | Holm rejections | BH (FDR 5%) rejections | smallest Holm p |")
    out("|---|---|---|---|---|---|---|")
    for fam, items in families.items():
        items = [(lbl, p) for lbl, p in items if p is not None and math.isfinite(p)]
        if not items:
            out(f"| {fam} | 0 | - | - | - | - | - |")
            continue
        ps = [p for _, p in items]
        holm, bh = _holm(ps), _bh(ps)
        i_min = min(range(len(ps)), key=lambda i: holm[i])
        out(f"| {fam} | {len(ps)} | {sum(p < 0.05 for p in ps)} | {0.05 * len(ps):.1f} | "
            f"{sum(h < 0.05 for h in holm)} | {sum(q < 0.05 for q in bh)} | {holm[i_min]:.3f} ({items[i_min][0]}) |")
    out("")
    out("Joint wild-bootstrap Wald tests from the event study (estimator.fit_twfe `joint`; H0: every "
        "bin in the family = 0; the transition bin is in no family):")
    if not joint_rows:
        out("none in this JSON -- re-run run_causal_experiment with the current estimator")
        return
    out("| outcome | family | bins tested | Wald (CR1) | wild p |")
    out("|---|---|---|---|---|")
    for name, fam, j in joint_rows:
        bins = ", ".join(nm.split("[")[1].rstrip("]") for nm in j["names"])
        out(f"| {name} | {fam} | {j['q']} ({bins}) | {_num(j['wald'], '.3g')} | {_p(j['p'])} |")


def main() -> None:
    """Run the selected reviewer checks for each config and save the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append",
                        help="experiment config(s); default: main.yaml")
    parser.add_argument("--only", nargs="+", choices=CHECKS, default=CHECKS)
    parser.add_argument("--n-boot", type=int, default=None,
                        help="wild-bootstrap draws; default: the config's n_bootstrap (as the main run)")
    parser.add_argument("--quarterly", action="store_true",
                        help="replace the monthly outcomes with the quarter-end-only construction for "
                             "every check (the 'main' columns then mean quarterly, main weights)")
    args = parser.parse_args()
    n_boot_arg = args.n_boot

    for path in args.config or DEFAULT_CONFIGS:
        cfg = load_causal_experiment_config(path)
        args.n_boot = cfg.n_bootstrap if n_boot_arg is None else n_boot_arg
        inp = build_inputs(cfg, path)
        if args.quarterly:
            if not cfg.pnl_ytd_cumulative:
                raise SystemExit(f"{path}: --quarterly needs pnl_ytd_cumulative: true")
            inp.y = _quarterly_y(inp)
        _REF.clear()
        _PERIOD_DESIGNS.clear()
        out = print
        with captured_output() as buf:
            out(f"# Reviewer checks: {path.stem} (N={inp.n} x T={inp.t}, {inp.periods[0]} .. {inp.periods[-1]}, "
                f"n_boot={args.n_boot}, outcomes {'quarterly' if args.quarterly else cfg.outcome_construction}, "
                f"main = {cfg.weighting} TWFE)")
            for check in args.only:
                out("")
                fn = globals()[f"check_{check}"]
                if check == "multiple_tests":
                    fn(out, cfg, inp, args, path)
                else:
                    fn(out, cfg, inp, args)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = RESULTS_DIR / f"reviewer_checks_{path.stem}{'_quarterly' if args.quarterly else ''}.md"
        dest.write_text(buf.getvalue())
        print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
