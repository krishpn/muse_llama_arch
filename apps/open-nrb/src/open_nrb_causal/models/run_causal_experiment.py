#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
# ]
# ///
"""Run the Paper 2 causal experiment end to end from a config.

Paper 2 orchestration script: consumes open_nrb's already-saved real-data
artifact (no retraining, no edits to open_nrb), builds the frozen-pre-mandate
dose x post design (mandate_design.py), and per Sec 4b outcome fits:
  - the headline beta: the dose-only total effect on quarterly outcomes from Baisakh 2070
    (total_effect.py; quarterly configs only), with a pre-trend lead;
  - the pooled treatment-intensity DiD (Eq. A', the source of gamma), under W_hat and
    (robustness) the RAS baseline network;
  - the fiscal-year event study (pre bins = parallel-trends check);
  - the secondary pooled-window RDD around the capital-floor threshold.
Inference: bank-clustered CR1 SEs + WCR wild-cluster-bootstrap p-values.

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.models.run_causal_experiment --config src/open_nrb_causal/experiments/configs/main.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

try:
    from .artifact_io import load_artifact
    from .estimator import fit_event_study, fit_rdd
    from .experiment_config import CONFIG_DIR, load_causal_experiment_config
    from .mandate_design import (
        apply_weighting,
        bank_size,
        did_design,
        event_bins,
        fiscal_year,
        freeze_pre_mandate,
        mandate_timing,
        outcome_ratio,
        positive_size,
        ytd_cumulation_diagnostic,
    )
    from .panel_alignment import align_to_artifact, check_marginals, load_commercial_panel
    from .robustness import compare_against_ras_baseline
    from .total_effect import TOTAL_EFFECT_START, fit_total_effect
    from .variables import (
        OUTCOME_FIELDS,
        compute_dose,
        controls_from_panel,
        load_outcome_panel,
        zero_fill_blanks,
    )
except ImportError:
    from artifact_io import load_artifact  # type: ignore[no-redef]
    from estimator import fit_event_study, fit_rdd  # type: ignore[no-redef]
    from experiment_config import CONFIG_DIR, load_causal_experiment_config  # type: ignore[no-redef]
    from mandate_design import (  # type: ignore[no-redef]
        apply_weighting,
        bank_size,
        did_design,
        event_bins,
        fiscal_year,
        freeze_pre_mandate,
        mandate_timing,
        outcome_ratio,
        positive_size,
        ytd_cumulation_diagnostic,
    )
    from panel_alignment import align_to_artifact, check_marginals, load_commercial_panel  # type: ignore[no-redef]
    from robustness import compare_against_ras_baseline  # type: ignore[no-redef]
    from total_effect import TOTAL_EFFECT_START, fit_total_effect  # type: ignore[no-redef]
    from variables import (  # type: ignore[no-redef]
        OUTCOME_FIELDS,
        compute_dose,
        controls_from_panel,
        load_outcome_panel,
        zero_fill_blanks,
    )

import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = CONFIG_DIR / "baseline.yaml"


def _fmt(beta: float, se: float, p: float | None) -> str:
    p_str = "" if p is None else f" p={p:.3f}"
    return f"{beta:.4g} ({se:.3g}){p_str}"


@dataclass
class CausalInputs:
    """Bundle everything the estimators consume, built once from a config.

    Shared by main() and the Sec 5b robustness scripts (power_mde.py, ...) so
    they all use exactly the same sample.
    """
    artifact: object
    bfi_names: list[str]
    periods: list
    n: int
    t: int
    timing: object
    bins: list
    size: torch.Tensor
    n_bad_size: int
    dose: torch.Tensor
    dose_pre: torch.Tensor
    controls_pre: dict[str, torch.Tensor]
    W_main: torch.Tensor  # (T, N, N), sliced at post_window_end
    W_ras: torch.Tensor
    design_main: object  # weights per cfg.weighting (the estimator every script reports)
    design_ras: object
    design_main_ht: object  # same design, always the stabilised HT weights (weight diagnostics)
    design_ras_ht: object
    y: dict[str, torch.Tensor]  # outcome / size, per cfg.outcome_construction, (N, T)
    y_raw: dict[str, torch.Tensor]  # outcome in NPR mn as extracted (no de-cumulation, no scaling)
    ytd_ratio: float
    rdd_idx: list[int]
    panel: object = None  # aligned Panel before the post_window_end slice
    outcomes_raw: object = None  # variables.OutcomePanel on that panel, before zero_fill_blanks


def build_inputs(cfg, config_path: Path) -> CausalInputs:
    """Build the shared CausalInputs for one config: panel, design, weights, outcomes.

    Loads the saved artifact, aligns and marginal-checks the panel, slices to
    post_window_end, freezes the pre-mandate dose/controls, builds the W_hat and
    RAS designs under cfg.weighting, and constructs the scaled outcomes. Raises
    if capital_floor is unset or W_hat is NaN inside the window.
    """
    if cfg.capital_floor is None:
        raise ValueError(
            f"{config_path}: capital_floor is still the unset placeholder (null) -- supply "
            "the real NRB capital-floor figure before running this experiment."
        )

    artifact = load_artifact(cfg.artifact_path)
    panel = load_commercial_panel(cfg.csv_path)
    panel = align_to_artifact(panel, artifact)
    check_marginals(panel, artifact)
    periods = panel.periods
    n, t = len(panel.bfi_names), len(periods)

    timing = mandate_timing(periods, cfg.pre_window_end, cfg.effective_period)
    # post_window_end truncates the panel's tail (clean_window.yaml: the last bulletin before
    # the extraction corruption, Sec 5a). Slicing from the end keeps every timing index valid.
    end = t
    if cfg.post_window_end is not None:
        if cfg.post_window_end not in periods:
            raise ValueError(f"post_window_end {cfg.post_window_end} is not in the panel's periods")
        end = periods.index(cfg.post_window_end) + 1
        if end <= timing.effective_idx + 1:
            raise ValueError(f"post_window_end {cfg.post_window_end} leaves no post-mandate cells")
    periods, t = periods[:end], end
    W_main, W_ras = artifact.W_hat[:end], artifact.W_baseline[:end]
    # Paper 1 NaN-fills W_hat (and zeroes W_baseline) on non-finite / degenerate-marginal months.
    # A NaN exposure cell poisons the unmasked GPS lstsq, so every weight (and, unweighted, every
    # cell) would drop: refuse instead of fitting an empty sample.
    nan_months = sorted({t for t in artifact.non_finite_periods + artifact.degenerate_periods if t < end}
                        | {t for t in range(end) if not torch.isfinite(W_main[t]).all()})
    if nan_months:
        raise ValueError(f"W_hat is NaN on {len(nan_months)} month(s) inside the window "
                         f"({', '.join(f'{m} {y}' for y, m in (periods[t] for t in nan_months))}) -- "
                         "Paper 1 marked them non-finite/degenerate; move post_window_end before them")
    bins = event_bins(periods, timing)
    raw_size = bank_size(panel)[:, :end]
    size = positive_size(raw_size)
    n_bad_size = int((~(raw_size > 0)).sum())

    dose = compute_dose(panel, cfg.capital_floor)[:, :end]
    dose_pre = freeze_pre_mandate(dose, timing)
    controls_pre = {name: freeze_pre_mandate(c, timing) for name, c in controls_from_panel(panel).items()}
    controls_pre["log_size"] = freeze_pre_mandate(size.log(), timing)

    design_main_ht = did_design(W_main, dose_pre, controls_pre, timing)
    design_ras_ht = did_design(W_ras, dose_pre, controls_pre, timing)
    design_main = apply_weighting(design_main_ht, cfg.weighting)
    design_ras = apply_weighting(design_ras_ht, cfg.weighting)

    outcomes_raw = outcomes = load_outcome_panel(cfg.pnl_csv_path, panel)
    unknown = [f for f in cfg.blank_as_zero if f not in OUTCOME_FIELDS]
    if unknown:
        raise ValueError(f"{config_path}: blank_as_zero names unknown outcome fields {unknown}; "
                         f"expected a subset of {list(OUTCOME_FIELDS)}")
    if cfg.blank_as_zero:
        outcomes = zero_fill_blanks(outcomes, cfg.blank_as_zero)  # item 11b, before any differencing
    ytd_ratio = ytd_cumulation_diagnostic(outcomes.nim[:, :end], periods)
    y_all: dict[str, torch.Tensor] = {}
    y_raw: dict[str, torch.Tensor] = {}
    for name in OUTCOME_FIELDS:
        y_raw[name] = getattr(outcomes, name)[:, :end]
        y_all[name] = outcome_ratio(y_raw[name], periods, size, cfg.outcome_construction == "quarterly",
                                    cfg.pnl_ytd_cumulative)

    rdd_start, rdd_end = cfg.rdd_window
    for name, p in [("rdd_window start", rdd_start), ("rdd_window end", rdd_end)]:
        if p not in periods:
            raise ValueError(f"{name} {p} is not in the panel's periods")
    rdd_idx = list(range(periods.index(rdd_start), periods.index(rdd_end) + 1))

    return CausalInputs(artifact=artifact, bfi_names=list(panel.bfi_names), periods=periods, n=n, t=t,
                        timing=timing, bins=bins, size=size, n_bad_size=n_bad_size, dose=dose,
                        dose_pre=dose_pre, controls_pre=controls_pre, W_main=W_main, W_ras=W_ras,
                        design_main=design_main, design_ras=design_ras,
                        design_main_ht=design_main_ht, design_ras_ht=design_ras_ht, y=y_all, y_raw=y_raw,
                        ytd_ratio=ytd_ratio, rdd_idx=rdd_idx, panel=panel, outcomes_raw=outcomes_raw)


def main() -> None:
    """Run one experiment from --config: build inputs, fit every estimator, print and save JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_causal_experiment_config(args.config)
    inp = build_inputs(cfg, args.config)
    artifact, periods, n, t, timing, bins = inp.artifact, inp.periods, inp.n, inp.t, inp.timing, inp.bins
    dose, dose_pre, n_bad_size = inp.dose, inp.dose_pre, inp.n_bad_size
    design_main, design_ras = inp.design_main, inp.design_ras

    # --- diagnostics -------------------------------------------------------
    print(f"panel: N={n} banks x T={t} periods ({periods[0]} .. {periods[-1]}); "
          f"pre window {periods[0]}..{periods[timing.pre_end_idx]} "
          f"({timing.pre_end_idx + 1} months), transition to {periods[timing.effective_idx]}, "
          f"post {t - timing.effective_idx - 1} months"
          + ("" if cfg.post_window_end is None else f" (truncated at post_window_end {cfg.post_window_end})"))
    print(f"outcomes: {cfg.outcome_construction}; estimator weights: {cfg.weighting}; "
          f"blank read as 0 (other P&L reported): {list(cfg.blank_as_zero) or 'none'}")
    if n_bad_size:
        print(f"size <= 0 or missing on {n_bad_size}/{n * t} cells -> set NaN (outcome/size masked there)")
    below_by_fy: Counter[int] = Counter()
    cells_by_fy: Counter[int] = Counter()
    for t_idx, p in enumerate(periods):
        fy = fiscal_year(p)
        below_by_fy[fy] += int((dose[:, t_idx] > 0).sum())
        cells_by_fy[fy] += n
    print("cells below the floor by fiscal year: "
          + ", ".join(f"FY{fy}: {below_by_fy[fy]}/{cells_by_fy[fy]}" for fy in sorted(cells_by_fy)))
    print(f"dose_pre: {int((dose_pre > 0).sum())}/{n} banks below the floor pre-mandate, "
          f"range [{dose_pre.min().item():.4f}, {dose_pre.max().item():.4f}]")
    post_mask = design_main.dose_x_post != 0
    if post_mask.any():
        corr = torch.corrcoef(torch.stack([design_main.dose_x_post[post_mask],
                                           design_main.exposure_x_post[post_mask]]))[0, 1].item()
        print(f"corr(dose_pre x post, exposure x post) over post cells: {corr:.3f}")

    ytd_ratio, rdd_idx = inp.ytd_ratio, inp.rdd_idx
    print(f"YTD diagnostic (median |NIM Shrawan| / |NIM prior Asar|): {ytd_ratio:.3f} "
          f"(~0.08 => fiscal-YTD cumulative, ~1 => monthly); "
          f"pnl_ytd_cumulative={cfg.pnl_ytd_cumulative}")

    # --- estimation --------------------------------------------------------
    # Headline beta = dose-only total effect from Baisakh 2070 (total_effect.py, user decision
    # 2026-11-03); quarterly configs only. Eq. A' below supplies gamma.
    total_panel, total = None, {}
    total_note = ""
    if cfg.outcome_construction == "quarterly":
        try:
            total_panel, total = fit_total_effect(cfg, inp, n_boot=cfg.n_bootstrap, seed=cfg.bootstrap_seed)
        except ValueError as e:
            total_note = f"not computed: {e}"
    else:
        total_note = "not computed: needs outcome_construction quarterly"
    if total_panel is not None:
        print(f"total-effect panel: {total_panel.start} .. {total_panel.end}, T={total_panel.n_periods}; "
              "months absent from the extraction: "
              + (", ".join(f"{m} {y}" for y, m in total_panel.missing_months) or "none"))
    else:
        print(f"total effect (dose-only from {TOTAL_EFFECT_START[1]} {TOTAL_EFFECT_START[0]}): {total_note}")

    results: dict[str, dict] = {}
    for name in OUTCOME_FIELDS:
        y = inp.y[name]
        cmp = compare_against_ras_baseline(y, design_main, design_ras,
                                           n_boot=cfg.n_bootstrap, seed=cfg.bootstrap_seed)
        es = fit_event_study(y, dose_pre, design_main.exposure, bins, design_main.controls,
                             design_main.event_weights, n_boot=cfg.n_bootstrap, seed=cfg.bootstrap_seed)
        rdd = fit_rdd(y, dose, rdd_idx, cfg.rdd_bandwidth)

        results[name] = {
            "n_missing_outcome": int(y.isnan().sum()),
            "did": asdict(cmp.main),
            "did_ras_baseline": asdict(cmp.baseline),
            "beta_dose_diff": cmp.beta_dose_diff,
            "beta_exposure_diff": cmp.beta_exposure_diff,
            "event_study": asdict(es),
            "rdd": asdict(rdd),
        }
        if name in total:
            results[name]["total_effect"] = asdict(total[name])
        m, b = cmp.main, cmp.baseline
        print(f"\n[{name}] (outcome / size) missing={results[name]['n_missing_outcome']}/{n * t}, "
              f"DiD n_obs={m.n_obs}, clusters={m.n_clusters}")
        if name in total:
            te = total[name]
            print(f"  HEADLINE beta, dose-only total effect from {total_panel.start}: "
                  f"dose x post {_fmt(te.beta_dose, te.se_dose, te.p_dose)}  "
                  f"lead {_fmt(te.beta_lead, te.se_lead, te.p_lead)}  n_obs={te.n_obs}, clusters={te.n_clusters}"
                  + (f"  [{te.note}]" if te.note else ""))
        print(f"  DiD  W_hat: dose {_fmt(m.beta_dose, m.se_dose, m.p_dose)}  "
              f"exposure {_fmt(m.beta_exposure, m.se_exposure, m.p_exposure)}")
        print(f"  DiD  RAS  : dose {_fmt(b.beta_dose, b.se_dose, b.p_dose)}  "
              f"exposure {_fmt(b.beta_exposure, b.se_exposure, b.p_exposure)}")
        print("  event study dose: " + ", ".join(
            f"{k}: {_fmt(v['beta'], v['se'], v['p'])}" for k, v in es.dose.items()))
        print("  event study exposure: " + ", ".join(
            f"{k}: {_fmt(v['beta'], v['se'], v['p'])}" for k, v in es.exposure.items()))
        if es.joint:
            print("  event study joint Wald (H0: all bins in family = 0): " + ", ".join(
                f"{fam} q={j['q']} W={j['wald']:.3g}" + ("" if j["p"] is None else f" p={j['p']:.3f}")
                for fam, j in es.joint.items()))
        if es.dropped_bins:
            print(f"  event study bins dropped (no valid outcome cell): {es.dropped_bins}")
        if es.note:
            print(f"  event study not estimated: {es.note}")
        for label, d in [("DiD W_hat", m), ("DiD RAS", b), ("event study", es)]:
            dropped = d.dropped_regressors
            if dropped:
                print(f"  {label}: regressors unidentified on valid cells, reported NaN: {dropped}")
        for label, d in [("DiD W_hat", m), ("DiD RAS", b)]:
            if d.n_cells_nonfinite_regressor:
                print(f"  {label}: {d.n_cells_nonfinite_regressor} cells dropped for a non-finite regressor")
        disc = "nan" if math.isnan(rdd.discontinuity) else f"{rdd.discontinuity:.4g}"
        print(f"  RDD: discontinuity={disc}  cells L/R={rdd.n_left}/{rdd.n_right}  "
              f"banks L/R={rdd.n_banks_left}/{rdd.n_banks_right}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.config.stem}_seed{artifact.seed}.json"
    out_path.write_text(json.dumps({
        "config": str(args.config),
        "capital_floor": cfg.capital_floor,
        "n_banks": n,
        "periods": [list(p) for p in periods],
        "pre_window_end": list(cfg.pre_window_end),
        "effective_period": list(cfg.effective_period),
        "post_window_end": None if cfg.post_window_end is None else list(cfg.post_window_end),
        "event_bins": [str(b) for b in bins],
        "pnl_ytd_cumulative": cfg.pnl_ytd_cumulative,
        "outcome_construction": cfg.outcome_construction,
        "blank_as_zero": list(cfg.blank_as_zero),
        "weighting": cfg.weighting,
        "total_effect_panel": None if total_panel is None else asdict(total_panel),
        "total_effect_note": total_note,
        "ytd_diagnostic": ytd_ratio,
        "outcome_scaling": "divided by size (sum of top-level assets)",
        "dose_pre": dict(zip(inp.bfi_names, dose_pre.tolist())),
        "n_bootstrap": cfg.n_bootstrap,
        "outcomes": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
