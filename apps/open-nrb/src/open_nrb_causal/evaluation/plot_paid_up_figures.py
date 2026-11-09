#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
#     "scipy",
#     "matplotlib",
# ]
# ///
"""Plot Figures 2 and 3 of paper2_draft.md, the paid-up dose.

Same cells and fits as the reviewer checks they illustrate (reviewer_checks paid_up_path /
paid_up_threats (3) / paid_up_trend), imported from there, so the plotted numbers are the ones
in the text.

Writes to results/figures/:
  paid_up_compliance.png  Figure 2. (a) each bank's paid-up capital, Baisakh 2070 .. panel end,
                          with the requirement, the pre-window end and the Asar 2074 deadline;
                          (b) raise vs shortfall, pre-window end -> Asar 2075, 45-degree line;
                          (c) paid-up at the pre-window end vs Asar 2075, band [T, 1.05 T)
  paid_up_event_study.png Figure 3. NIM and net profit / size on the paid-up dose by fiscal-year
                          bin, no trend, reference bin -1 at 0, 95% CR1 CI; filled marker = WCR
                          wild-bootstrap p < 0.05
  paid_up_figures.md      the numbers behind both figures

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.evaluation.plot_paid_up_figures
  uv run python -m open_nrb_causal.evaluation.plot_paid_up_figures --config src/open_nrb_causal/experiments/configs/main.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

try:
    from ..models.experiment_config import CONFIG_DIR, load_causal_experiment_config
    from ..models.mandate_design import period_key
    from ..models.run_causal_experiment import RESULTS_DIR, build_inputs
    from .report_format import fmt_num as _num
    from .reviewer_checks import (
        PAID_UP_DEADLINE, POST_DEADLINE, _at, _bin_span, _months_between, _paid_up_setup, no_trend_event_study,
        paid_up_trend_series)
    from .run_event_study_report import OUTCOME_LABELS, Z, _sci
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from experiment_config import CONFIG_DIR, load_causal_experiment_config  # type: ignore[no-redef]
    from mandate_design import period_key  # type: ignore[no-redef]
    from run_causal_experiment import RESULTS_DIR, build_inputs  # type: ignore[no-redef]
    from report_format import fmt_num as _num  # type: ignore[no-redef]
    from reviewer_checks import (  # type: ignore[no-redef]
        PAID_UP_DEADLINE, POST_DEADLINE, _at, _bin_span, _months_between, _paid_up_setup, no_trend_event_study,
        paid_up_trend_series)
    from run_event_study_report import OUTCOME_LABELS, Z, _sci  # type: ignore[no-redef]

EVENT_STUDY_OUTCOMES = ["nim", "net_profit_loss"]
BUNCH_BAND = 1.05  # paid_up_threats (3a): [T, 1.05 T)


def _label(p) -> str:
    return f"{p[1]} {p[0]}"


def plot_compliance(cfg, inp, req: float, out: Path, md: list[str]) -> None:
    """Plot Figure 2 (paid-up compliance) and append its numbers to `md`."""
    _, ext, _, pu_ext, _, _, _ = _paid_up_setup(cfg, inp, req)
    periods = ext.periods
    pre_end = tuple(cfg.pre_window_end)
    if period_key(POST_DEADLINE) > period_key(periods[-1]):
        raise SystemExit(f"{_label(POST_DEADLINE)} is past the panel end {_label(periods[-1])}")
    # Calendar-month x axis, so the bulletin months with no rows show as gaps, not joined lines.
    x = [_months_between(periods[0], p) for p in periods]
    at = {p: x[periods.index(p)] for p in periods}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax = axes[0]
    cmap = plt.get_cmap("tab20")
    for i, bank in enumerate(inp.bfi_names):
        grid = [math.nan] * (x[-1] + 1)
        for t, xt in enumerate(x):
            grid[xt] = pu_ext[i, t].item()
        ax.plot(range(len(grid)), grid, color=cmap(i % 20), linewidth=0.9, label=bank)
    ax.axhline(req, color="black", linewidth=0.8, linestyle="--")
    ax.text(0, req, f" requirement {req:,.0f}", va="bottom", fontsize=7)
    for p, name in [(pre_end, "pre-window end"), (PAID_UP_DEADLINE, "deadline")]:
        if p in at:
            ax.axvline(at[p], color="grey", linewidth=0.8, linestyle=":")
            ax.text(at[p], ax.get_ylim()[1], f" {name}", rotation=90, va="top", fontsize=7, color="grey")
    ticks = [(xt, p) for xt, p in zip(x, periods) if p[1] == "Shrawan"]
    ax.set_xticks([xt for xt, _ in ticks])
    ax.set_xticklabels([f"FY{p[0]}/{str(p[0] + 1)[-2:]}" for _, p in ticks], fontsize=7, rotation=45)
    ax.set_ylabel("paid-up capital (NPR mn)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(a) Paid-up capital by bank", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="upper left", frameon=False)

    n = inp.n
    before = torch.tensor([_at(pu_ext, periods, i, pre_end) for i in range(n)], dtype=torch.float64)
    after = torch.tensor([_at(pu_ext, periods, i, POST_DEADLINE) for i in range(n)], dtype=torch.float64)
    short = (req - before).clamp(min=0)
    raise_ = after - before

    ax = axes[1]
    ax.scatter(short, raise_, s=14, color="C0")
    hi = max(float(short.max()), float(raise_.max())) * 1.05
    ax.plot([0, hi], [0, hi], color="black", linewidth=0.7, linestyle="--", label="raise = shortfall")
    for i, bank in enumerate(inp.bfi_names):
        ax.annotate(bank, (short[i].item(), raise_[i].item()), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"shortfall at {_label(pre_end)} (NPR mn)", fontsize=8)
    ax.set_ylabel(f"raise, {_label(pre_end)} -> {_label(POST_DEADLINE)}", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(b) Raise vs shortfall", fontsize=9)

    ax = axes[2]
    ax.axhspan(req, BUNCH_BAND * req, color="C2", alpha=0.2, label=f"[{req:,.0f}, {BUNCH_BAND * req:,.0f})")
    ax.axhline(req, color="black", linewidth=0.7, linestyle="--")
    ax.scatter(before, after, s=14, color="C0")
    for i, bank in enumerate(inp.bfi_names):
        ax.annotate(bank, (before[i].item(), after[i].item()), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"paid-up at {_label(pre_end)} (NPR mn)", fontsize=8)
    ax.set_ylabel(f"paid-up at {_label(POST_DEADLINE)} (NPR mn)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(c) Start vs end", fontsize=9)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    md += [f"## {out.name}", "",
           f"Requirement {req:,.0f} NPR mn; panel {_label(periods[0])} .. {_label(periods[-1])}; "
           "value at the last month with a finite cell at or before each date.", "",
           f"| bank | paid-up {_label(pre_end)} | shortfall | paid-up {_label(POST_DEADLINE)} | raise | in band |",
           "|---|---|---|---|---|---|"]
    for i, bank in enumerate(inp.bfi_names):
        band = req <= after[i].item() < BUNCH_BAND * req
        md.append(f"| {bank} | {_num(before[i].item(), ',.0f')} | {_num(short[i].item(), ',.0f')} | "
                  f"{_num(after[i].item(), ',.0f')} | {_num(raise_[i].item(), ',.0f')} | {'yes' if band else ''} |")
    md.append("")


def plot_event_study(cfg, inp, req: float, n_boot: int, out: Path, md: list[str]) -> None:
    """Plot Figure 3 (paid-up dose event study) and append its numbers to `md`."""
    _, dose, _, _, panel, timing, series = paid_up_trend_series(cfg, inp, req)
    names = [nm for nm in EVENT_STUDY_OUTCOMES if nm in series]
    periods = panel.periods
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 3.2), squeeze=False)
    md += [f"## {out.name}", "",
           f"Paid-up dose, no trend, reference bin -1; 95% CI = beta +/- {Z} x CR1 SE; wild p from "
           f"{n_boot} WCR draws (seed {cfg.bootstrap_seed}). Same fit as the 'no trend' column of "
           "reviewer_checks paid_up_trend.", ""]
    for ax, name in zip(axes[0], names):
        bins, order, fit = no_trend_event_study(series[name], dose, inp, periods, timing, n_boot, cfg.bootstrap_seed)
        md += [f"{OUTCOME_LABELS.get(name, name)} (n_obs {fit.n_obs}):", "",
               "| bin | months | beta | CR1 SE | wild p |", "|---|---|---|---|---|"]
        pts = []
        for k, b in enumerate(order):
            nm = f"dose[{b}]"
            if b == -1:
                pts.append((k, 0.0, 0.0, None))
                md.append(f"| {b} | {_bin_span(periods, bins, b)} | 0 (ref) | | |")
            elif nm in fit.beta and math.isfinite(fit.beta[nm]) and math.isfinite(fit.se[nm]):
                p = fit.p_wild.get(nm)
                pts.append((k, fit.beta[nm], fit.se[nm], p))
                md.append(f"| {b} | {_bin_span(periods, bins, b)} | {fit.beta[nm]:.4g} | {fit.se[nm]:.3g} | "
                          f"{_num(p, '.3f')} |")
            else:
                md.append(f"| {b} | {_bin_span(periods, bins, b)} | n/a | | |")
        md.append("")
        xs = [p[0] for p in pts]
        ax.errorbar(xs, [p[1] for p in pts], yerr=[Z * p[2] for p in pts], fmt="none", ecolor="C0",
                    elinewidth=1, capsize=2)
        ax.plot(xs, [p[1] for p in pts], color="C0", linewidth=0.8, alpha=0.6)
        for xk, beta, _, p in pts:
            ax.plot(xk, beta, "o", color="C0", markersize=4,
                    markerfacecolor="C0" if p is not None and p < 0.05 else "white")
        ax.axhline(0, color="black", linewidth=0.5)
        if "transition" in order:
            k = order.index("transition")
            ax.axvspan(k - 0.5, k + 0.5, color="grey", alpha=0.15)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["trans." if b == "transition" else str(b) for b in order], fontsize=8)
        ax.set_xlabel("fiscal years relative to mandate (ref = -1)", fontsize=8)
        ax.set_title(f"{OUTCOME_LABELS.get(name, name)} - paid-up dose", fontsize=9)
        _sci(ax, "y")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build both paid-up figures and their markdown table for one config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "main.yaml")
    parser.add_argument("--n-boot", type=int, default=None, help="default: the config's n_bootstrap")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "figures")
    args = parser.parse_args()

    cfg = load_causal_experiment_config(args.config)
    req = cfg.paid_up_capital_requirement
    if req is None:
        raise SystemExit("paid_up_capital_requirement is null in the config: nothing to plot")
    inp = build_inputs(cfg, args.config)
    n_boot = cfg.n_bootstrap if args.n_boot is None else args.n_boot
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# Paid-up figures: {args.config.stem}", ""]
    plot_compliance(cfg, inp, req, args.out_dir / "paid_up_compliance.png", md)
    plot_event_study(cfg, inp, req, n_boot, args.out_dir / "paid_up_event_study.png", md)
    (args.out_dir / "paid_up_figures.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote paid_up_compliance.png, paid_up_event_study.png, paid_up_figures.md to {args.out_dir}")


if __name__ == "__main__":
    main()
