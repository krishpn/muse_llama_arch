#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "matplotlib",
#     "numpy",
# ]
# ///


"""Exploratory data analysis for survivorship bias in the complete-case bank filter
(run_real_experiment.py's _filter_complete_banks) -- read-only, no training, no changes
to data_loader.py/train_step.py/run_real_experiment.py. Answers, with descriptive stats
only, directly from the raw panel:

  1. How many banks does the complete-case filter actually drop, and for how long were
     they present (late entry / early exit / internal gap / all-window zero-denominator)?
  2. Were the dropped banks systematically bigger / more interbank-active (during their
     own active window) than the banks the filter keeps -- i.e. is this survivorship bias
     a real, directionally-biased effect, or just missing-at-random noise?

Usage: uv run eda_survivorship.py [--csv PATH]
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import warnings

import torch



import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import torch

try:
    from .data_loader import Panel, filter_commercial_banks, load_panel, read_rows
except ImportError:
    from data_loader import Panel, filter_commercial_banks, load_panel, read_rows


                   #/src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv

DEFAULT_CSV = Path("src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _has_rows_matrix(csv_path: Path, panel: Panel) -> torch.Tensor:
    """(N, T) bool: True where bank i has >=1 extracted row (any field) in period t --
    independent of load_panel's NaN pattern, which ALSO NaNs case-C zero-denominator cells
    for banks that DO have rows that period (data_loader.py's size==0/deposits==0 note).
    Needed to tell true entry/exit (no rows at all) apart from "present the whole window
    but excluded for a different reason" below.
    """
    rows = read_rows(csv_path)
    period_index = {pk: t for t, pk in enumerate(panel.periods)}
    name_index = {name: i for i, name in enumerate(panel.bfi_names)}
    has_rows = torch.zeros(len(panel.bfi_names), len(panel.periods), dtype=torch.bool)
    for r in rows:
        pk = (r.year_bs, r.month_bs)  # already normalized by read_rows, matches panel.periods labels
        if pk in period_index and r.bfi_name in name_index:
            has_rows[name_index[r.bfi_name], period_index[pk]] = True
    return has_rows


def classify_banks(panel: Panel, has_rows: torch.Tensor) -> dict[str, list[int]]:
    """Buckets every bank index by why run_real_experiment.py's complete-case filter would
    drop it (or confirm it's kept), over the FULL period range (no START_PERIOD/END_PERIOD
    narrowing -- that's a separate per-run knob; this is the underlying panel-wide picture).
    """
    n, t = has_rows.shape
    nan_any_cell = (
        panel.X.isnan().any(dim=2) | panel.r.isnan() | panel.c.isnan() | panel.K.isnan()
    )  
    buckets: dict[str, list[int]] = {
        "complete": [], "late_entry": [], "early_exit": [], "internal_gap": [],
        "zero_denom_only": [],
    }
    for i in range(n):
        present = has_rows[i]
        if not bool(present.any()):
            continue  # bank never appears at all in this CSV -- not a filter victim, skip
        idxs = present.nonzero(as_tuple=True)[0]
        first, last = idxs[0].item(), idxs[-1].item()
        fully_present = bool(present.all())
        no_nan = not bool(nan_any_cell[i].any())

        if fully_present and no_nan:
            buckets["complete"].append(i)
        elif fully_present and not no_nan:
            buckets["zero_denom_only"].append(i)
        elif not bool(present[first : last + 1].all()):
            buckets["internal_gap"].append(i)
        elif first > 0 and last == t - 1:
            buckets["late_entry"].append(i)
        elif last < t - 1 and first == 0:
            buckets["early_exit"].append(i)
        else:
            buckets["internal_gap"].append(i)  # both late entry AND early exit
    return buckets


def _nanmean(x: torch.Tensor) -> float:
    finite = x[~x.isnan()]
    return finite.mean().item() if finite.numel() else float("nan")


def _bank_activity_stats(panel: Panel, has_rows: torch.Tensor, i: int) -> dict[str, float]:
    """Centrality proxies computed directly from the raw panel -- no model, no training --
    over bank i's OWN active window only, so a late entrant/early exiter is judged on the
    data it actually has, not penalized again for the periods it was never going to report.
    """
    present = has_rows[i]
    size = panel.X[i, present, 0]  # COVARIATE_ORDER[0] == "size"
    activity = panel.r[i, present] + panel.c[i, present]  # NaN-propagates per period (case C)
    return {
        "n_active_periods": int(present.sum()),
        "mean_size": _nanmean(size),
        "mean_interbank_activity": _nanmean(activity),
    }


def _plot(stats_by_bucket: dict[str, list[dict]]) -> None:
    labels = [l for l, s in stats_by_bucket.items() if s]
    size_data = [[s["mean_size"] for s in stats_by_bucket[l] if s["mean_size"] == s["mean_size"]] for l in labels]
    act_data = [
        [s["mean_interbank_activity"] for s in stats_by_bucket[l] if s["mean_interbank_activity"] == s["mean_interbank_activity"]]
        for l in labels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(size_data, tick_labels=labels)
    axes[0].set_ylabel("mean size (own active window)")
    axes[0].set_title("Size by complete-case bucket")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].boxplot(act_data, tick_labels=labels)
    axes[1].set_ylabel("mean interbank lending+borrowing (own active window)")
    axes[1].set_title("Interbank activity by complete-case bucket")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Survivorship-bias EDA: does the complete-case filter drop bigger/more-active banks?")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "eda_survivorship.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    panel = load_panel(args.csv)
    panel = filter_commercial_banks(panel)
    has_rows = _has_rows_matrix(args.csv, panel)
    buckets = classify_banks(panel, has_rows)

    print(f"=== survivorship/complete-case EDA ({len(panel.bfi_names)} banks, {len(panel.periods)} periods, full range) ===\n")
    for label, idxs in buckets.items():
        print(f"{label:16s}: {len(idxs):3d} banks")
    accounted = sum(len(v) for v in buckets.values())
    never_seen = len(panel.bfi_names) - accounted
    if never_seen:
        print(f"{'never_appears':16s}: {never_seen:3d} banks (zero rows in entire CSV -- check name matching)")

    stats_by_bucket = {label: [_bank_activity_stats(panel, has_rows, i) for i in idxs] for label, idxs in buckets.items()}

    print("\n=== mean size / interbank activity, by bucket (descriptive, during each bank's OWN active window) ===")
    for label, stats in stats_by_bucket.items():
        if not stats:
            continue
        sizes = [s["mean_size"] for s in stats if s["mean_size"] == s["mean_size"]]
        acts = [s["mean_interbank_activity"] for s in stats if s["mean_interbank_activity"] == s["mean_interbank_activity"]]
        mean_size = sum(sizes) / len(sizes) if sizes else float("nan")
        mean_act = sum(acts) / len(acts) if acts else float("nan")
        print(f"{label:16s}  n={len(stats):3d}  mean_size={mean_size:14.2f}  mean_interbank_activity={mean_act:12.2f}")

    print("\n=== dropped banks, by bucket (name: n_active_periods / total_periods) ===")
    t_total = len(panel.periods)
    for label, idxs in buckets.items():
        if label == "complete" or not idxs:
            continue
        print(f"\n{label}:")
        for i in idxs:
            n_active = int(has_rows[i].sum())
            print(f"  {panel.bfi_names[i]:40s} {n_active:3d}/{t_total}")

    _plot(stats_by_bucket)


if __name__ == "__main__":
    main()
