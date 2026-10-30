#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
# ]
# ///
"""Diff two config variants' metrics side by side (e.g. `baseline` vs `ablation_no_kl_pressure`)
instead of making the user eyeball two separate runs/scripts. Reuses the metric functions already
defined in eda_reviewer_questions.py and eda_network_structure.py directly -- no metric logic is
reimplemented here, this script only loads artifacts, calls those functions, and tabulates.

Real-data comparison needs a trained real_run_seed{seed}.pt on each side (single run, so metrics
are the seed-specific numbers run_section_b would print). Synthetic comparison needs a
multiseed_summary.json on each side (metrics are averaged across that config's own seeds).

Usage:
  uv run compare_configs.py --config-a PATH --config-b PATH [--seed-a N] [--seed-b N]
  uv run compare_configs.py --synthetic-config-a PATH --synthetic-config-b PATH
"""

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from .eda_network_structure import (
        aggregate_debtrank,
        centrality_rank_agreement,
        density,
        eigenvector_centrality,
        marginal_preservation_error,
        weighted_reciprocity,
    )
    from .eda_reviewer_questions import _valid_period_mask, cap_binding_fraction, lag1_autocorrelation
    from .report_io import captured_output, save_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eda_network_structure import (
        aggregate_debtrank,
        centrality_rank_agreement,
        density,
        eigenvector_centrality,
        marginal_preservation_error,
        weighted_reciprocity,
    )
    from eda_reviewer_questions import _valid_period_mask, cap_binding_fraction, lag1_autocorrelation
    from report_io import captured_output, save_report

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_real_metrics(config_path: Path, seed: int) -> dict[str, float]:
    artifact_path = RESULTS_DIR / config_path.stem / f"real_run_seed{seed}.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"{artifact_path} not found -- run run_real_experiment.py --config {config_path} first.")
    data = torch.load(artifact_path, weights_only=False)
    W_hat, W_baseline, cap = data["W_hat"], data["W_baseline"], data["cap"]
    valid = _valid_period_mask(W_hat, W_baseline)

    metrics = {
        "lag1_autocorrelation (model)": lag1_autocorrelation(W_hat, valid),
        "cap_binding_fraction (model)": cap_binding_fraction(W_hat, cap, valid),
        "density (model)": density(W_hat, valid)["mean"],
        "weighted_reciprocity (model)": weighted_reciprocity(W_hat, valid)["mean"],
        "debtrank mean_shock_impact (model)": aggregate_debtrank(W_hat, cap, valid)["mean_shock_impact"]["mean"],
    }
    if "r" in data and "c" in data:
        mp = marginal_preservation_error(W_hat, data["r"].T, data["c"].T, valid)
        metrics["marginal_preservation row err (model)"] = mp["row"]["mean"]
        metrics["marginal_preservation col err (model)"] = mp["col"]["mean"]

    centrality_hat = eigenvector_centrality(W_hat)
    centrality_base = eigenvector_centrality(W_baseline)
    metrics["centrality_rank_agreement (model vs baseline)"] = centrality_rank_agreement(centrality_hat, centrality_base, valid)["mean"]
    return metrics


def compute_synthetic_metrics(config_path: Path) -> dict[str, float]:
    summary_path = RESULTS_DIR / config_path.stem / "multiseed_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found -- run run_synthetic_experiment.py --config {config_path} first.")
    with open(summary_path) as f:
        per_seed = json.load(f)["per_seed"]
    n = len(per_seed)
    return {
        "model_mean (avg over seeds)": sum(r["model_mean"] for r in per_seed) / n,
        "baseline_mean (avg over seeds)": sum(r["baseline_mean"] for r in per_seed) / n,
        "block_acc (avg over seeds)": sum(r["block_acc"] for r in per_seed) / n,
        "trivial_baseline (avg over seeds)": sum(r["trivial_baseline"] for r in per_seed) / n,
    }


def print_comparison(label_a: str, metrics_a: dict[str, float], label_b: str, metrics_b: dict[str, float]) -> dict:
    print(f"\n{'metric':42s} {label_a:>16s} {label_b:>16s} {'delta':>12s} {'% change':>12s}")
    print("-" * 102)
    result = {}
    for name in metrics_a:
        a, b = metrics_a[name], metrics_b.get(name, float("nan"))
        delta = a - b
        pct = (delta / abs(b) * 100) if b == b and b != 0 else float("nan")
        print(f"{name:42s} {a:16.4f} {b:16.4f} {delta:12.4f} {pct:11.2f}%")
        result[name] = {"a": a, "b": b, "delta": delta, "pct_change": pct}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", type=Path, default=None)
    parser.add_argument("--config-b", type=Path, default=None)
    parser.add_argument("--seed-a", type=int, default=0)
    parser.add_argument("--seed-b", type=int, default=0)
    parser.add_argument("--synthetic-config-a", type=Path, default=None)
    parser.add_argument("--synthetic-config-b", type=Path, default=None)
    args = parser.parse_args()

    if args.config_a and args.config_b:
        with captured_output() as buf:
            metrics_a = compute_real_metrics(args.config_a, args.seed_a)
            metrics_b = compute_real_metrics(args.config_b, args.seed_b)
            comparison = print_comparison(args.config_a.stem, metrics_a, args.config_b.stem, metrics_b)
        out_dir = RESULTS_DIR / f"{args.config_a.stem}_vs_{args.config_b.stem}" / "evaluation"
        save_report(out_dir, "compare_configs_real", buf.getvalue(), comparison)

    if args.synthetic_config_a and args.synthetic_config_b:
        with captured_output() as buf:
            metrics_a = compute_synthetic_metrics(args.synthetic_config_a)
            metrics_b = compute_synthetic_metrics(args.synthetic_config_b)
            comparison = print_comparison(args.synthetic_config_a.stem, metrics_a, args.synthetic_config_b.stem, metrics_b)
        out_dir = RESULTS_DIR / f"{args.synthetic_config_a.stem}_vs_{args.synthetic_config_b.stem}" / "evaluation"
        save_report(out_dir, "compare_configs_synthetic", buf.getvalue(), comparison)

    if not (args.config_a and args.config_b) and not (args.synthetic_config_a and args.synthetic_config_b):
        parser.error("provide either --config-a/--config-b or --synthetic-config-a/--synthetic-config-b")


if __name__ == "__main__":
    main()
