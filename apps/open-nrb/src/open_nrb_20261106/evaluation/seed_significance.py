#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "scipy",
# ]
# ///
"""Paired significance tests on the model-vs-baseline gap across seeds (Sec 7/10's "is the margin
real or single-seed noise?" question) -- this is the only script in the codebase that needs scipy;
hand-rolling a paired t-test/Wilcoxon signed-rank test isn't worth it for one self-contained
script, everything else here is hand-rolled torch/math on purpose.

Reads only multiseed_summary.json (never retrains). Caveat printed below and worth repeating here:
at n=5 seeds (today's synthetic_baseline.yaml), Wilcoxon's minimum achievable two-sided p-value is
1/16 = 0.0625 -- it structurally cannot reject at alpha=0.05 at this seed count. The paired mean
difference (effect size) is the number that actually carries information until more seeds are run.

Usage: uv run seed_significance.py [--synthetic-config PATH] [--compare-config PATH]
"""

import argparse
import json
import sys
from pathlib import Path

from scipy import stats

try:
    from .report_io import captured_output, save_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report_io import captured_output, save_report

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_SYNTHETIC_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "synthetic_baseline.yaml"


def _load_per_seed(config_path: Path) -> list[dict]:
    summary_path = RESULTS_DIR / config_path.stem / "multiseed_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found -- run run_synthetic_experiment.py --config {config_path} first."
        )
    with open(summary_path) as f:
        return json.load(f)["per_seed"]


def _paired_report(label: str, a: list[float], b: list[float]) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    mean_diff = sum(diffs) / len(diffs)
    print(f"\n=== {label} ===")
    print(f"  n_seeds={len(a)}  mean paired difference (a - b): {mean_diff:.6f}")
    t_stat, t_p = stats.ttest_rel(a, b)
    print(f"  paired t-test:      statistic={t_stat:.4f}  p={t_p:.4f}")
    result = {
        "n_seeds": len(a),
        "mean_paired_difference": mean_diff,
        "ttest_rel": {"statistic": float(t_stat), "p_value": float(t_p)},
        "wilcoxon": None,
    }
    try:
        w_stat, w_p = stats.wilcoxon(diffs)
        print(f"  Wilcoxon signed-rank: statistic={w_stat:.4f}  p={w_p:.4f}")
        result["wilcoxon"] = {"statistic": float(w_stat), "p_value": float(w_p)}
    except ValueError as e:
        # All-zero differences (or too few non-tied pairs) -- scipy raises rather than
        # returning a degenerate result; report that explicitly instead of crashing.
        print(f"  Wilcoxon signed-rank: not computable ({e})")
        result["wilcoxon"] = {"error": str(e)}
    n = len(a)
    if n <= 6:
        min_p = 1 / (2 ** (n - 1))
        print(
            f"  NOTE: at n={n} seeds, Wilcoxon's minimum achievable two-sided p-value is "
            f"1/2^(n-1) = {min_p:.4f} -- it structurally cannot reject at alpha=0.05 here. "
            "Judge the mean paired difference (effect size) above, not this p-value; running "
            "more seeds is the only way to get a p-value that can actually move."
        )
        result["wilcoxon_min_achievable_p"] = min_p
    return result


def run_within_config(config_path: Path) -> None:
    with captured_output() as buf:
        per_seed = _load_per_seed(config_path)
        print(f"\n########## Seed significance: {config_path.stem} (n={len(per_seed)} seeds) ##########")
        model_vs_baseline = _paired_report(
            "model_mean vs. baseline_mean (reconstruction error, lower is better)",
            [r["model_mean"] for r in per_seed],
            [r["baseline_mean"] for r in per_seed],
        )
        block_acc_vs_trivial = _paired_report(
            "block_acc vs. trivial_baseline (block recovery accuracy, higher is better)",
            [r["block_acc"] for r in per_seed],
            [r["trivial_baseline"] for r in per_seed],
        )
    metrics = {"model_mean_vs_baseline_mean": model_vs_baseline, "block_acc_vs_trivial_baseline": block_acc_vs_trivial}
    save_report(RESULTS_DIR / config_path.stem / "evaluation", "seed_significance", buf.getvalue(), metrics)


def run_between_configs(config_a: Path, config_b: Path) -> None:
    per_seed_a = {r["seed"]: r for r in _load_per_seed(config_a)}
    per_seed_b = {r["seed"]: r for r in _load_per_seed(config_b)}
    shared_seeds = sorted(set(per_seed_a) & set(per_seed_b))
    if len(shared_seeds) < 2:
        print(
            f"\nskip between-config comparison: only {len(shared_seeds)} shared seed(s) between "
            f"{config_a.stem} and {config_b.stem} -- need matched seeds on both sides to pair."
        )
        return
    with captured_output() as buf:
        print(
            f"\n########## Seed significance: {config_a.stem} vs. {config_b.stem} "
            f"(n={len(shared_seeds)} shared seeds) ##########"
        )
        model_mean = _paired_report(
            "model_mean: a vs. b",
            [per_seed_a[s]["model_mean"] for s in shared_seeds],
            [per_seed_b[s]["model_mean"] for s in shared_seeds],
        )
        block_acc = _paired_report(
            "block_acc: a vs. b",
            [per_seed_a[s]["block_acc"] for s in shared_seeds],
            [per_seed_b[s]["block_acc"] for s in shared_seeds],
        )
    metrics = {"model_mean": model_mean, "block_acc": block_acc}
    save_report(RESULTS_DIR / f"{config_a.stem}_vs_{config_b.stem}" / "evaluation", "seed_significance", buf.getvalue(), metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-config", type=Path, default=DEFAULT_SYNTHETIC_CONFIG)
    parser.add_argument("--compare-config", type=Path, default=None)
    args = parser.parse_args()

    run_within_config(args.synthetic_config)
    if args.compare_config is not None:
        run_between_configs(args.synthetic_config, args.compare_config)


if __name__ == "__main__":
    main()
