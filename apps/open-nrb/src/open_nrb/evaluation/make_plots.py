#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "matplotlib",
# ]
# ///
"""Figures for the paper (Sec 6/8/12) from the artifacts run_synthetic_experiment.py and
run_real_experiment.py already save under results/ -- never re-runs training itself.

Synthetic figures (multiseed_summary.json + seed_*.pt) are built unconditionally; they don't
change once that experiment is done. Real-data figures (real_run_seed{seed}.pt/
real_run_summary_seed{seed}.json) are built only if those files exist yet -- the real run is
still in progress on the remote machine as of this script's first version, so this half is a
no-op (printed, not an error) until that artifact lands, rather than something to re-run by
hand later.

Usage: uv run make_plots.py [--seed N] [--config PATH] [--synthetic-config PATH]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this runs on the remote machine next to training, no display
import matplotlib.pyplot as plt
import torch

# evaluation/ is a sibling of results/ and experiments/ under src/open_nrb/, so parent.parent
# lands on src/open_nrb/ regardless of where the repo is cloned -- same location
# run_real_experiment.py/run_synthetic_experiment.py write to and load their default configs
# from, this script only ever reads from here.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "baseline.yaml"
DEFAULT_SYNTHETIC_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "synthetic_baseline.yaml"


def _savefig(fig: plt.Figure, config_stem: str, name: str) -> None:
    # Namespaced by the config file's stem, same convention run_real_experiment.py/
    # run_synthetic_experiment.py write results under -- two variants' figures must not collide
    # just because they share a file name. Lives under the same evaluation/ dir every other
    # evaluation script writes its report/metrics to for this config, not a separate top-level
    # figures/ directory.
    out_dir = RESULTS_DIR / config_stem / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def plot_synthetic(config_path: Path) -> None:
    # Namespaced by the config file's stem, matching the directory run_synthetic_experiment.py
    # actually wrote results under for this variant.
    results_dir = RESULTS_DIR / config_path.stem
    summary_path = results_dir / "multiseed_summary.json"
    if not summary_path.exists():
        print(f"skip synthetic figures: {summary_path} not found")
        return

    with open(summary_path) as f:
        summary = json.load(f)
    per_seed = summary["per_seed"]
    seeds = [r["seed"] for r in per_seed]

    # Fig 1: model vs. classical baseline relative Frobenius error, per seed (Sec 7's mandatory
    # comparison -- grouped bars so a reader can see it's not a single lucky seed).
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    x = range(len(seeds))
    ax.bar([i - width / 2 for i in x], [r["model_mean"] for r in per_seed], width, label="model")
    ax.bar([i + width / 2 for i in x], [r["baseline_mean"] for r in per_seed], width, label="RAS/Sinkhorn baseline")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("mean relative Frobenius error")
    ax.set_title("Model vs. classical baseline reconstruction error (Sec 6/7)")
    ax.legend()
    _savefig(fig, config_path.stem, "synthetic_error_vs_baseline.png")

    # Fig 2: block_acc vs the trivial majority-guess floor, per seed (Sec 10 open item: is the
    # margin real or single-seed noise?) -- same grouped-bar framing as Fig 1.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], [r["block_acc"] for r in per_seed], width, label="block_acc")
    ax.bar([i + width / 2 for i in x], [r["trivial_baseline"] for r in per_seed], width, label="trivial majority-guess")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("accuracy")
    ax.set_title("Block recovery accuracy vs. trivial floor (Sec 10)")
    ax.legend()
    _savefig(fig, config_path.stem, "synthetic_block_acc_vs_trivial.png")

    # Fig 3: training trajectories across all seeds -- error_trend / block_acc_trend vs epoch,
    # one faint line per seed plus the across-seed mean, to show the collapse/recovery dynamics
    # Sec 10 describes rather than just the single post-training number Figs 1-2 report.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    all_error_series, all_block_series = [], []
    for seed in seeds:
        artifact = torch.load(results_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
        epochs_e, errors = zip(*artifact["error_trend"])
        epochs_b, block_accs = zip(*artifact["block_acc_trend"])
        axes[0].plot(epochs_e, errors, alpha=0.3, color="C0")
        axes[1].plot(epochs_b, block_accs, alpha=0.3, color="C0")
        all_error_series.append(errors)
        all_block_series.append(block_accs)

    mean_errors = [sum(vals) / len(vals) for vals in zip(*all_error_series)]
    mean_block_accs = [sum(vals) / len(vals) for vals in zip(*all_block_series)]
    axes[0].plot(epochs_e, mean_errors, color="C0", linewidth=2, label="mean across seeds")
    axes[1].plot(epochs_b, mean_block_accs, color="C0", linewidth=2, label="mean across seeds")

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("model mean relative Frobenius error")
    axes[0].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("block recovery accuracy")
    axes[1].legend()
    fig.suptitle("Training trajectories across seeds (Sec 10)")
    _savefig(fig, config_path.stem, "synthetic_training_trajectories.png")


def plot_real(seed: int, config_path: Path) -> None:
    # Namespaced by the config file's stem, matching the directory run_real_experiment.py
    # actually wrote results under for this variant.
    results_dir = RESULTS_DIR / config_path.stem
    summary_path = results_dir / f"real_run_summary_seed{seed}.json"
    artifact_path = results_dir / f"real_run_seed{seed}.pt"
    if not summary_path.exists() or not artifact_path.exists():
        print(f"skip real-data figures: {summary_path} / {artifact_path} not found yet")
        return

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    periods = artifact["periods"]  # list of (year_bs, month_bs), one per period, chronological
    period_labels = [f"{y}-{m}" for y, m in periods]
    model_stability = artifact["model_stability"]  # length T-1, one per consecutive pair
    baseline_stability = artifact["baseline_stability"]
    baseline_diff = artifact["baseline_diff"]

    # Fig 4: temporal stability over time, model vs classical baseline (Sec 8/12's headline
    # indirect evidence -- there's no W_true for real data, so this IS the evaluation, not a
    # supplement to an accuracy number).
    fig, ax = plt.subplots(figsize=(10, 4))
    x = period_labels[1:]  # stability is defined on consecutive pairs, so it starts at period 1
    ax.plot(x, model_stability, label="model", marker="o", markersize=3)
    ax.plot(x, baseline_stability, label="RAS/Sinkhorn baseline", marker="o", markersize=3)
    ax.set_ylabel("||W_t - W_(t-1)||_F / ||W_t||_F")
    ax.set_title("Temporal stability of reconstructed interbank network (Sec 8/12)")
    ax.legend()
    ax.tick_params(axis="x", rotation=90)
    if len(x) > 24:
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % (len(x) // 24 + 1) == 0)
    _savefig(fig, config_path.stem, "real_temporal_stability.png")

    # Fig 5: descriptive relative difference from the classical baseline, over time -- NOT a
    # win/lose framing (no ground truth to call a winner against, Sec 7).
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(period_labels, baseline_diff, marker="o", markersize=3, color="C2")
    ax.set_ylabel("relative Frobenius difference from baseline")
    ax.set_title("Model vs. classical baseline, descriptive only (Sec 7/8)")
    ax.tick_params(axis="x", rotation=90)
    if len(period_labels) > 24:
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % (len(period_labels) // 24 + 1) == 0)
    _savefig(fig, config_path.stem, "real_baseline_diff.png")

    # Fig 6: descriptive block assignment -- top-10 banks by mean posterior, one panel per block
    # (no ground truth z for real data, so this is a face-validity check, not an accuracy claim).
    bfi_names = artifact["bfi_names"]
    q_mean = artifact["q"].mean(dim=0)  # (N, K), time-averaged posterior-predictive mean
    k = q_mean.shape[1]
    fig, axes = plt.subplots(1, k, figsize=(6 * k, 5))
    if k == 1:
        axes = [axes]
    for block in range(k):
        scores = q_mean[:, block].tolist()
        ranked = sorted(zip(bfi_names, scores), key=lambda x: -x[1])[:10]
        names, vals = zip(*ranked)
        axes[block].barh(range(len(names)), vals, color=f"C{block}")
        axes[block].set_yticks(range(len(names)))
        axes[block].set_yticklabels(names)
        axes[block].invert_yaxis()
        axes[block].set_xlabel("mean posterior probability")
        axes[block].set_title(f"block {block}")
    fig.suptitle("Top banks by block, descriptive only -- no ground truth (Sec 8/12)")
    _savefig(fig, config_path.stem, "real_top_banks_by_block.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--synthetic-config", type=Path, default=DEFAULT_SYNTHETIC_CONFIG)
    args = parser.parse_args()

    plot_synthetic(args.synthetic_config)
    plot_real(args.seed, args.config)


if __name__ == "__main__":
    main()
