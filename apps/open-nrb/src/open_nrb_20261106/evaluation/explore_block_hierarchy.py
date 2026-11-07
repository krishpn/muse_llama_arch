#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "numpy",
#     "scipy",
#     "matplotlib",
# ]
# ///
"""EXPLORATORY, not yet an official evaluation/ step -- sketched to check whether the model's
flat 2-block split (printed by run_real_experiment.py's top-banks table) is an artifact of
forcing K=2, or something an independent, unsupervised hierarchical clustering would recover on
its own. If graph-tool's nested-SBM-style dendrograms ever become worth the non-pip install cost
(apt/conda-only, no PyPI wheels -- see the discussion this script came out of), this is the
plain scipy/matplotlib stand-in: same question, no system package manager required.

Per-bank feature vector = time-averaged lending profile (row of W_hat) concatenated with
borrowing profile (column of W_hat), z-scored across banks -- periods flagged non-finite or
degenerate by run_real_experiment.py are excluded first via nanmean, same convention
_top_banks_by_block already uses for q. Hierarchical (average-linkage) clustering on that
feature vector is then cut at the model's own K to get a flat partition, and compared to the
model's block argmax via the same best-of-K! label-alignment search plot_synthetic_network.py
uses (block labels are unidentifiable, so comparing them directly would be meaningless without
aligning first).

Reads only the artifact run_real_experiment.py already saves (results/<config stem>/
real_run_seed{seed}.pt) -- never retrains.

Usage: uv run explore_block_hierarchy.py [--config PATH] [--seed N] [--out PATH]
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "baseline.yaml"

# Same block-color convention as plot_synthetic_network.py, so a reader who has seen that
# figure doesn't have to learn a second color scheme for the same kind of label.
BLOCK_COLORS = {0: "#e08214", 1: "#1b9e77"}


def _bank_features(W_hat: torch.Tensor, bad_periods: set[int]) -> np.ndarray:
    """(N, 2N) feature matrix: each bank's time-averaged outgoing exposure row concatenated
    with its incoming exposure column, z-scored per column so lending and borrowing scales
    don't dominate the distance just because one happens to run larger in NPR-million units."""
    good = [t for t in range(W_hat.shape[0]) if t not in bad_periods]
    mean_W = W_hat[good].mean(dim=0)  # (N, N)
    feat = torch.cat([mean_W, mean_W.T], dim=1).numpy()  # (N, 2N)
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
    return feat


def _best_permutation(flat_clusters: np.ndarray, model_blocks: np.ndarray, k: int) -> tuple[tuple[int, ...], float]:
    """Best-of-K! label alignment between the hierarchical cut's cluster IDs and the model's
    block IDs -- both are unidentifiable labels (same reason plot_synthetic_network.py's
    _best_permutation exists), so raw agreement would be meaningless without this."""
    best_perm, best_acc = tuple(range(k)), -1.0
    for perm in itertools.permutations(range(k)):
        perm_arr = np.array(perm)
        acc = float((perm_arr[flat_clusters] == model_blocks).mean())
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    return best_perm, best_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    artifact_path = RESULTS_DIR / args.config.stem / f"real_run_seed{args.seed}.pt"
    if not artifact_path.exists():
        print(f"not found: {artifact_path}\nRun run_real_experiment.py --config {args.config} first.")
        sys.exit(1)

    data = torch.load(artifact_path, weights_only=False)
    bfi_names = data["bfi_names"]
    W_hat = data["W_hat"]  # (T, N, N)
    q = data["q"]  # (T, N, K)
    bad_periods = set(data["non_finite_periods"]) | set(data["degenerate_periods"])

    k = q.shape[-1]
    good = [t for t in range(q.shape[0]) if t not in bad_periods]
    model_blocks = q[good].nanmean(dim=0).argmax(dim=-1).numpy()  # (N,)

    feat = _bank_features(W_hat, bad_periods)
    Z = linkage(pdist(feat, metric="euclidean"), method="average")

    flat_clusters = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed, to match model_blocks
    perm, agreement = _best_permutation(flat_clusters, model_blocks, k)
    print(f"hierarchical-clustering vs. model-block agreement (best label alignment, K={k}): {agreement:.3f}")

    fig, ax = plt.subplots(figsize=(9, max(6.0, 0.28 * len(bfi_names))))
    ddata = dendrogram(
        Z, labels=bfi_names, orientation="right", ax=ax,
        link_color_func=lambda i: "#999999",  # links left uncolored -- only leaf labels carry
                                                # the model's block color, so the two things
                                                # being compared (independent clustering's
                                                # structure vs. the model's own labels) stay
                                                # visually distinct.
    )
    for tick, leaf_idx in zip(ax.get_ymajorticklabels(), ddata["leaves"]):
        tick.set_color(BLOCK_COLORS.get(int(model_blocks[leaf_idx]), "#999999"))

    ax.set_title(
        f"bank hierarchy from time-averaged exposure profile (config={args.config.stem}, seed={args.seed})\n"
        f"leaf label color = model's block assignment -- flat-cut agreement vs. model: {agreement:.2f}"
    )
    ax.set_xlabel("euclidean distance (z-scored lending+borrowing profile, average linkage)")
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markersize=10, label=f"model block {b}")
        for b, c in BLOCK_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()

    out_path = args.out or (
        RESULTS_DIR / args.config.stem / "evaluation" / f"block_hierarchy_seed{args.seed}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
