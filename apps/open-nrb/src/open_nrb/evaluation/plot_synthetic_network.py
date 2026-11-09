#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "matplotlib",
# ]
# ///
"""Plot the synthetic core/periphery network: ground truth vs reconstruction.

Qualitative core/periphery network plot for the synthetic validation (design doc Sec 6/12)
-- the visual companion eda_model_checks.py explicitly scopes out ("formal core-periphery
block-fitting" is out of scope there; eigenvector centrality + DebtRank answer the quantitative
"does reconstruction preserve who's systemically important" question without it). Plots ground
truth W_true next to the model's reconstructed W_hat for one period, nodes laid out on two
concentric rings by TRUE block membership (inner ring = whichever block has the denser mean
within-block weight in W_true that period, i.e. "core" -- data-driven, not assumed to be label
0) so a correct reconstruction should visibly concentrate edges in/between the inner ring, not
just score well on Frobenius error. Node outlines mark whether the model's inferred block
(argmax q, best-permutation-aligned the same way _block_recovery_accuracy is, since block
labels are unidentifiable -- Eq. 1 has no canonical ordering) agrees with the true label for
that bank in that period.

Caveat carried over from run_synthetic_experiment.py's _build_panel: z is resampled
independently each period (no time index in Eq. 1 either way), so this is a snapshot of ONE
period's block draw, not a stable bank identity across periods -- pick --period deliberately,
and don't read the same bank index as "the same bank's block" across plots of different
periods.

Reads only the artifact run_synthetic_experiment.py already saves
(results/<config stem>/seed_{seed}.pt) -- never retrains.

Usage: uv run plot_synthetic_network.py [--config PATH] [--seed N] [--period N] [--top-k N] [--out PATH]
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "synthetic_baseline.yaml"


def _best_permutation(q_final: torch.Tensor, zs: torch.Tensor) -> tuple[int, ...]:
    """Same best-of-K! permutation search as run_synthetic_experiment.py's
    _block_recovery_accuracy, pooled over ALL periods (not just the one being plotted) --
    Lambda/eta_block are fixed across periods, so the model's label<->semantic-block mapping
    is one global correspondence, not something to re-fit per period.
    """
    k = q_final.shape[-1]
    preds = q_final.argmax(dim=-1).reshape(-1)
    trues = zs.reshape(-1)
    best_perm, best_acc = tuple(range(k)), -1.0
    for perm in itertools.permutations(range(k)):
        perm_t = torch.tensor(perm)
        acc = (perm_t[preds] == trues).float().mean().item()
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    return best_perm


def _core_label(W: torch.Tensor, z: torch.Tensor, k: int) -> int:
    """Whichever block has the higher mean within-block edge weight in this period's W is
    "core" for layout purposes -- data-driven rather than assuming block 0 is always core,
    since block labels have no canonical meaning (same reason _best_permutation exists)."""
    means = []
    for b in range(k):
        idx = (z == b).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            means.append(0.0)
            continue
        sub = W[idx][:, idx]
        means.append(sub.mean().item())
    return int(np.argmax(means))


def _block_order(z: torch.Tensor, core_label: int) -> list[int]:
    """Core-block indices first, periphery second -- shared ordering for both the ring
    layout and the difference heatmap, so "core" means the same set of banks in every panel."""
    core_idx = [i for i in range(len(z)) if z[i].item() == core_label]
    periph_idx = [i for i in range(len(z)) if z[i].item() != core_label]
    return core_idx + periph_idx


def _ring_layout(order: list[int], core_size: int) -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    core_idx, periph_idx = order[:core_size], order[core_size:]
    for rank, i in enumerate(core_idx):
        theta = 2 * np.pi * rank / max(len(core_idx), 1)
        pos[i] = (0.45 * np.cos(theta), 0.45 * np.sin(theta))
    for rank, i in enumerate(periph_idx):
        theta = 2 * np.pi * rank / max(len(periph_idx), 1)
        pos[i] = (1.0 * np.cos(theta), 1.0 * np.sin(theta))
    return pos


def _draw_network(
    ax, W: torch.Tensor, z: torch.Tensor, pos: dict, misclassified: set[int], top_k: int,
    max_w: float, title: str,
) -> None:
    n = W.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    w_abs = W.abs() * mask

    # Keep only each node's top_k heaviest outgoing edges -- a full dense (N, N) matrix drawn
    # at once is unreadable past a handful of nodes.
    edges = []
    for i in range(n):
        row = w_abs[i]
        k_i = min(top_k, int((row > 0).sum().item()))
        if k_i == 0:
            continue
        top = torch.topk(row, k_i).indices.tolist()
        for j in top:
            edges.append((i, j, W[i, j].item()))

    # max_w is passed in SHARED across both the truth and model panels (not recomputed per
    # panel) -- otherwise each panel's edges get rescaled to its own max, which can make a
    # genuinely under/over-confident reconstruction look just as "thick" as the ground truth
    # even when the absolute weights differ substantially.
    #
    # Curved (not straight) edges: a straight i->j line sits exactly on top of j->i, hiding
    # reciprocal lending pairs entirely. Curvature sign flips on i<j so each direction of a
    # reciprocal pair bows the opposite way -- same convention igraph/ggraph use for directed
    # graphs with mutual edges.
    #
    # Edge color is drawn from the SAME "RdBu_r" colormap as the W_hat-W_true heatmap, so a
    # reader learns one color scale for the whole figure instead of two unrelated ones. Only
    # the warm (red) half is used: edge weights here are raw exposures (always >= 0 by
    # construction -- a log-normal zero-inflated draw, see synthetic.py), not signed errors,
    # so mapping them onto the colormap's blue half would falsely imply they can go negative.
    # White -> dark red still means the same thing it means in the heatmap: "larger value."
    edge_cmap = plt.get_cmap("RdBu_r")
    node_r = 0.045
    for i, j, w in edges:
        frac = abs(w) / max_w
        rad = 0.18 if i < j else -0.18
        arrow = FancyArrowPatch(
            pos[i], pos[j],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=8 + 10 * frac,
            linewidth=0.4 + 2.4 * frac,
            color=edge_cmap(0.5 + 0.48 * frac),
            alpha=0.9,
            shrinkA=13, shrinkB=13,
            zorder=1,
        )
        ax.add_patch(arrow)

    face_colors = {0: "#e08214", 1: "#1b9e77"}
    for i in range(n):
        x, y = pos[i]
        face = face_colors.get(z[i].item(), "#999999")
        halo = "#d7191c" if i in misclassified else "white"
        ax.add_patch(Circle((x, y), node_r * 1.6, facecolor=halo, edgecolor="none", zorder=2))
        ax.add_patch(Circle((x, y), node_r, facecolor=face, edgecolor="black", linewidth=1.0, zorder=3))
        ax.annotate(
            str(i), (x, y), ha="center", va="center", fontsize=6.5, zorder=4,
            color="white", fontweight="bold",
        )

    for radius in (0.45, 1.0):
        ax.add_patch(
            Circle((0, 0), radius, facecolor="none", edgecolor="#cccccc", linewidth=0.8, linestyle="--", zorder=0)
        )

    ax.set_title(title)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    """Load one seed's synthetic artifact and render the truth/model/difference figure.

    Resolves the requested period, aligns the model's block labels to the truth via the
    best-of-K! permutation, lays nodes on core/periphery rings, and draws the ground-truth
    and reconstructed networks side by side with a reordered W_hat - W_true heatmap.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--period", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    artifact_path = RESULTS_DIR / args.config.stem / f"seed_{args.seed}.pt"
    if not artifact_path.exists():
        print(f"not found: {artifact_path}\nRun run_synthetic_experiment.py --config {args.config} first.")
        sys.exit(1)

    data = torch.load(artifact_path, weights_only=False)
    zs = data["zs"]  # (N_PERIODS, N)
    W_true = data["W_true"]  # (N_PERIODS, N, N)
    W_hat = data["W_hat_final"]  # (N_PERIODS, N, N)
    q_final = data["q_final"]  # (N_PERIODS, N, K)

    n_periods = zs.shape[0]
    period = args.period if args.period >= 0 else n_periods + args.period
    if not (0 <= period < n_periods):
        print(f"--period must resolve into [0, {n_periods}); got {args.period}")
        sys.exit(1)

    k = q_final.shape[-1]
    perm = _best_permutation(q_final, zs)
    perm_t = torch.tensor(perm)

    z_t = zs[period]
    preds_t = perm_t[q_final[period].argmax(dim=-1)]
    misclassified = {i for i in range(len(z_t)) if preds_t[i].item() != z_t[i].item()}
    acc = 1.0 - len(misclassified) / len(z_t)

    core = _core_label(W_true[period], z_t, k)
    order = _block_order(z_t, core)
    core_size = int((z_t == core).sum().item())
    pos = _ring_layout(order, core_size)

    # Shared across both ring panels (see _draw_network's comment) so edge thickness is
    # comparable in absolute terms, not just relative to each panel's own max.
    max_w = max(W_true[period].abs().max().item(), W_hat[period].abs().max().item()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    _draw_network(
        axes[0], W_true[period], z_t, pos, set(), args.top_k, max_w,
        f"ground truth W_true (period {period})",
    )
    _draw_network(
        axes[1], W_hat[period], z_t, pos, misclassified, args.top_k, max_w,
        f"model W_hat (period {period}) -- block acc this period: {acc:.2f}",
    )

    # Direct W_hat - W_true view: the ring panels can look deceptively similar once the model
    # is doing reasonably well (same top-k edges selected, similar relative thickness) even
    # when the underlying weights still differ -- this panel surfaces that difference
    # directly instead of relying on eyeballing two near-identical layouts. Reordered so the
    # core block (same ordering as the rings) is grouped first in both axes, with a line
    # marking the core/periphery boundary.
    order_t = torch.tensor(order)
    diff = (W_hat[period] - W_true[period])[order_t][:, order_t].numpy()
    diff_max = max(abs(diff.min()), abs(diff.max())) or 1.0
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
    axes[2].axvline(core_size - 0.5, color="black", linewidth=1)
    axes[2].axhline(core_size - 0.5, color="black", linewidth=1)
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels(order, fontsize=6, rotation=90)
    axes[2].set_yticks(range(len(order)))
    axes[2].set_yticklabels(order, fontsize=6)
    axes[2].set_title("W_hat - W_true\n(reordered: core block first)", fontsize=10)
    # Tick labels are bank indices -- the SAME numbers drawn inside the nodes in the two ring
    # plots -- just reordered core-block-first rather than left in their ring position, so a
    # reader can trace e.g. bank "7" from a ring node to its row/column here.
    axes[2].set_xlabel("bank index (borrower, same numbering as ring plots)", fontsize=7)
    axes[2].set_ylabel("bank index (lender, same numbering as ring plots)", fontsize=7)

    # Label the core/periphery split directly on the heatmap axes (above the top edge, below
    # the x tick labels) instead of relying on a reader to cross-reference the black divider
    # line against the ring plots' legend.
    n_total = len(order)
    axes[2].text(core_size / 2 - 0.5, -1.6, "core", ha="center", va="bottom", fontsize=7, color="#555555")
    axes[2].text(core_size + (n_total - core_size) / 2 - 0.5, -1.6, "periphery", ha="center", va="bottom", fontsize=7, color="#555555")

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.035, pad=0.03, shrink=0.85)
    cbar.set_label("W_hat - W_true (red = overestimate, blue = underestimate)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#e08214", markeredgecolor="black", markersize=10, label=f"block {core} (core)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1b9e77", markeredgecolor="black", markersize=10, label="other block (periphery)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#d7191c", markersize=10, markeredgewidth=2, label="misclassified (W_hat panel)"),
        Line2D([0], [0], color="#b2182b", linewidth=2.5, label="edge: top-5 outgoing weights/bank\n(same RdBu_r scale as heatmap: white→dark red = larger |weight|)"),
        Line2D([0], [0], color="#cccccc", linewidth=1, linestyle="--", label="ring = block membership\n(radius, not distance)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=7.5)

    fig.suptitle(
        f"config={args.config.stem}  seed={args.seed}  "
        f"(inner ring = block {core}, red outline = misclassified bank)"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_path = args.out or (
        RESULTS_DIR / args.config.stem / "evaluation" / f"network_seed{args.seed}_period{period}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
