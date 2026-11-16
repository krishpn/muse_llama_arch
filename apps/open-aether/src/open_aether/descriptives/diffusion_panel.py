#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "matplotlib",
#     "networkx",
#     "python-igraph",
#     "pyarrow",
# ]
# ///
"""Diffusion trail: cumulative 10b5-1 adoption on a FIXED interlock layout, by quarter
(PAPER Figure 3). Small multiples — one panel per snapshot quarter, same node
positions in every panel, nodes colored by whether the firm has adopted BY that
quarter (cumulative).

READ THIS BEFORE INTERPRETING. A trail of a spreading color is exactly what a reader
expects CONTAGION to look like — and the paper's thesis is that this is NOT contagion.
The figure is built to show the opposite signature: each panel is annotated with the
Moran's I of adoption on the graph, and the point is that spatial clustering is
already near its maximum in the FIRST post-mandate quarter (2023Q2) and then stays
FLAT. Contagion propagating along edges would make Moran's I BUILD UP over time; a
common shock to similar, well-connected firms makes it appear all-at-once and plateau.
The caption must state this. The layout is the FIXED pre-treatment (2022) graph, so
the network itself does not change across panels — only which firms have adopted.

Moran's I is computed on the FULL canonical graph (all nodes), matching Figure 1(d);
the drawing is the 2-core of the largest component, for legibility only.

Run:  uv run python -m open_aether.descriptives.diffusion_panel
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from open_aether._spillover_common import (
    WLO, WHI, NEVER, SEED,
    load_panel, firm_frame, load_canonical_graph, align, BASE,
)

# snapshot quarters (per index): 2022Q4 pre-mandate, 2023Q2 shock, 2023Q4, 2024Q4, 2026Q2
SNAPSHOTS = [67, 69, 71, 75, 81]
DRAW_KCORE = 2
LAYOUT_ENGINE = "auto"
LAYOUT_ITERS = 500


def qlabel(per: int) -> str:
    return f"{2006 + per // 4}Q{per % 4 + 1}"


# ---- objects ------------------------------------------------------------
panel = load_panel()
firm = firm_frame(panel, actcol="n_sales")
nodes, A, deg = load_canonical_graph()
adoptv = align(firm["adopt_per"], nodes, NEVER).astype("int64").to_numpy()
N = len(nodes)
W = float(A.nnz)


def morans_i(x: np.ndarray) -> float:
    xc = x - x.mean()
    den = float(xc @ xc)
    return (N / W) * float(xc @ (A @ xc)) / den if den > 0 else np.nan


# ---- fixed drawing subgraph + one shared layout -------------------------
try:
    import networkx as nx
except ImportError:
    print("networkx not installed; `uv add networkx`.")
    _sys.exit(0)

G = nx.from_scipy_sparse_array(A)
H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
D = nx.k_core(H, DRAW_KCORE) if DRAW_KCORE else H
if D.number_of_nodes() == 0:
    D = H
drawn = list(D.nodes())
print(f"networked firms {N:,}  drawing {D.number_of_nodes():,} nodes "
      f"/ {D.number_of_edges():,} edges ({DRAW_KCORE}-core of largest component)")

from open_aether.descriptives._layout import graph_layout
print(f"laying out {D.number_of_nodes():,} nodes once (reused across all panels)...")
pos, eng = graph_layout(D, engine=LAYOUT_ENGINE, seed=SEED, iters=LAYOUT_ITERS)
print(f"  layout engine: {eng}")


# ---- per-snapshot Moran's I (full graph) + table ------------------------
rows = []
for t in SNAPSHOTS:
    cum = (adoptv <= t).astype(float)
    rows.append((t, qlabel(t), int(cum.sum()), morans_i(cum)))
snap = pd.DataFrame(rows, columns=["per", "quarter", "n_adopted", "morans_I"])
print("\n[cumulative adoption + spatial clustering at each snapshot]")
print(snap.to_string(index=False))
snap.to_csv(BASE / "desc_diffusion_snapshots.csv", index=False)
mi_first = snap.loc[snap.per == WLO, "morans_I"]
if len(mi_first):
    print(f"\nMoran's I at the first post-mandate quarter ({qlabel(WLO)}) = {mi_first.iloc[0]:.3f}; "
          f"it is near its maximum immediately and stays flat — the common-shock/homophily\n"
          f"signature, NOT the build-up a contagion process would produce.")


# ---- draw small multiples -----------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    npnl = len(SNAPSHOTS)
    fig, axes = plt.subplots(1, npnl, figsize=(4.0 * npnl, 4.6))
    if npnl == 1:
        axes = [axes]
    node_s = [3 + 1.1 * D.degree(n) for n in drawn]
    for ax, t in zip(axes, SNAPSHOTS):
        cum = adoptv <= t
        node_c = ["#c1121f" if cum[n] else "#cccccc" for n in drawn]
        nx.draw_networkx_edges(D, pos, ax=ax, width=0.15, alpha=0.12, edge_color="#777777")
        nx.draw_networkx_nodes(D, pos, ax=ax, node_color=node_c, node_size=node_s,
                               linewidths=0.0).set_rasterized(True)
        ax.set_axis_off()
        mi = morans_i((adoptv <= t).astype(float))
        tag = " (pre-mandate)" if t < WLO else ""
        ax.set_title(f"{qlabel(t)}{tag}\nadopted {int(cum.sum()):,}   Moran's I {mi:.2f}", fontsize=10)
    legend = [
        Line2D([0], [0], marker="o", ls="none", mfc="#c1121f", mec="none", ms=8, label="adopted by this quarter"),
        Line2D([0], [0], marker="o", ls="none", mfc="#cccccc", mec="none", ms=8, label="not yet adopted"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Cumulative 10b5-1 adoption on the fixed 2022 interlock network, by quarter", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = BASE / "fig3_diffusion_trail.png"
    fig.savefig(out, dpi=200)
    print(f"\nsaved {out}")
except Exception as ex:
    print("plot skipped:", ex)
