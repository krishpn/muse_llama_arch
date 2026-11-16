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
"""Node-link plot of the interlock network, colored by adoption status (Figure 2).

Reporting only — produces NO estimand. Figure 1 (network_desc.py) is entirely
moment plots (CCDF, S-curve, hazard, Moran's I); this adds the one thing those
cannot show: the graph itself, with adopters drawn in so the clustering is visible.

IMPORTANT framing — the clustering this figure shows is the phenomenon the paper
EXPLAINS, not evidence of contagion. The decomposition (Section 4) attributes ~88-94%
of it to homophily. The caption must say so; a reader must not read "adopters sit
together" as "adoption spread between them."

To keep the figure honest as well as legible it prints, alongside the picture:
  - the largest connected component size (only the LCC is drawn),
  - the attribute assortativity of ever-adopted (do adopters link to adopters?),
  - the degree assortativity,
  - the adopter-adopter edge share vs the share expected under random mixing (p^2).

Run:  uv run python -m open_aether.descriptives.network_plot
"""
from __future__ import annotations
import numpy as np
import pandas as pd
# --- standalone bootstrap: make `open_aether` importable when run as a loose
# --- script (uv run --script); harmless no-op under `python -m open_aether...`.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from open_aether._spillover_common import (
    WLO, WHI, NEVER, SEED,
    load_panel, firm_frame, load_canonical_graph, align, BASE,
)

LAYOUT_ITERS = 500         # force-directed iterations (igraph FR; nx spring uses iters//8)
LAYOUT_ENGINE = "auto"     # "auto" (igraph if available) | "igraph" | "networkx"
DRAW_KCORE = 2             # draw the k-core only (drops degree-1 pendants/spokes); 0 = full LCC.
                           # Stats below are ALWAYS computed on the full LCC regardless.


# ---- objects (same as the models) ---------------------------------------
panel = load_panel()
firm = firm_frame(panel, actcol="n_sales")
nodes, A, deg = load_canonical_graph()
adoptv = align(firm["adopt_per"], nodes, NEVER).astype("int64").to_numpy()
N = len(nodes)

in_win = (adoptv >= WLO) & (adoptv <= WHI)
pre = adoptv < WLO
never = adoptv == NEVER
ever = ~never                                       # adopted at any time
p_ever = float(ever.mean())
print(f"networked firms {N:,}  ever-adopters {int(ever.sum()):,} ({100*p_ever:.1f}%)  "
      f"in-window {int(in_win.sum()):,}  pre {int(pre.sum()):,}  never {int(never.sum()):,}")


# ---- build graph, take the largest connected component ------------------
try:
    import networkx as nx
except ImportError:
    print("\nnetworkx not installed; `uv add networkx` (or run via the --script shebang).")
    _sys.exit(0)

G = nx.from_scipy_sparse_array(A)
# status: 0 never, 1 pre-window adopter, 2 in-window adopter
status = np.where(in_win, 2, np.where(pre, 1, 0))
nx.set_node_attributes(G, {i: int(s) for i, s in enumerate(status)}, "status")
nx.set_node_attributes(G, {i: int(e) for i, e in enumerate(ever.astype(int))}, "ever")

comps = sorted(nx.connected_components(G), key=len, reverse=True)
lcc_nodes = comps[0]
H = G.subgraph(lcc_nodes).copy()
print(f"\ncomponents {len(comps):,}  largest component {H.number_of_nodes():,} nodes "
      f"/ {H.number_of_edges():,} edges  "
      f"({100*H.number_of_nodes()/N:.1f}% of nodes)")


# ---- quantitative anchors (so the picture is not just decoration) -------
attr_assort = nx.attribute_assortativity_coefficient(H, "ever")
deg_assort = nx.degree_assortativity_coefficient(H)
# edge-type shares on the LCC vs random-mixing expectation
e_aa = e_an = e_nn = 0
for u, v in H.edges():
    a, b = H.nodes[u]["ever"], H.nodes[v]["ever"]
    if a and b:
        e_aa += 1
    elif a or b:
        e_an += 1
    else:
        e_nn += 1
E = H.number_of_edges()
pa = np.mean([H.nodes[n]["ever"] for n in H.nodes()])      # adopter share in LCC
print(f"\n[graph-level homophily anchors on the LCC]")
print(f"  attribute assortativity (ever-adopt)   {attr_assort:+.3f}")
print(f"  degree assortativity                   {deg_assort:+.3f}")
print(f"  adopter-adopter edge share (observed)  {e_aa / E:.3f}")
print(f"  ...expected under random mixing (p^2)  {pa ** 2:.3f}")
print(f"  edge mix  AA {e_aa:,}  AN {e_an:,}  NN {e_nn:,}")
pd.DataFrame([{
    "lcc_nodes": H.number_of_nodes(), "lcc_edges": E,
    "attr_assortativity_ever": attr_assort, "degree_assortativity": deg_assort,
    "adopter_share_lcc": pa,
    "edge_share_AA_obs": e_aa / E, "edge_share_AA_exp": pa ** 2,
    "edges_AA": e_aa, "edges_AN": e_an, "edges_NN": e_nn,
}]).to_csv(BASE / "desc_network_plot.csv", index=False)


# ---- layout + draw -------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # draw the k-core only (strips degree-1 pendants that otherwise form radial
    # spokes and obscure the core); stats above are still on the full LCC.
    D = nx.k_core(H, DRAW_KCORE) if DRAW_KCORE else H
    if D.number_of_nodes() == 0:
        D = H
    core_note = (f"  (drawing {DRAW_KCORE}-core: {D.number_of_nodes():,} nodes / "
                 f"{D.number_of_edges():,} edges; pendants dropped)") if DRAW_KCORE else ""
    print(f"\nlaying out {D.number_of_nodes():,} nodes...{core_note}")
    from open_aether.descriptives._layout import graph_layout
    pos, eng = graph_layout(D, engine=LAYOUT_ENGINE, seed=SEED, iters=LAYOUT_ITERS)
    print(f"  layout engine: {eng}")

    color = {0: "#cccccc", 1: "#f4a259", 2: "#c1121f"}      # never / pre / in-window
    node_c = [color[D.nodes[n]["status"]] for n in D.nodes()]
    node_s = [6 + 2.0 * D.degree(n) for n in D.nodes()]

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(D, pos, ax=ax, width=0.2, alpha=0.15, edge_color="#777777")
    nx.draw_networkx_nodes(D, pos, ax=ax, node_color=node_c, node_size=node_s,
                           linewidths=0.0).set_rasterized(True)
    ax.set_axis_off()
    core_title = f" ({DRAW_KCORE}-core)" if DRAW_KCORE else ""
    ax.set_title(f"Interlock network (largest component{core_title}), colored by 10b5-1 adoption status")
    legend = [
        Line2D([0], [0], marker="o", ls="none", mfc="#c1121f", mec="none", ms=8,
               label="adopts in window (2023Q2-2026Q2)"),
        Line2D([0], [0], marker="o", ls="none", mfc="#f4a259", mec="none", ms=8,
               label="adopted before window"),
        Line2D([0], [0], marker="o", ls="none", mfc="#cccccc", mec="none", ms=8,
               label="never adopts"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    out = BASE / "fig2_network_layout.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
except Exception as ex:
    print("plot skipped:", ex)
