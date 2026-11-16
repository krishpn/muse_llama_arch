"""Shared graph-layout helper for the descriptives figures.

Prefers igraph (force-directed layouts implemented in C — fast and cleaner cluster
separation than networkx's pure-Python spring_layout) and falls back to networkx
spring if igraph is not installed. Returns ({node: (x, y)}, engine_name), keyed by
the input graph's own node labels so the caller can map colors/sizes straight back.

Install the fast path on the box with:  uv add python-igraph
"""
from __future__ import annotations


def graph_layout(H, engine: str = "auto", seed: int = 0, iters: int = 500):
    """Lay out a networkx graph H. engine in {"auto","igraph","networkx"}.

    Returns (pos, engine_used). `pos` is {node_label: (x, y)}.
    """
    nodelist = list(H.nodes())
    if engine in ("auto", "igraph"):
        try:
            import igraph as ig
            import random as _r
            _r.seed(seed)
            idx = {n: i for i, n in enumerate(nodelist)}
            edges = [(idx[u], idx[v]) for u, v in H.edges()]
            g = ig.Graph(n=len(nodelist), edges=edges)
            lay = g.layout_fruchterman_reingold(niter=iters)
            co = lay.coords
            pos = {n: (float(co[idx[n]][0]), float(co[idx[n]][1])) for n in nodelist}
            return pos, "igraph-fr"
        except ImportError:
            if engine == "igraph":
                print("igraph not installed (`uv add python-igraph`); "
                      "falling back to networkx spring.")
    import networkx as nx
    pos = nx.spring_layout(H, seed=seed, iterations=max(60, iters // 8))
    return pos, "networkx-spring"
