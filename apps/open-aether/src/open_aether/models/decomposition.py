#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "pyarrow",
# ]
# ///
"""Homophily-vs-contagion decomposition of 10b5-1 adoption (PAPER PRIMARY, section 4).

Design-based randomization inference: permute adoption TIMING within size strata
among in-window adopters. As strata get finer the null absorbs more homophily; the
residual (observed - null mean) is the contagion component net of that homophily.
The ladder shows what share of the raw association is homophily and whether the
contagion residual survives.

Config, the canonical 2022 graph, the RI statistic, and strata construction all
come from open_aether._spillover_common (single source of truth). Run on the box:
    uv run python -m open_aether.models.decomposition
Must reproduce: observed slope +0.02978; 4x4 primary ~93% homophily / ~7% contagion.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
# --- standalone bootstrap: make `open_aether` importable when run as a loose
# --- script (uv run --script); harmless no-op under `python -m open_aether...`.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from open_aether._spillover_common import (
    WLO, WHI, SEED, NEVER,
    load_panel, firm_frame, load_canonical_graph, align, ri_slope, strata_labels,
)

B = 2000                        # RI permutations per strata spec
LADDER = [1, 2, 4, 8]           # bins per dimension (K x K size strata)

# ---- data: adoption + pre-activity on the canonical graph node set ------
panel = load_panel()
# pre_act from n_sales here (not the default n_tx) to preserve the published ladder.
firm = firm_frame(panel, actcol="n_sales")
nodes, A, deg = load_canonical_graph()
print(f"[graph] networked firms = {len(nodes)}  edges = {int(A.nnz//2)}  mean deg = {deg.mean():.1f}")

adoptv = align(firm["adopt_per"], nodes, NEVER).astype("int64").to_numpy()
preact = align(firm["pre_act"], nodes, 0.0).to_numpy()
in_win = (adoptv >= WLO) & (adoptv <= WHI)
print(f"[firm] in-window adopters = {in_win.sum()}  pre-window = {(adoptv<WLO).sum()}  "
      f"never = {(adoptv==NEVER).sum()}")

obs = ri_slope(A, adoptv)
print(f"\n[observed] pooled within-quarter exposed->adopt slope = {obs:+.5f}\n")


# ---- strata builder: permutable adopters grouped by KxK size strata -----
def strata_groups(K: int):
    labs = strata_labels(deg, preact, K)
    pool = np.where(in_win)[0]
    groups = []
    for s in np.unique(labs[pool]):
        g = pool[labs[pool] == s]
        if len(g) >= 2:
            groups.append(g)
    return groups


# ---- run the ladder -----------------------------------------------------
rng = np.random.default_rng(SEED)
print(f"{'strata':>8} {'#str':>5} {'null_mean':>10} {'null_sd':>9} "
      f"{'residual':>9} {'homoph%':>8} {'contag%':>8} {'1-sided p':>9}")
for K in LADDER:
    groups = strata_groups(K)
    null = np.empty(B)
    base = adoptv.copy()
    for b in range(B):
        av = base.copy()
        for g in groups:
            av[g] = base[g][rng.permutation(len(g))]
        null[b] = ri_slope(A, av)
    nm, nsd = np.nanmean(null), np.nanstd(null)
    resid = obs - nm
    pval = (1 + np.sum(null >= obs)) / (B + 1)
    homoph = 100 * nm / obs
    contag = 100 * resid / obs
    print(f"{K}x{K:<6} {len(groups):>5} {nm:>10.5f} {nsd:>9.5f} "
          f"{resid:>9.5f} {homoph:>7.1f}% {contag:>7.1f}% {pval:>9.4f}")

print("\n[read] As strata get finer the null mean rises toward the observed slope")
print("       (homophily absorbed). Watch the contagion residual + its p-value:")
print("       stays significant at 8x8  -> robust small contagion net of homophily")
print("       collapses to ~0 / p>0.1   -> clustering is essentially pure homophily")
