#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "matplotlib",
#     "pyarrow",
# ]
# ///
"""Network descriptives + intro figure for the paper (PAPER section 3 / Figure 1).

Reporting only — produces NO estimand; it DESCRIBES the clustering that Section 4
then decomposes. Consumes the panel + canonical graph via open_aether._spillover_common.
Four descriptive objects, saved to `_parsed/`:

  (a) degree distribution  (complementary CDF, log-log)              -> fig panel 1
  (b) adoption S-curve     (cumulative share of in-window adopters)  -> fig panel 2
  (c) exposed-vs-unexposed adoption hazard, by quarter               -> fig panel 3
  (d) spatial autocorrelation of adoption on the interlock graph:
      Moran's I of cumulative adoption + a permutation join-count
      z/p for adopter-adopter edges, by quarter                      -> fig panel 4

The (c) gap and the (d) Moran's I are the RAW clustering signal the paper attributes
~88-94% to homophily; this module only shows it, it does not claim contagion.

Run:  uv run python -m open_aether.descriptives.network_desc
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

B = 1000                                 # permutation draws for the join-count null
rng = np.random.default_rng(SEED)


def qlabel(per: int) -> str:
    return f"{2006 + per // 4}Q{per % 4 + 1}"


# ---- objects (same as the models) ---------------------------------------
panel = load_panel()
firm = firm_frame(panel, actcol="n_sales")
nodes, A, deg = load_canonical_graph()
adoptv = align(firm["adopt_per"], nodes, NEVER).astype("int64").to_numpy()
N = len(nodes)
W = float(A.nnz)                         # sum of (binary) edge weights = 2E
in_win = (adoptv >= WLO) & (adoptv <= WHI)
n_adopt = int(in_win.sum())
print(f"networked firms {N:,}  edges {int(A.nnz//2):,}  mean deg {deg.mean():.1f}  "
      f"in-window adopters {n_adopt:,}")


# ---- (a) degree distribution --------------------------------------------
d = np.sort(deg[deg > 0])
ccdf = 1.0 - np.arange(len(d)) / len(d)
print(f"\n[degree] min {int(d.min())}  median {int(np.median(d))}  "
      f"mean {d.mean():.1f}  p95 {int(np.quantile(d,0.95))}  max {int(d.max())}")


# ---- (b) adoption S-curve -----------------------------------------------
scurve = []
for t in range(WLO, WHI + 1):
    frac = float((adoptv[in_win] <= t).mean())
    scurve.append((t, frac))
scurve = pd.DataFrame(scurve, columns=["per", "cum_share"])


# ---- (c) exposed-vs-unexposed adoption hazard, by quarter ---------------
haz = []
for t in range(WLO, WHI + 1):
    risk = adoptv >= t                                      # not-yet-adopted at t
    if risk.sum() < 2:
        continue
    exposed = (A @ (adoptv <= t - 1).astype(float)) >= 1    # >=1 neighbor adopted by t-1
    y = (adoptv == t).astype(float)                         # adopts this quarter
    re, ru = risk & exposed, risk & ~exposed
    he = float(y[re].mean()) if re.sum() else np.nan
    hu = float(y[ru].mean()) if ru.sum() else np.nan
    haz.append((t, he, hu, int(re.sum()), int(ru.sum())))
haz = pd.DataFrame(haz, columns=["per", "haz_exposed", "haz_unexposed", "n_exp", "n_unexp"])
haz["gap"] = haz["haz_exposed"] - haz["haz_unexposed"]
print("\n[exposed-vs-unexposed adoption hazard, by quarter]")
print(haz.assign(quarter=haz["per"].map(qlabel))
      [["quarter", "haz_exposed", "haz_unexposed", "gap", "n_exp", "n_unexp"]]
      .to_string(index=False))


# ---- (d) spatial autocorrelation of adoption, by quarter ----------------
def morans_i(x: np.ndarray) -> float:
    xc = x - x.mean()
    den = float(xc @ xc)
    if den <= 0:
        return np.nan
    return (N / W) * float(xc @ (A @ xc)) / den


def joincount_z(xbin: np.ndarray):
    """Adopter-adopter edge count vs a label-permutation null; returns (BB, z, p)."""
    bb = 0.5 * float(xbin @ (A @ xbin))
    k = int(xbin.sum())
    if k < 2 or k > N - 2:
        return bb, np.nan, np.nan
    null = np.empty(B)
    base = np.zeros(N)
    base[:k] = 1.0
    for b in range(B):
        xb = base[rng.permutation(N)]
        null[b] = 0.5 * float(xb @ (A @ xb))
    mu, sd = null.mean(), null.std()
    z = (bb - mu) / sd if sd > 0 else np.nan
    p = (np.sum(null >= bb) + 1) / (B + 1)              # one-sided (clustering)
    return bb, z, p


auto = []
for t in range(WLO, WHI + 1):
    xbin = (adoptv <= t).astype(float)                 # cumulative adoption status
    mi = morans_i(xbin)
    bb, z, p = joincount_z(xbin)
    auto.append((t, int(xbin.sum()), mi, bb, z, p))
auto = pd.DataFrame(auto, columns=["per", "n_adopted", "morans_I", "BB_edges", "jc_z", "jc_p"])
print("\n[spatial autocorrelation of cumulative adoption, by quarter]")
print(auto.assign(quarter=auto["per"].map(qlabel))
      [["quarter", "n_adopted", "morans_I", "jc_z", "jc_p"]].to_string(index=False))


# ---- (e) transitivity / clustering (the Bramoullé exclusion test) -------
# The friends-of-friends IV (Section 5.1) needs network INTRANSITIVITY: two-hop
# neighbors must not themselves be one-hop neighbors. These statistics measure
# exactly how badly that fails on the interlock graph. All computed from the
# sparse adjacency A (no networkx), using diag(A^3)_i = 2 * (triangles through i).
A2 = A @ A
diagA3 = np.asarray((A2.multiply(A)).sum(axis=1)).ravel()   # (A^3)_ii
tri_total = diagA3.sum() / 6.0                               # number of triangles
triples = float((deg * (deg - 1)).sum())                    # ordered 2-paths = 2*C(deg,2) summed
transitivity = diagA3.sum() / triples if triples > 0 else np.nan   # global clustering
with np.errstate(invalid="ignore", divide="ignore"):
    local_c = np.where(deg >= 2, diagA3 / (deg * (deg - 1)), np.nan)
avg_local_c = float(np.nanmean(local_c))
# fraction of two-hop pairs that are ALSO directly linked (direct intransitivity read):
# among node pairs sharing >=1 common neighbor, the share that are themselves adjacent.
A2off = A2.copy(); A2off.setdiag(0); A2off.eliminate_zeros()
pairs_2hop = (A2off > 0)                                     # pairs with a common neighbor
n_pairs_2hop = int(pairs_2hop.nnz // 2)
n_pairs_2hop_linked = int((pairs_2hop.multiply(A)).nnz // 2) # ...that are also edges
frac_closed = n_pairs_2hop_linked / n_pairs_2hop if n_pairs_2hop else np.nan
er_clustering = 2.0 * (A.nnz // 2) / (N * (N - 1))          # ER benchmark = density
print("\n[transitivity / clustering — the Bramoullé exclusion diagnostic]")
print(f"  triangles                              {int(tri_total):,}")
print(f"  global transitivity (clustering coef)  {transitivity:.3f}")
print(f"  mean local clustering (deg>=2)         {avg_local_c:.3f}")
print(f"  Erdos-Renyi benchmark (=density)       {er_clustering:.5f}")
print(f"  clustering / ER ratio                  {transitivity / er_clustering:,.0f}x")
print(f"  pairs sharing a common neighbor        {n_pairs_2hop:,}")
print(f"  ...of which also directly linked       {n_pairs_2hop_linked:,}  ({100*frac_closed:.1f}%)")
pd.DataFrame([{
    "triangles": int(tri_total),
    "transitivity": transitivity,
    "mean_local_clustering": avg_local_c,
    "er_benchmark": er_clustering,
    "clustering_over_er": transitivity / er_clustering,
    "pairs_common_neighbor": n_pairs_2hop,
    "pairs_common_neighbor_linked": n_pairs_2hop_linked,
    "frac_two_hop_closed": frac_closed,
}]).to_csv(BASE / "desc_transitivity.csv", index=False)


# ---- save tidy data ------------------------------------------------------
scurve.assign(quarter=scurve["per"].map(qlabel)).to_csv(BASE / "desc_scurve.csv", index=False)
haz.assign(quarter=haz["per"].map(qlabel)).to_csv(BASE / "desc_hazard_gap.csv", index=False)
auto.assign(quarter=auto["per"].map(qlabel)).to_csv(BASE / "desc_autocorr.csv", index=False)


# ---- intro figure (2x2) --------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    ax[0, 0].loglog(d, ccdf, marker=".", ls="none", ms=3)
    ax[0, 0].set_xlabel("degree"); ax[0, 0].set_ylabel("P(deg >= x)")
    ax[0, 0].set_title("(a) interlock degree distribution (CCDF)")

    ax[0, 1].plot(scurve["per"], scurve["cum_share"], marker="o", ms=4)
    ax[0, 1].set_xlabel("quarter (per)"); ax[0, 1].set_ylabel("cumulative share of adopters")
    ax[0, 1].set_title("(b) 10b5-1 adoption S-curve")

    ax[1, 0].plot(haz["per"], haz["haz_exposed"], marker="o", ms=4, label=">=1 adopted neighbor")
    ax[1, 0].plot(haz["per"], haz["haz_unexposed"], marker="s", ms=4, label="no adopted neighbor")
    ax[1, 0].set_xlabel("quarter (per)"); ax[1, 0].set_ylabel("adoption hazard")
    ax[1, 0].set_title("(c) adoption hazard, exposed vs unexposed"); ax[1, 0].legend()

    ax[1, 1].plot(auto["per"], auto["morans_I"], marker="o", ms=4, color="crimson")
    ax[1, 1].axhline(0, lw=.8, color="k")
    ax[1, 1].set_xlabel("quarter (per)"); ax[1, 1].set_ylabel("Moran's I (cumulative adoption)")
    ax[1, 1].set_title("(d) spatial autocorrelation on interlocks")

    fig.tight_layout()
    fig.savefig(BASE / "fig1_network_descriptives.png", dpi=150)
    print(f"\nsaved {BASE / 'fig1_network_descriptives.png'}")
except Exception as ex:
    print("plot skipped:", ex)
