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
"""Aronow-Samii (2017) design-based Horvitz-Thompson volume spillover (PAPER section 6).

Fixed pre-treatment CANONICAL interlock graph; exposure = >=1 treated neighbor.
Estimand: spillover on UNTREATED firms, c01 (untreated+exposed) vs c00
(untreated+unexposed), on asinh insider-sale volume. Inference: Bernoulli
randomization inference (no parallel-trends assumption).

This is the final arbiter on the VOLUME question and returns a NULL — the paper's
secondary result. Re-run on the canonical 4,392-node graph (consistent with the
decomposition) rather than the earlier denser graph.
    uv run python -m open_aether.models.ht_spillover
"""
from __future__ import annotations
import numpy as np
import pandas as pd
# --- standalone bootstrap: make `open_aether` importable when run as a loose
# --- script (uv run --script); harmless no-op under `python -m open_aether...`.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from open_aether._spillover_common import (
    CUT, OUT_LO, OUT_HI, SEED, NEVER,
    load_panel, firm_frame, load_canonical_graph, align, strata_labels,
)

B = 5000                    # randomization-inference draws
WINSOR = 0.99               # cap HT weights at this quantile (positivity guard)
rng = np.random.default_rng(SEED)

# ---- firm-level: treatment Z, outcome Y, pre-activity -------------------
panel = load_panel()
firm = firm_frame(panel, actcol="n_tx")
panel["asinh_sale"] = np.arcsinh(pd.to_numeric(panel["sale_shr"], errors="coerce").fillna(0.0))
Yser = (panel[panel["per"].between(OUT_LO, OUT_HI)].groupby("cik")["asinh_sale"].mean())

nodes, A, deg = load_canonical_graph()
Z = ((align(firm["adopt_per"], nodes, NEVER) <= CUT)).astype(float).to_numpy()
Y = align(Yser, nodes, np.nan).to_numpy()
preact = align(firm["pre_act"], nodes, 0.0).to_numpy()
has_y = ~np.isnan(Y)
N = len(nodes)
print(f"networked firms: {N}; with outcome: {int(has_y.sum())}; treated Z=1: {int(Z.sum())}")

# ---- nonparametric propensity p_i: adoption rate in degree x pre-act cells
labs = strata_labels(deg, preact, 4)
cell = pd.Series(labs)
p = cell.map(pd.Series(Z).groupby(labs).mean()).to_numpy().astype(float)
p = np.clip(p, 1e-4, 1 - 1e-4)
print(f"propensity p: mean {p.mean():.3f}, range [{p.min():.3f},{p.max():.3f}]")

# ---- analytic exposure probabilities (independent Bernoulli design) -----
logq = np.log1p(-p)                        # log(1-p_j)
prod = np.exp(A @ logq)                    # Prod_{j in N(i)} (1-p_j)
pi00 = (1 - p) * prod                      # untreated & no treated neighbor
pi01 = (1 - p) * (1 - prod)                # untreated & >=1 treated neighbor


def hajek_contrast(Zvec):
    """Hajek-normalized E[Y|c01]-E[Y|c00] with design weights, winsorized.
    Only firms with an observed outcome enter the means."""
    exp_ = (A @ Zvec) > 0                  # >=1 treated neighbor
    unt = Zvec < 0.5
    out = {}
    for cond, mask, pr in (("c01", unt & exp_, pi01), ("c00", unt & ~exp_, pi00)):
        m = mask & (pr > 1e-6) & has_y
        w = 1.0 / pr[m]
        cap = np.quantile(w, WINSOR); w = np.minimum(w, cap)
        out[cond] = np.sum(w * Y[m]) / np.sum(w)
    return out["c01"] - out["c00"], out


tau, parts = hajek_contrast(Z)
obs01 = int(((Z < .5) & ((A @ Z) > 0) & has_y).sum())
obs00 = int(((Z < .5) & ((A @ Z) == 0) & has_y).sum())
naive = (Y[(Z < .5) & ((A @ Z) > 0) & has_y].mean()
         - Y[(Z < .5) & ((A @ Z) == 0) & has_y].mean())
print(f"\nobserved cells: c01(untreated+exposed)={obs01}, c00(untreated+unexposed)={obs00}")
print(f"Hajek means: c01={parts['c01']:.3f}  c00={parts['c00']:.3f}")
print(f"naive diff (unweighted): {naive:+.3f}")
print(f"HT/Hajek spillover tau-hat (c01-c00): {tau:+.4f}")

# ---- Bernoulli randomization inference (sharp null: exposure irrelevant) -
null = np.empty(B)
for b in range(B):
    Zb = (rng.random(N) < p).astype(float)
    null[b], _ = hajek_contrast(Zb)
p_two = (np.sum(np.abs(null) >= abs(tau)) + 1) / (B + 1)
print(f"RI null: mean {null.mean():+.4f}, sd {null.std():.4f}")
print(f"RI two-sided p-value: {p_two:.4f}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=60, color="lightsteelblue")
    ax.axvline(tau, color="crimson", lw=2, label=f"observed {tau:+.3f} (p={p_two:.3f})")
    ax.set_xlabel("c01-c00 spillover contrast"); ax.set_ylabel("RI draws")
    ax.legend(); fig.tight_layout()
    from open_aether._spillover_common import BASE
    fig.savefig(f"{BASE}/ht_spillover_ri.png", dpi=150)
    print("saved", f"{BASE}/ht_spillover_ri.png")
except Exception as ex:
    print("plot skipped:", ex)
