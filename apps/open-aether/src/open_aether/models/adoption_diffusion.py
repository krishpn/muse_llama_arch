#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "pyfixest",
#     "pyarrow",
# ]
# ///
"""Governance-practice diffusion + friends-of-friends IV diagnostic (PAPER section 5.1).

Does an interlock neighbor's 10b5-1 adoption raise a firm's OWN adoption hazard?
Discrete-time hazard on the CANONICAL fixed-2022 graph, hardened with (RI)
design-based randomization inference and (IV) the Bramoulle 2-hop-only instrument.

The headline here is diagnostic: on this dense, transitive interlock graph the
2-hop neighborhood dwarfs the 1-hop one (deg2 >> deg), so the friends-of-friends
exclusion restriction fails and the 2SLS estimate is an uninformative null. The
module PRINTS mean deg2 to document exactly that (draft section 5.1).

Graph, config and the RI statistic come from open_aether._spillover_common so this
runs on the SAME 4,392-node graph as the decomposition (earlier versions drifted
to a denser multi-year graph — that drift is removed by construction now).
    uv run python -m open_aether.models.adoption_diffusion
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyfixest as pf
# --- standalone bootstrap: make `open_aether` importable when run as a loose
# --- script (uv run --script); harmless no-op under `python -m open_aether...`.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from open_aether._spillover_common import (
    WLO, WHI, SEED, NEVER,
    load_panel, firm_frame, load_canonical_graph, two_hop_only, align,
    ri_slope, strata_labels,
)

B_RI = 2000
rng = np.random.default_rng(SEED)

# ---- firm-level adoption + pre-activity, aligned to the canonical graph --
panel = load_panel()
firm = firm_frame(panel, actcol="n_tx")
nodes, A, deg = load_canonical_graph()
A2, deg2 = two_hop_only(A)
print(f"[graph] networked firms = {len(nodes)}; mean deg {deg.mean():.2f}")
print(f"[2hop]  firms w/ 2-hop-only nbrs: {(deg2>0).sum()}; mean deg2 {deg2[deg2>0].mean():.1f} "
      f"(>> deg -> Bramoulle exclusion restriction fails)")

adoptv = align(firm["adopt_per"], nodes, NEVER).astype("int64").to_numpy()
preact = align(firm["pre_act"], nodes, 0.0).to_numpy()
lpre, ldeg = np.log1p(preact), np.log1p(deg)
print(f"[firm] ever-adopt in universe: {(adoptv < NEVER).sum()}")

# ---- discrete-time hazard panel (+ nbr_frac, exposed, z2_frac) ----------
blocks = []
for t in range(WLO, WHI + 1):
    ap = (adoptv <= t - 1).astype(float)
    nbr = A @ ap
    z2 = A2 @ ap
    in_risk = adoptv >= t
    if in_risk.sum() == 0:
        continue
    m = np.where(in_risk)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        z2f = np.where(deg2[m] > 0, z2[m] / np.maximum(deg2[m], 1), 0.0)
    blocks.append(pd.DataFrame({
        "cik": np.asarray(nodes)[m], "per": t,
        "adopt_now": (adoptv[m] == t).astype(float),
        "nbr_frac": nbr[m] / deg[m], "exposed": (nbr[m] >= 1).astype(float),
        "z2_frac": z2f, "ldeg": ldeg[m], "lpre": lpre[m],
    }))
d = pd.concat(blocks, ignore_index=True)
print(f"\nhazard panel: {len(d):,} risk cells; base hazard {d.adopt_now.mean():.4f}")

# ---- (1) reduced-form discrete-time hazard ------------------------------
print("\n=== reduced form: binary exposure (quarter FE, cluster firm) ===")
m2 = pf.feols("adopt_now ~ exposed + ldeg + lpre | per", d, vcov={"CRV1": "cik"})
print(m2.tidy().loc[["exposed", "ldeg", "lpre"], ["Estimate", "Std. Error", "Pr(>|t|)"]].to_string())

# ---- (2) design-based RI: permute adoption timing WITHIN size strata ----
labs = strata_labels(deg, preact, 4)
adopters = np.where(adoptv < NEVER)[0]
is_adopter = np.zeros(len(nodes), bool); is_adopter[adopters] = True
groups = []
for s in np.unique(labs):
    g = np.where((labs == s) & is_adopter)[0]
    if len(g) > 1:
        groups.append(g)

obs = ri_slope(A, adoptv)
null = np.empty(B_RI)
for b in range(B_RI):
    avp = adoptv.copy()
    for ix in groups:
        avp[ix] = avp[ix][rng.permutation(len(ix))]
    null[b] = ri_slope(A, avp)
p_ri = (np.sum(null >= obs) + 1) / (B_RI + 1)          # one-sided (contagion => positive)
print(f"\n=== design-based RI (permute timing within size strata) ===")
print(f"observed slope {obs:+.5f}; null mean {null.mean():+.5f} sd {null.std():.5f}")
print(f"RI one-sided p (contagion>0): {p_ri:.4f}")

# ---- (3) Bramoulle friends-of-friends IV --------------------------------
print("\n=== Bramoulle 2SLS: nbr_frac instrumented by 2-hop-only adoption ===")
try:
    fs = pf.feols("nbr_frac ~ z2_frac + ldeg + lpre | per", d, vcov={"CRV1": "cik"})
    fst = fs.tidy().loc["z2_frac"]
    print(f"first stage z2_frac: coef {fst['Estimate']:+.3f}  t {fst['Estimate']/fst['Std. Error']:.1f}")
    # modern pyfixest IV syntax: endogenous term inline with [ ... ]
    iv = pf.feols("adopt_now ~ ldeg + lpre + [nbr_frac ~ z2_frac] | per", d, vcov={"CRV1": "cik"})
    ivt = iv.tidy().loc["nbr_frac"]
    print(f"2SLS nbr_frac: coef {ivt['Estimate']:+.4f}  se {ivt['Std. Error']:.4f}  p {ivt['Pr(>|t|)']:.4f}")
except Exception as ex:
    print("IV step failed:", ex)
