#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyfixest",
#     "pyarrow",
# ]
# ///
"""Within-director (shared-director FE) contagion design for 10b5-1 (PAPER section 5.2).

Mechanism: a director d sits on boards of firms {F_d}. Does board i adopt right
after d's OTHER board j adopts? Director FE absorbs the transmitting director's own
governance propensity (the main homophily channel), so the exposed_d coefficient is
identified off WITHIN-director variation in transmission timing.

The within-director coefficient (~ -0.18) is a MECHANICAL immortal-time artifact,
not a peer effect — see models/director_placebo.py for the permutation proof. This
module just estimates the pooled vs within-director coefficients.

Loaders/config/graph from open_aether._spillover_common.
    uv run python -m open_aether.models.director_fe
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
    GRAPH_YEAR, WLO, WHI, NEVER,
    load_panel, firm_frame, load_canonical_graph,
    director_memberships, add_other_min,
)

MIN_BOARDS = 2                 # a director must serve >=2 universe firms to transmit

# ---- firm-level adoption + controls + canonical-graph degree ------------
panel = load_panel()
firm = firm_frame(panel, actcol="n_tx")           # adopt_per (9999=never), pre_act
nodes, A, deg = load_canonical_graph()
degS = pd.Series(deg, index=nodes)
firm["deg"] = pd.to_numeric(degS.reindex(firm.index), errors="coerce").fillna(0).astype("int64")
universe = set(firm.index.tolist())
print(f"[firm] universe={len(universe)}  ever-adopters={(firm['adopt_per']<NEVER).sum()}  "
      f"networked(deg>0)={(firm['deg']>0).sum()}")

# ---- director<->firm memberships in GRAPH_YEAR --------------------------
mem = director_memberships(universe, GRAPH_YEAR)
print(f"[edges] director<->firm memberships in {GRAPH_YEAR}: {len(mem)} "
      f"({mem['dir'].nunique()} directors, {mem['cik'].nunique()} firms)")

# ---- earliest OTHER-board adoption per (dir, firm) ----------------------
mem = mem.merge(firm[["adopt_per", "deg", "pre_act"]], left_on="cik", right_index=True, how="left")
mem = mem.dropna(subset=["adopt_per"]); mem["adopt_per"] = mem["adopt_per"].astype("int64")

sz = mem.groupby("dir")["adopt_per"].transform("size")
mem = mem[sz >= MIN_BOARDS].copy()
mem = add_other_min(mem, "adopt_per")
mem = mem[mem["deg"] > 0].copy()
print(f"[mem] transmitting memberships (>= {MIN_BOARDS} boards, deg>0): {len(mem)} "
      f"across {mem['dir'].nunique()} directors")

# ---- build (dir, firm, quarter) discrete-time hazard panel --------------
cik_i, dir_d = mem["cik"].to_numpy(), mem["dir"].to_numpy()
adoptv, othmin = mem["adopt_per"].to_numpy(), mem["other_min"].to_numpy()
ldeg = np.log1p(mem["deg"].to_numpy().astype(float))
lpre = np.log1p(mem["pre_act"].to_numpy().astype(float))

rows = []
for t in range(WLO, WHI + 1):
    in_risk = adoptv >= t
    if not in_risk.any():
        continue
    rows.append(pd.DataFrame({
        "adopt_now": (adoptv[in_risk] == t).astype(int),
        "exposed_d": (othmin[in_risk] <= t - 1).astype(int),
        "dirid": dir_d[in_risk], "cik": cik_i[in_risk], "per": t,
        "ldeg": ldeg[in_risk], "lpre": lpre[in_risk],
    }))
d = pd.concat(rows, ignore_index=True)
print(f"\n[panel] dir-firm-quarter risk cells = {len(d)}  base hazard = {d['adopt_now'].mean():.4f}")
print(f"[panel] exposed_d share = {d['exposed_d'].mean():.3f}")

var = d.groupby("dirid")["exposed_d"].nunique()
print(f"[ident] directors with within-director variation in exposed_d: {(var==2).sum()} / {d['dirid'].nunique()} "
      f"(these identify the FE coefficient)")

print("\n=== (1) pooled, quarter FE only (no director FE) ===")
m1 = pf.feols("adopt_now ~ exposed_d + ldeg + lpre | per", d, vcov={"CRV1": "cik"})
print(m1.summary())
print("\n=== (2) WITHIN-DIRECTOR: director FE + quarter FE ===")
m2 = pf.feols("adopt_now ~ exposed_d | dirid + per", d, vcov={"CRV1": "cik"})
print(m2.summary())
print("\n=== (3) within-director + firm controls (ldeg,lpre) ===")
m3 = pf.feols("adopt_now ~ exposed_d + ldeg + lpre | dirid + per", d, vcov={"CRV1": "cik"})
print(m3.summary())

b1, b3 = m1.coef()["exposed_d"], m3.coef()["exposed_d"]
print(f"\n[read] exposed_d: pooled {b1:+.5f}  ->  within-director {b3:+.5f}  "
      f"(shrinkage {100*(1-b3/b1):.0f}% if same sign)")
print("[read] large negative within-director coef is immortal-time, NOT contagion "
      "(see director_placebo.py).")
