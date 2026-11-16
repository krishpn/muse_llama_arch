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
"""Immortal-time PLACEBO for the within-director FE hazard (PAPER section 5.2).

Claim: the within-director FE coefficient on exposed_d (observed ~ -0.18, see
models/director_fe.py) is a MECHANICAL artifact of an absorbing exposure covariate
(survivorship / immortal time), not contagion. Test: randomize WHICH adopter gets
WHICH adoption timing, rebuild the design, re-estimate. If the FE coef stays ~ -0.18
under random timing, it is mechanical. (A genuine contagion signal would collapse.)

Loaders/config/graph from open_aether._spillover_common.
    uv run python -m open_aether.models.director_placebo
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
    GRAPH_YEAR, WLO, WHI, NEVER, SEED,
    load_panel, firm_frame, load_canonical_graph, director_memberships,
)

MIN_BOARDS = 2
B = 50

# ---- firm adoption + canonical-graph degree -----------------------------
panel = load_panel()
firm = firm_frame(panel, actcol="n_tx")
nodes, A, deg = load_canonical_graph()
degS = pd.Series(deg, index=nodes)
firm["deg"] = pd.to_numeric(degS.reindex(firm.index), errors="coerce").fillna(0).astype("int64")
universe = set(firm.index.tolist())

# ---- director<->firm memberships (GRAPH_YEAR, directors, >=2 boards) ----
mem = director_memberships(universe, GRAPH_YEAR)
mem = mem.merge(firm[["deg"]], left_on="cik", right_index=True, how="left")
sz = mem.groupby("dir")["cik"].transform("size")
mem = mem[sz >= MIN_BOARDS].copy()
print(f"[mem] memberships={len(mem)}  directors={mem['dir'].nunique()}  firms={mem['cik'].nunique()}")

CIKv = mem["cik"].to_numpy(); DEGv = mem["deg"].to_numpy()
dir_codes = pd.factorize(mem["dir"].to_numpy())[0]       # dense director ids


# ---- within-director FE coef for a given firm->adopt_per map ------------
def within_dir_coef(adopt_by_cik: dict) -> float:
    ap = np.array([adopt_by_cik[c] for c in CIKv], dtype="int64")
    df = pd.DataFrame({"dir": dir_codes, "ap": ap})
    g = df.groupby("dir")["ap"]
    min1 = g.transform("min").to_numpy()
    cnt1 = g.transform(lambda s: (s == s.min()).sum()).to_numpy()
    min2 = g.transform(lambda s: (np.sort(s.unique())[1] if s.nunique() > 1 else NEVER)).to_numpy()
    other_min = np.where((ap == min1) & (cnt1 == 1), min2, min1).astype("int64")
    keep = DEGv > 0                        # focal firm must be networked
    ap_k, om_k, dir_k, cik_k = ap[keep], other_min[keep], dir_codes[keep], CIKv[keep]
    rows = []
    for t in range(WLO, WHI + 1):
        ir = ap_k >= t
        if not ir.any():
            continue
        rows.append(pd.DataFrame({
            "adopt_now": (ap_k[ir] == t).astype(int),
            "exposed_d": (om_k[ir] <= t - 1).astype(int),
            "dirid": dir_k[ir], "cik": cik_k[ir], "per": t}))
    d = pd.concat(rows, ignore_index=True)
    fit = pf.feols("adopt_now ~ exposed_d | dirid + per", d, vcov={"CRV1": "cik"})
    return float(fit.coef()["exposed_d"])


# ---- observed ------------------------------------------------------------
obs_map = firm["adopt_per"].to_dict()
obs = within_dir_coef(obs_map)
print(f"\n[observed] within-director FE exposed_d coef = {obs:+.4f}")

# ---- placebo: shuffle adoption TIMING among in-window adopters ----------
ever = firm.index[(firm["adopt_per"] >= WLO) & (firm["adopt_per"] <= WHI)].to_numpy()
ever_pers = firm.loc[ever, "adopt_per"].to_numpy()
rng = np.random.default_rng(SEED)
placebo = np.empty(B)
for b in range(B):
    perm = ever_pers[rng.permutation(len(ever_pers))]
    mp = dict(obs_map)
    for c, pp in zip(ever, perm):
        mp[c] = int(pp)
    placebo[b] = within_dir_coef(mp)
pm, psd = placebo.mean(), placebo.std()
print(f"[placebo] random-timing null: mean {pm:+.4f}  sd {psd:.4f}  "
      f"range [{placebo.min():+.4f}, {placebo.max():+.4f}]")
z = (obs - pm) / psd if psd > 0 else np.nan
print(f"[placebo] observed vs null: z = {z:+.2f}  "
      f"(|z|<~2 => observed is INDISTINGUISHABLE from random timing => MECHANICAL artifact)")
print("[read] If the null mean is ~ -0.18 like the observed, the within-director FE effect")
print("       is immortal-time survivorship, NOT contagion -> report as a cautionary methods result.")
