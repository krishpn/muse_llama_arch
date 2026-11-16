#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""Build the firm x quarter analysis panel -> _parsed/panel_firm_quarter.parquet.

This is THE upstream builder every model reads. It was previously the first ~90
lines of models/event_study_panel.py (concatenated with several abandoned
volume event-studies); it has been extracted here, unchanged in logic, because
(a) it is a data-construction step, not a model, and (b) every analysis module
depends on its output. Run it first on the box:

    uv run python -m open_aether.data.build_panel

Outputs one row per (firm, quarter): transaction/sale counts, the own-10b5-1
adoption flag, and the firm-year interlock-neighbor exposure cells. `q` is a
pandas Period; the analysis `per` index = (year-2006)*4 + (quarter-1) is derived
downstream (see open_aether._spillover_common.per_index).
"""
import sys, glob, os
import numpy as np, pandas as pd

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                          else "/home/krishna/Documents/sec_data/sec_insider/extracted")
BASE = os.path.join(ROOT, "_parsed")

# ---------- 1. firm x quarter outcomes + own treatment ----------
tx = pd.concat(
    (pd.read_parquet(f, columns=["ISSUERCIK", "FILING_DATE", "TRANS_CODE",
                                 "TRANS_SHARES", "AFF10B5ONE"])
     for f in sorted(glob.glob(f"{BASE}/tx/*.parquet"))),
    ignore_index=True,
)
tx["cik"] = pd.to_numeric(tx["ISSUERCIK"], errors="coerce")
tx["q"] = pd.to_datetime(tx["FILING_DATE"], errors="coerce", format="mixed").dt.to_period("Q")
tx["is10b5"] = tx["AFF10B5ONE"].astype(str).str.lower().map({"1": 1, "true": 1, "0": 0, "false": 0})
tx["shares"] = pd.to_numeric(tx["TRANS_SHARES"], errors="coerce")
tx["is_sale"] = tx["TRANS_CODE"].astype(str).str.upper().eq("S")
tx["sale_shr"] = np.where(tx["is_sale"], tx["shares"].fillna(0.0), 0.0)
tx["sale_10b5"] = np.where(tx["is_sale"] & (tx["is10b5"] == 1), 1.0, 0.0)

tx = tx.dropna(subset=["cik", "q"])
tx["cik"] = tx["cik"].astype("int64")
fq = tx.groupby(["cik", "q"]).agg(
    n_tx=("cik", "size"),
    any_10b5=("is10b5", "max"),
    n_sales=("is_sale", "sum"),
    sale_shr=("sale_shr", "sum"),
    n_sale_10b5=("sale_10b5", "sum"),
).reset_index()

# first adoption quarter per firm
adopt = fq[fq["any_10b5"] == 1].groupby("cik")["q"].min()
adopt_map = adopt.to_dict()                       # cik -> Period

# ---------- 2. interlock-neighbor exposure, per firm-year graph ----------
fe = pd.read_parquet(f"{BASE}/interlock/firm_firm_edges.parquet")
fe["u"] = pd.to_numeric(fe["firm_u"], errors="coerce")
fe["v"] = pd.to_numeric(fe["firm_v"], errors="coerce")
fe = fe.dropna(subset=["u", "v", "year"]).astype({"u": "int64", "v": "int64", "year": "int64"})

NBY = {}                                          # year -> {cik: set(neighbors)}
for y, g in fe.groupby("year"):
    adj = {}
    for u, v in zip(g["u"].tolist(), g["v"].tolist()):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    NBY[int(y)] = adj

ciks = fq["cik"].to_numpy()
qs = fq["q"].tolist()
deg = np.zeros(len(fq), int); d_tr = np.zeros(len(fq), int); own = np.zeros(len(fq), int)
for k in range(len(fq)):
    cik, qq = int(ciks[k]), qs[k]
    nbrs = NBY.get(qq.year, {}).get(cik, ())
    deg[k] = len(nbrs)
    d_tr[k] = sum(1 for nb in nbrs if nb in adopt_map and adopt_map[nb] <= qq)
    own[k] = 1 if (cik in adopt_map and adopt_map[cik] <= qq) else 0

fq["degree"] = deg
fq["nbr_treated"] = d_tr
fq["nbr_frac"] = np.where(deg > 0, d_tr / np.maximum(deg, 1), np.nan)
fq["own_treated"] = own
fq["nbr_exposed"] = (d_tr > 0).astype(int)
fq["exposure"] = np.select(
    [(fq.own_treated == 1) & (fq.nbr_exposed == 1),
     (fq.own_treated == 1) & (fq.nbr_exposed == 0),
     (fq.own_treated == 0) & (fq.nbr_exposed == 1),
     (fq.own_treated == 0) & (fq.nbr_exposed == 0)],
    ["c11_both", "c10_own", "c01_spill", "c00_control"],
    default="none")

fq.to_parquet(f"{BASE}/panel_firm_quarter.parquet", index=False)

# ---------- 3. quick reads ----------
print("panel rows:", f"{len(fq):,}  (firms {fq['cik'].nunique():,})")
post = fq[(fq["q"] >= pd.Period('2023Q2')) & (fq["degree"] > 0)]
print("\nexposure-condition counts, post-2023Q2, networked firms:")
print(post["exposure"].value_counts())
print("\nadoption timing (first 10b5-1 quarter):")
print(adopt.value_counts().sort_index().loc[pd.Period('2022Q4'):pd.Period('2024Q4')])
