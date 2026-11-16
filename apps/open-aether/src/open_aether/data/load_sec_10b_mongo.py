#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pymongo",
#     "pyarrow",
# ]
# ///
"""Load the 3 parsed SEC parquet outputs into insTrader, adding int-typed
cik columns (*_INT) so they join cleanly to the CCM link table."""
import glob, os, sys
import numpy as np
import pandas as pd
from pymongo import MongoClient

ROOT   = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                              else "/home/krishna/Documents/sec_data/sec_insider/extracted")
BASE   = os.path.join(ROOT, "_parsed")     # parse outputs live under <root>/_parsed
assert os.path.isdir(BASE), f"no _parsed dir under {ROOT} -- pass the correct root as argv[1]"
client = MongoClient("mongodb://localhost:30017/")
db = client["insTrader"]

CIK_COLS = ("ISSUERCIK", "RPTOWNERCIK", "firm_u", "firm_v")

def _int_or_none(series):
    """string/NaN -> python int or None (pymongo-safe, no pd.NA)."""
    v = pd.to_numeric(series, errors="coerce")
    return [int(x) if pd.notna(x) else None for x in v]

def load(glob_pat, coll, chunk=50000, drop=True):
    files = sorted(glob.glob(glob_pat))
    if not files:
        print(f"!! no files match {glob_pat}"); return 0
    if drop:
        db[coll].drop()
    total = 0
    for fp in files:
        df = pd.read_parquet(fp)
        for c in CIK_COLS:
            if c in df.columns:
                df[c + "_INT"] = _int_or_none(df[c])
        # NaN floats are fine for pymongo; pd.NA is not -> already handled above
        recs = df.to_dict("records")
        for i in range(0, len(recs), chunk):
            db[coll].insert_many(recs[i:i + chunk], ordered=False)
        total += len(recs)
        print(f"  {os.path.basename(fp):28s} +{len(df):>9,}  (total {total:,})")
    print(f"== {coll}: {total:,} docs")
    return total

load(f"{BASE}/tx/*.parquet",                      "sec_insider_tx")
load(f"{BASE}/edges/*.parquet",                   "sec_insider_edges")
load(f"{BASE}/interlock/firm_firm_edges.parquet", "sec_firm_edges")

print("building indexes...")
db.sec_insider_tx.create_index("ISSUERCIK_INT")
db.sec_insider_tx.create_index("ACCESSION_NUMBER")
db.sec_insider_edges.create_index("ISSUERCIK_INT")
db.sec_insider_edges.create_index("RPTOWNERCIK")
db.sec_firm_edges.create_index([("year", 1)])
print("done.")
