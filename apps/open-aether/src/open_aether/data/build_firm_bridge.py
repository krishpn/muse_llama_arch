#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "pymongo",
#     "pyarrow",
# ]
# ///
"""Build a date-aware cik->permno crosswalk from the CCM link table and
report what fraction of the SEC firm universe maps to a PERMNO."""
import glob, os, sys
import pandas as pd
from pymongo import MongoClient


ROOT   = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                            else "/home/krishna/Documents/sec_data/sec_insider/extracted")
BASE   = os.path.join(ROOT, "_parsed")     # parse outputs live under <root>/_parsed
assert os.path.isdir(BASE), f"no _parsed dir under {ROOT} -- pass the correct root as argv[1]"
client = MongoClient("mongodb://localhost:30017/")
db = client["insTrader"]

# ---- 1. pull the tiny (~32k row) CCM link table into pandas ----
link = pd.DataFrame(list(db.CRSP_compustat_mrg_db_lnk_tbl.find({}, {"_id": 0})))
print("link cols:", list(link.columns))
print("link rows:", f"{len(link):,}")

# normalize the mixed int/string cik -> Int64
link["cik_int"] = pd.to_numeric(link["cik"], errors="coerce").astype("Int64")

# dates -> int YYYYMMDD; non-numeric / 'E' / blank end-date -> open (99999999)
link["LINKDT_i"]    = pd.to_numeric(link["LINKDT"], errors="coerce").fillna(0).astype("int64")
link["LINKENDDT_i"] = pd.to_numeric(link["LINKENDDT"], errors="coerce").fillna(99999999).astype("int64")

# keep only usable primary links IF those columns exist
for col, keep in (("LINKTYPE", {"LC", "LU"}), ("LINKPRIM", {"P", "C"})):
    if col in link.columns:
        vals = link[col].astype(str).str.strip()
        print(f"{col} dist:", vals.value_counts().to_dict())
        before = len(link)
        link = link[vals.isin(keep)]
        print(f"  {col} filter: {before:,} -> {len(link):,}")

link["LPERMNO"] = pd.to_numeric(link["LPERMNO"], errors="coerce").astype("Int64")
link = link.dropna(subset=["cik_int", "LPERMNO"])

xwalk = (link[["cik_int", "LPERMNO", "LINKDT_i", "LINKENDDT_i"]]
        .drop_duplicates()
        .rename(columns={"cik_int": "ISSUERCIK_INT"}))
xwalk.to_parquet(f"{BASE}/cik_permno_xwalk.parquet", index=False)
print(f"xwalk rows: {len(xwalk):,}  distinct cik: {xwalk['ISSUERCIK_INT'].nunique():,}")

# ---- 2. match rate against the SEC firm universe ----
edges = pd.concat(
    (pd.read_parquet(f, columns=["ISSUERCIK"]) for f in glob.glob(f"{BASE}/edges/*.parquet")),
    ignore_index=True,
)
sec_ciks = pd.to_numeric(edges["ISSUERCIK"], errors="coerce").dropna().astype("int64").unique()
linked   = set(xwalk["ISSUERCIK_INT"].dropna().astype("int64").tolist())
hit      = sum(c in linked for c in sec_ciks)
print(f"SEC distinct firm CIKs : {len(sec_ciks):,}")
print(f"  with a CCM PERMNO     : {hit:,} ({hit / len(sec_ciks):.1%})")