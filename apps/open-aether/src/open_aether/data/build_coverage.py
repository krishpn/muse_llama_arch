#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "pyarrow",
# ]
# ///
"""How much of what we ACTUALLY analyze is CRSP-linked:
transaction-weighted, window-restricted, and network-restricted coverage."""
import sys, glob, os
import pandas as pd

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                        else "/home/krishna/Documents/sec_data/sec_insider/extracted")
BASE = os.path.join(ROOT, "_parsed")

xwalk  = pd.read_parquet(f"{BASE}/cik_permno_xwalk.parquet")
linked = set(xwalk["ISSUERCIK_INT"].dropna().astype("int64").tolist())

# ---- transaction-weighted coverage ----
tx = pd.concat(
    (pd.read_parquet(f, columns=["ISSUERCIK", "FILING_DATE"])
    for f in sorted(glob.glob(f"{BASE}/tx/*.parquet"))),
    ignore_index=True,
)
tx["cik"]     = pd.to_numeric(tx["ISSUERCIK"], errors="coerce").astype("Int64")
tx["year"]    = pd.to_datetime(tx["FILING_DATE"], errors="coerce", format="mixed").dt.year
tx["matched"] = tx["cik"].isin(linked)

print("== transaction-weighted firm coverage ==")
print(f"all txns        : {len(tx):>10,}   matched {tx['matched'].mean():6.1%}")
win = tx[tx["year"].between(2021, 2025)]
print(f"2021-2025 txns  : {len(win):>10,}   matched {win['matched'].mean():6.1%}")
rf  = win["cik"].dropna().unique()
hit = sum(c in linked for c in rf)
print(f"2021-2025 firms : {len(rf):>10,}   matched {hit:,} ({hit/len(rf):.1%})")

# ---- interlock-network coverage ----
fe = pd.read_parquet(f"{BASE}/interlock/firm_firm_edges.parquet")
def cov(df, lbl):
    f = pd.to_numeric(pd.concat([df["firm_u"], df["firm_v"]]),
                    errors="coerce").dropna().astype("int64").unique()
    h = pd.Series(f).isin(linked).sum()
    print(f"{lbl:22s}: {len(f):>10,}   matched {h:,} ({h/len(f):.1%})")

print("== interlock network coverage ==")
cov(fe,                               "network firms (all yr)")
cov(fe[fe["year"].between(2021, 2025)], "network firms 21-25")
