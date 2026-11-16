#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "pyarrow",
# ]
# ///
"""Parse SEC Form 345 insider data sets -> (1) insider-firm edges, (2) 10b5-1-flagged transactions."""
import glob, os, sys
import pandas as pd


ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "/home/krishna/Documents/sec_data/sec_insider/extracted")
OUT  = os.path.join(ROOT, "_parsed")
os.makedirs(os.path.join(OUT, "edges"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tx"), exist_ok=True)


def rd(path):
    # TSV, no quoting, keep empty strings as-is, everything as text
    return pd.read_csv(path, sep="\t", dtype=str, quoting=3,
                        keep_default_na=False, na_values=[], on_bad_lines="warn")

quarters = sorted(glob.glob(os.path.join(ROOT, "*_form345")))
print(f"{len(quarters)} quarters found")
ROOT

rel_counts, aff_by_q = {}, {}

for q in quarters:
    qid = os.path.basename(q).replace("_form345", "")
    try:
        sub = rd(os.path.join(q, "SUBMISSION.tsv"))
        own = rd(os.path.join(q, "REPORTINGOWNER.tsv"))
        nd  = rd(os.path.join(q, "NONDERIV_TRANS.tsv"))
    except Exception as e:
        print(f"  !! {qid}: {e}"); continue


    sub_cols = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE",
                    "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL"]
    subk = sub[sub_cols].copy()
    subk["AFF10B5ONE"] = sub["AFF10B5ONE"] if "AFF10B5ONE" in sub.columns else ""
    subk["QUARTER"] = qid

    # relationship flags (verify encoding from the printed value_counts)
    own = own.copy()
    rel = own["RPTOWNER_RELATIONSHIP"].str.lower().fillna("")
    own["is_director"] = rel.str.contains("director")
    own["is_officer"]  = rel.str.contains("officer")
    own["is_tenpct"]   = rel.str.contains("tenpercent|ten percent|10%", regex=True)

    # (1) insider<->firm edges: one row per (owner, filing)
    edge = own[["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                "RPTOWNER_RELATIONSHIP", "is_director", "is_officer", "is_tenpct"]] \
            .merge(subk, on="ACCESSION_NUMBER", how="left")
    edge.to_parquet(os.path.join(OUT, "edges", f"{qid}.parquet"), index=False)

    # (2) transactions enriched with issuer + owner + 10b5-1 flag
    n_owners = own.groupby("ACCESSION_NUMBER").size().rename("N_OWNERS")
    owner_slim = own[["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP"]]
    tx = (nd.merge(subk, on="ACCESSION_NUMBER", how="left")
            .merge(owner_slim, on="ACCESSION_NUMBER", how="left")
            .merge(n_owners, on="ACCESSION_NUMBER", how="left"))
    keep = ["ACCESSION_NUMBER", "QUARTER", "FILING_DATE", "TRANS_DATE", "DOCUMENT_TYPE",
            "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL",
            "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "N_OWNERS",
            "TRANS_CODE", "TRANS_ACQUIRED_DISP_CD", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
            "SHRS_OWND_FOLWNG_TRANS", "DIRECT_INDIRECT_OWNERSHIP", "AFF10B5ONE"]
    tx[keep].to_parquet(os.path.join(OUT, "tx", f"{qid}.parquet"), index=False)

    # diagnostics
    for k, v in own["RPTOWNER_RELATIONSHIP"].value_counts().items():
        rel_counts[k] = rel_counts.get(k, 0) + int(v)
    aff_by_q[qid] = subk["AFF10B5ONE"].value_counts().to_dict()
    # print(f"  {qid}: {len(edge):>7} owners  {len(tx):>8} txns")

    # break

print("\n=== RPTOWNER_RELATIONSHIP encodings (top 20) ===")
for k, v in sorted(rel_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {v:>10}  {k!r}")
print("\n=== AFF10B5ONE values, 2022q4..2024q4 ===")


for qid in sorted(aff_by_q):
    if "2022q4" <= qid <= "2024q4":
        print(f"  {qid}: {aff_by_q[qid]}")