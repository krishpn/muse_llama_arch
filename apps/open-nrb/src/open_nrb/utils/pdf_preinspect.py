#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pypdf>=5.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///

import os
import pypdf
import argparse
from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

KNOWN_BFIS = {
    "NBL","RBB","NABIL","NIMB","NIBL","SCBNL","HBL","NSBI","EBL","NIC","MBL",
    "Kumari","Laxmi","SBL","ADBNL","Global","Citizen","Citizens","Prime","NMB",
    "Prabhu","Sanima","Mahalaxmi","Narayani","Karnali","Shangrila","Excel",
    "Miteri","Muktinath","Garima","Jyoti","Lumbini","Kamana","Shine",
    "Central","Goodwill","Pokhara","Saptakoshi","Green","ICFC","Manjushree",
    "NBB","NBBL","BOK","NCC","Mega","Kist","Janata","Civil","Century","CTBNL","NIFRA",
}

def classify_page(text):
    low = text.lower()
    head = low[:200]
    if "aggregate" in head:
        return None  # overview/comparison table, not per-bank detail
    if "statement of assets and liabilities" in head:
        kind = "balance_sheet"
    elif "profit and loss statement" in head:
        kind = "profit_and_loss"
    elif "statement of credit" in head:
        kind = "statement_of_credit"
    else:
        return None

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bank_hits = sum(1 for l in lines if l in KNOWN_BFIS)
    if bank_hits < 2:
        return None  # title matched but no real per-bank table here
    return kind

pdf_dir= '/home/krishna/Documents/personal/neupane_article/network_papers/00_llama_arch/muse_llama_arch/apps/open-nrb/src/open_nrb/data/raw_pdfs/monthly_statistical_bfis'
records = []
for filename in sorted(os.listdir(pdf_dir)):
    if not filename.endswith(".pdf"):
        continue
    path = os.path.join(pdf_dir, filename)
    try:
        reader = pypdf.PdfReader(path)
    except Exception as e:
        print(f"ERROR opening {filename}: {e}")
        continue
    for idx in range(len(reader.pages)):
        txt = reader.pages[idx].extract_text() or ""
        kind = classify_page(txt)
        records.append({"source_file": filename, "page": idx, "kind": kind})

page_class = pd.DataFrame(records)

# per-file page ranges for each statement type
summary = (
    page_class.dropna(subset=["kind"])
    .groupby(["source_file", "kind"])["page"]
    .agg(lambda s: (s.min(), s.max(), len(s)))
    .unstack()
)
print(summary)

# sanity check: any file where classified ranges still overlap between statements?
overlap_rows = []
for f, grp in page_class.dropna(subset=["kind"]).groupby("source_file"):
    sets = {k: set(grp.loc[grp.kind == k, "page"]) for k in grp.kind.unique()}
    kinds = list(sets)
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            ov = sets[kinds[i]] & sets[kinds[j]]
            if ov:
                overlap_rows.append({"source_file": f, "a": kinds[i], "b": kinds[j], "overlap_pages": ov})
print(pd.DataFrame(overlap_rows))

