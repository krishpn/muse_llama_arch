#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "networkx",
#     "pyarrow",
# ]
# ///
"""From _parsed/edges/, build the insider-interlock firm-firm network per year and report density."""
import glob, os, sys, itertools
import pandas as pd
import networkx as nx

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "/home/krishna/Documents/sec_data/sec_insider/extracted")
EDG  = os.path.join(ROOT, "_parsed", "edges")
OUT  = os.path.join(ROOT, "_parsed", "interlock"); os.makedirs(OUT, exist_ok=True)
ROLE = "is_director"   # switch to None for ALL insiders, or "is_officer"

print(f"ROOT={ROOT}\nEDG={EDG}\nOUT={OUT}\nROLE={ROLE}")


edges = pd.concat(map(pd.read_parquet, sorted(glob.glob(os.path.join(EDG, "*.parquet")))),
                    ignore_index=True)
print("raw edge rows:", len(edges))
print("raw edge columns:", edges.columns.tolist())
print("raw edge sample:", edges.head(3).to_dict(orient="records"))
print("FILING_DATE sample:", edges["FILING_DATE"].dropna().head(5).tolist())


# robust year parse (handles YYYY-MM-DD or DD-MON-YYYY etc.)
yr = pd.to_datetime(edges["FILING_DATE"], errors="coerce", format="mixed")
edges["year"] = yr.dt.year
print("unparsed dates:", int(edges["year"].isna().sum()))


if ROLE:
    edges = edges[edges[ROLE] == True]
pf = (edges[["year", "RPTOWNERCIK", "ISSUERCIK"]]
    .dropna().astype({"year": "Int64"}).drop_duplicates())

rows = []
all_firm_edges = []
for y, g in pf.groupby("year"):
    person_firms = g.groupby("RPTOWNERCIK")["ISSUERCIK"].apply(lambda s: sorted(set(s)))
    n_persons, n_firms = len(person_firms), g["ISSUERCIK"].nunique()
    n_firms_per = person_firms.map(len)
    interlockers = int((n_firms_per >= 2).sum())

    # firm-firm projection: two firms linked if they share >=1 director that year
    fe = set()
    for firms in person_firms[n_firms_per >= 2]:
        fe.update(itertools.combinations(firms, 2))
    G = nx.Graph(); G.add_nodes_from(g["ISSUERCIK"].unique()); G.add_edges_from(fe)
    lcc = max((len(c) for c in nx.connected_components(G)), default=0)
    rows.append(dict(year=int(y), persons=n_persons, firms=n_firms,
                    interlockers=interlockers, firm_edges=G.number_of_edges(),
                    density=round(nx.density(G), 5),
                    components=nx.number_connected_components(G), largest_cc=lcc,
                    mean_firms_per_interlocker=round(float(n_firms_per[n_firms_per>=2].mean() or 0), 2)))
    for u, v in fe:
        all_firm_edges.append((int(y), u, v))

summary = pd.DataFrame(rows).sort_values("year")
summary.to_csv(os.path.join(OUT, "interlock_summary.csv"), index=False)
pd.DataFrame(all_firm_edges, columns=["year", "firm_u", "firm_v"]) \
.to_parquet(os.path.join(OUT, "firm_firm_edges.parquet"), index=False)

pd.set_option("display.width", 160, "display.max_rows", 100)
print("\n=== interlock network by year (role =", ROLE, ") ===")
print(summary.to_string(index=False))

