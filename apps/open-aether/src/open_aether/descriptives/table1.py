#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "pyarrow",
# ]
# ///
"""Table 1 summary statistics for the paper (PAPER section 3 / Table 1).

Reporting only — produces NO estimand. Consumes the panel + canonical graph via
open_aether._spillover_common (exactly the objects the models use) and prints four
panels, also written to `_parsed/table1_summary.md` and `_parsed/table1_summary.csv`:

  Panel A  sample sizes + interlock-graph structure (nodes, edges, degree, density)
  Panel B  adoption status on the networked universe (in-window / pre / never)
  Panel C  firm characteristics (degree, pre-mandate activity) by adoption status
  Panel D  adoption counts by quarter over the 2023Q2..2026Q2 window

Run:  uv run python -m open_aether.descriptives.table1
"""
from __future__ import annotations
import numpy as np
import pandas as pd
# --- standalone bootstrap: make `open_aether` importable when run as a loose
# --- script (uv run --script); harmless no-op under `python -m open_aether...`.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from open_aether._spillover_common import (
    WLO, WHI, NEVER,
    load_panel, firm_frame, load_canonical_graph, align, BASE,
)


def qlabel(per: int) -> str:
    """Integer per -> 'YYYYQn' (inverse of per_index)."""
    return f"{2006 + per // 4}Q{per % 4 + 1}"


def _fmt(x, nd=1):
    return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


# ---- load the same objects the models use -------------------------------
panel = load_panel()
firm = firm_frame(panel, actcol="n_sales")           # pre_act = pre-mandate sale count
nodes, A, deg = load_canonical_graph()

# networked subset, aligned onto the canonical nodes
adoptv = align(firm["adopt_per"], nodes, NEVER).astype("int64").to_numpy()
preact = align(firm["pre_act"], nodes, 0.0).to_numpy()
N = len(nodes)
E = int(A.nnz // 2)

in_win = (adoptv >= WLO) & (adoptv <= WHI)
pre_win = adoptv < WLO                                 # adopted before the mandate window
never = adoptv == NEVER
density = 2.0 * E / (N * (N - 1))

blocks: list[str] = []


# ---- Panel A: sample + graph structure ----------------------------------
A_rows = [
    ("Firm-quarter observations in panel", _fmt(len(panel))),
    ("Distinct firms in panel", _fmt(panel["cik"].nunique())),
    ("Networked firms (nodes in 2022 interlock graph)", _fmt(N)),
    ("Interlock edges", _fmt(E)),
    ("Mean degree", _fmt(float(deg.mean()))),
    ("Median degree", _fmt(float(np.median(deg)))),
    ("Max degree", _fmt(int(deg.max()))),
    ("Graph density", _fmt(density, 5)),
]
pa = pd.DataFrame(A_rows, columns=["Statistic", "Value"])
blocks.append("**Panel A. Sample and interlock-graph structure**\n\n" + pa.to_markdown(index=False))


# ---- Panel B: adoption status (networked universe) -----------------------
B_rows = [
    ("Networked firms", N, "100.0"),
    ("Adopt in window (2023Q2-2026Q2)", int(in_win.sum()), f"{100*in_win.mean():.1f}"),
    ("Adopted before window", int(pre_win.sum()), f"{100*pre_win.mean():.1f}"),
    ("Never adopt", int(never.sum()), f"{100*never.mean():.1f}"),
]
pb = pd.DataFrame(B_rows, columns=["Group", "Firms", "Share (%)"])
pb["Firms"] = pb["Firms"].map(lambda v: f"{v:,}")
blocks.append("**Panel B. Adoption status on the networked universe**\n\n" + pb.to_markdown(index=False))


# ---- Panel C: characteristics by adoption status -------------------------
def summ(mask, var):
    v = var[mask]
    return [f"{v.mean():.2f}", f"{np.median(v):.1f}", f"{v.std():.2f}", f"{int(mask.sum()):,}"]


groups = [("In-window adopters", in_win), ("Never-adopters", never)]
C_rows = []
for gname, gmask in groups:
    d = summ(gmask, deg)
    p = summ(gmask, preact)
    C_rows.append((gname, d[0], d[1], d[2], p[0], p[1], d[3]))
pc = pd.DataFrame(C_rows, columns=[
    "Group", "Degree mean", "Degree median", "Degree sd",
    "Pre-act mean", "Pre-act median", "N"])
blocks.append(
    "**Panel C. Firm characteristics by adoption status** "
    "(degree = interlock degree; pre-act = total insider sales 2021Q1-2022Q4)\n\n"
    + pc.to_markdown(index=False))


# ---- Panel D: adoption counts by quarter ---------------------------------
win = adoptv[in_win]
cnt = pd.Series(win).value_counts().sort_index()
D_rows = []
cum = 0
for per in range(WLO, WHI + 1):
    c = int(cnt.get(per, 0))
    cum += c
    D_rows.append((qlabel(per), c, cum, f"{100*cum/in_win.sum():.1f}"))
pd_ = pd.DataFrame(D_rows, columns=["Quarter", "Adopters", "Cumulative", "Cum. share (%)"])
blocks.append("**Panel D. Adoption counts by quarter (in-window adopters)**\n\n"
              + pd_.to_markdown(index=False))


# ---- emit ----------------------------------------------------------------
doc = "# Table 1. Summary statistics\n\n" + "\n\n".join(blocks) + "\n"
print(doc)

md_path = BASE / "table1_summary.md"
md_path.write_text(doc)
# one tidy long CSV for downstream typesetting
pd.concat([
    pa.assign(panel="A"),
    pb.rename(columns={"Group": "Statistic", "Firms": "Value"}).assign(panel="B")[["Statistic", "Value", "panel"]],
], ignore_index=True).to_csv(BASE / "table1_summary.csv", index=False)
print(f"\nwrote {md_path}")
print(f"wrote {BASE / 'table1_summary.csv'}")
