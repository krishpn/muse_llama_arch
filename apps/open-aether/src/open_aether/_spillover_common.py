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
"""Single source of truth for the 10b5-1 board-interlock spillover study.

Every analysis module (decomposition, adoption_diffusion, director_fe,
director_placebo, ht_spillover) imports config + loaders from here so that:

  * the CANONICAL graph is identical everywhere — fixed 2022 director-interlock
    projection from interlock/firm_firm_edges.parquet, year==2022, binarized,
    networked firms only (4,392 nodes / 12,761 edges / mean deg 5.8). Earlier
    copies of adoption_diffusion/ht_spillover drifted to a denser multi-year
    graph (2-hop deg ~185) which invalidated the Bramoulle IV diagnostic — the
    shared loader removes that drift by construction;
  * the `per` index convention is defined once: per = (year-2006)*4 + (quarter-1),
    so 2023Q2 = 69, 2022Q4 = 67, 2026Q2 = 81;
  * the randomization-inference statistic `ri_slope` is byte-identical across
    modules (the primary decomposition and the adoption_diffusion RI must agree).

Run `python -m open_aether._spillover_common` on the box as a self-test: it prints
every loader's stats and must reproduce the observed pooled slope +0.02978.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

# ---- config (one definition for the whole study) -----------------------
GRAPH_YEAR = 2022            # fixed pre-treatment interlock snapshot
WLO, WHI   = 69, 81          # risk / adoption window: 2023Q2 .. 2026Q2
PRE_LO, PRE_HI = 60, 67      # pre-mandate activity window: 2021Q1 .. 2022Q4
CUT        = 75              # ht_spillover: treated = first-adoption per <= CUT (2024Q4)
OUT_LO, OUT_HI = 76, 81      # ht_spillover outcome window: 2025Q1 .. 2026Q2
SEED       = 0
NEVER      = 9999            # sentinel adopt_per for never-adopters


def per_index(year: int, quarter: int) -> int:
    """(year, quarter in 1..4) -> integer period index; 2023Q2 -> 69."""
    return (int(year) - 2006) * 4 + (int(quarter) - 1)


# ---- base path (auto-detect; the box has several possible layouts) ------
_CANDS = [
    Path.home() / "Documents/sec_data/sec_insider/extracted/_parsed",
    Path.home() / "Documents/sec_insider/_parsed",
    Path.home() / "Documents/sec_insider/extracted/_parsed",
]


def base_dir() -> Path:
    for p in _CANDS:
        if (p / "panel_firm_quarter.parquet").exists():
            return p
    hits = glob.glob(str(Path.home() / "Documents/**/panel_firm_quarter.parquet"),
                     recursive=True)
    if not hits:
        raise FileNotFoundError("could not locate panel_firm_quarter.parquet under ~/Documents")
    return Path(hits[0]).parent


BASE = base_dir()


def _pick(cols, *needles):
    """First column whose lowercased name contains all `needles`."""
    for c in cols:
        l = c.lower()
        if all(n in l for n in needles):
            return c
    return None


# ---- panel: firm x quarter, with a robust integer `per` -----------------
def load_panel() -> pd.DataFrame:
    """Load panel_firm_quarter.parquet; guarantee int `cik` and int `per`.

    The period column is `per` if present, else `q` (pandas Period objects,
    possibly with NaNs, rendered like 'Period('2023Q2','Q-DEC')'). Both are
    handled; rows with an unparseable period are dropped.
    """
    panel = pd.read_parquet(BASE / "panel_firm_quarter.parquet")
    cik_col = _pick(panel.columns, "cik") or "cik"
    panel = panel.rename(columns={cik_col: "cik"})
    panel["cik"] = pd.to_numeric(panel["cik"], errors="coerce").astype("Int64")
    panel = panel.dropna(subset=["cik"])
    panel["cik"] = panel["cik"].astype("int64")

    percol = "per" if "per" in panel.columns else ("q" if "q" in panel.columns else None)
    if percol is None:
        percol = _pick(panel.columns, "per") or _pick(panel.columns, "quarter") or "q"
    num = pd.to_numeric(panel[percol], errors="coerce")
    if num.notna().mean() > 0.5:                       # already an integer index
        panel["per"] = num
    else:                                              # period strings 2023Q2 / 2023-Q2
        pp = pd.PeriodIndex(panel[percol].astype(str).str.replace("-", "", regex=False), freq="Q")
        panel["per"] = (pp.year - 2006) * 4 + (pp.quarter - 1)
    panel = panel.dropna(subset=["per"])
    panel["per"] = panel["per"].astype("int64")
    return panel


def firm_frame(panel: pd.DataFrame, actcol: str = "n_tx") -> pd.DataFrame:
    """Collapse the panel to one row per firm.

    Columns: adopt_per (first quarter any_10b5==1, else NEVER=9999),
             pre_act   (sum of `actcol` over PRE_LO..PRE_HI, 0 if none),
             deg_snap  (panel 'degree' at the 2022Q4 snapshot, else max; a rough
                        fallback only — prefer load_canonical_graph for analysis).

    `actcol` defaults to n_tx (total transactions). The decomposition module
    passes actcol='n_sales' to preserve its published strata ladder exactly.
    """
    flagcol = "any_10b5" if "any_10b5" in panel.columns else \
        [c for c in panel.columns if "10b5" in c.lower()][0]
    if actcol not in panel.columns:
        actcol = "n_sales" if "n_sales" in panel.columns else \
            [c for c in panel.columns if "sale" in c.lower()][0]

    adopt = panel.loc[panel[flagcol] == 1].groupby("cik")["per"].min()
    pre = panel[(panel["per"] >= PRE_LO) & (panel["per"] <= PRE_HI)].groupby("cik")[actcol].sum()

    firm = pd.DataFrame({"cik": panel["cik"].unique()}).set_index("cik")
    firm["adopt_per"] = adopt.reindex(firm.index).fillna(NEVER).astype("int64")
    firm["pre_act"] = pre.reindex(firm.index).fillna(0.0)

    gy_per = per_index(GRAPH_YEAR, 4)                  # 2022Q4
    if "degree" in panel.columns:
        degsnap = panel[panel["per"] == gy_per].set_index("cik")["degree"]
        if degsnap.empty:
            degsnap = panel.groupby("cik")["degree"].max()
        firm["deg_snap"] = pd.to_numeric(degsnap.reindex(firm.index),
                                         errors="coerce").fillna(0).astype("int64")
    else:
        firm["deg_snap"] = 0
    return firm


# ---- CANONICAL graph: fixed 2022 firm-firm interlock projection ---------
def load_canonical_graph():
    """Return (nodes, A, deg) for the fixed GRAPH_YEAR interlock projection.

    nodes : pd.Index of networked firm CIKs (sorted, unique)  -> 4,392
    A     : scipy.sparse CSR symmetric binary adjacency on `nodes`
    deg   : np.ndarray degree per node

    This is THE graph for every module. Align firm-level vectors onto `nodes`
    with align() before using A.
    """
    ff = pd.read_parquet(BASE / "interlock" / "firm_firm_edges.parquet")
    if "year" in ff.columns:
        ff = ff[ff["year"] == GRAPH_YEAR]
    u = pd.to_numeric(ff["firm_u"], errors="coerce")
    v = pd.to_numeric(ff["firm_v"], errors="coerce")
    e = pd.DataFrame({"u": u, "v": v}).dropna().astype("int64")
    e = e[e["u"] != e["v"]]
    nodes = pd.Index(sorted(set(e["u"]) | set(e["v"])))
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    ui = idx.reindex(e["u"]).to_numpy()
    vi = idx.reindex(e["v"]).to_numpy()
    n = len(nodes)
    A = sparse.coo_matrix(
        (np.ones(len(ui) * 2),
         (np.concatenate([ui, vi]), np.concatenate([vi, ui]))),
        shape=(n, n)).tocsr()
    A.data[:] = 1.0                                    # binarize (collapse multi-edges)
    deg = np.asarray(A.sum(1)).ravel()
    return nodes, A, deg


def two_hop_only(A):
    """Adjacency of pairs at graph distance EXACTLY 2 (binary CSR) + its degree.

    Used by the Bramoulle friends-of-friends instrument. On a dense, transitive
    interlock graph deg2 >> deg, which is exactly why the exclusion restriction
    fails — adoption_diffusion reports deg2 to document that.
    """
    W = (A @ A).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    onehop = A.copy()
    onehop.data[:] = 1.0
    W = W - W.multiply(onehop)                         # drop direct-edge positions
    W.eliminate_zeros()
    A2 = W.copy()
    A2.data[:] = 1.0
    deg2 = np.asarray(A2.sum(1)).ravel()
    return A2, deg2


def align(series, nodes, fill):
    """Reindex a cik-indexed Series (or dict) onto `nodes`, filling gaps."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return s.reindex(nodes).fillna(fill)


# ---- insider<->firm edges (for the director-interlock designs) ----------
def load_edges() -> pd.DataFrame:
    """Load edges/*.parquet and add detected _issuer/_owner/_year/_isdir columns."""
    efiles = sorted(glob.glob(str(BASE / "edges" / "*.parquet")))
    if not efiles:
        raise FileNotFoundError(f"no edge parquets under {BASE/'edges'}")
    ed = pd.concat([pd.read_parquet(f) for f in efiles], ignore_index=True)

    issuer_col = _pick(ed.columns, "issuer", "cik") or _pick(ed.columns, "issuer")
    owner_col = (_pick(ed.columns, "rptowner", "cik") or _pick(ed.columns, "owner", "cik")
                 or _pick(ed.columns, "rptowner") or _pick(ed.columns, "owner"))
    dir_col = _pick(ed.columns, "director") or _pick(ed.columns, "is_dir")
    year_col = _pick(ed.columns, "year")
    date_col = None if year_col else (_pick(ed.columns, "filing", "date") or _pick(ed.columns, "date"))
    if issuer_col is None or owner_col is None:
        raise RuntimeError(f"could not detect issuer/owner id columns in {list(ed.columns)}")

    ed = ed.copy()
    ed["_issuer"] = pd.to_numeric(ed[issuer_col], errors="coerce")
    ed["_owner"] = pd.to_numeric(ed[owner_col], errors="coerce")
    if year_col is not None:
        ed["_year"] = pd.to_numeric(ed[year_col], errors="coerce")
    else:
        ed["_year"] = pd.to_datetime(ed[date_col], format="mixed", errors="coerce").dt.year
    if dir_col is not None:
        dv = ed[dir_col]
        ed["_isdir"] = (dv.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
                        | (pd.to_numeric(dv, errors="coerce") == 1))
    else:
        ed["_isdir"] = True                            # no flag -> treat all as directors
    return ed


def director_memberships(universe: set, year: int = GRAPH_YEAR,
                         directors_only: bool = True) -> pd.DataFrame:
    """(dir, cik) memberships in `year` for issuers in `universe`."""
    ed = load_edges()
    m = ed["_year"].eq(year) & ed["_issuer"].isin(universe) & ed["_owner"].notna()
    if directors_only:
        m &= ed["_isdir"]
    mem = (ed.loc[m, ["_owner", "_issuer"]].dropna().astype("int64")
           .drop_duplicates().rename(columns={"_owner": "dir", "_issuer": "cik"}))
    return mem


def add_other_min(mem: pd.DataFrame, adopt_col: str = "adopt_per") -> pd.DataFrame:
    """For each (dir, firm), earliest adoption among the director's OTHER boards.

    Requires `mem` to already carry `adopt_col`. Leave-one-out: if the firm's own
    adoption is the unique earliest on that director, use the second-earliest.
    """
    g = mem.groupby("dir")[adopt_col]
    min1 = g.transform("min")
    cnt1 = g.transform(lambda s: (s == s.min()).sum())
    min2 = g.transform(lambda s: (np.sort(s.unique())[1] if s.nunique() > 1 else NEVER))
    mem = mem.copy()
    mem["other_min"] = np.where((mem[adopt_col] == min1) & (cnt1 == 1), min2, min1).astype("int64")
    return mem


# ---- the randomization-inference statistic (identical everywhere) -------
def ri_slope(A, adoptv, wlo: int = WLO, whi: int = WHI) -> float:
    """Pooled within-quarter slope of own-adoption on >=1-adopted-neighbor exposure.

    FWL within-quarter regression with quarter fixed effects only, over the risk
    set (not-yet-adopted) each quarter. This is the observed decomposition statistic
    and the adoption_diffusion RI statistic; it must return +0.02978 on the real data.
    """
    adoptv = np.asarray(adoptv)
    num = den = 0.0
    for t in range(wlo, whi + 1):
        risk = adoptv >= t
        if risk.sum() < 2:
            continue
        exposed = (A @ (adoptv <= t - 1).astype(float) >= 1).astype(float)
        y = (adoptv == t).astype(float)
        xr = exposed[risk] - exposed[risk].mean()
        yr = y[risk] - y[risk].mean()
        num += float(xr @ yr)
        den += float(xr @ xr)
    return num / den if den > 0 else np.nan


def strata_labels(deg, preact, K: int):
    """KxK quantile-bin label per node on (degree, pre-activity). K=1 -> all one stratum."""
    n = len(deg)
    if K == 1:
        return np.zeros(n, dtype=int)
    qd = pd.qcut(pd.Series(deg).rank(method="first"), K, labels=False)
    qp = pd.qcut(pd.Series(preact).rank(method="first"), K, labels=False)
    return (qd * K + qp).to_numpy()


# ---- self-test ----------------------------------------------------------
def _selftest():
    print(f"[common] BASE = {BASE}")
    panel = load_panel()
    print(f"[panel]  rows={len(panel):,}  firms={panel['cik'].nunique():,}  "
          f"per=[{panel['per'].min()},{panel['per'].max()}]")
    firm = firm_frame(panel)
    print(f"[firm]   universe={len(firm):,}  ever-adopt={(firm['adopt_per']<NEVER).sum():,}")
    nodes, A, deg = load_canonical_graph()
    print(f"[graph]  networked={len(nodes):,}  edges={int(A.nnz//2):,}  mean deg={deg.mean():.1f}")
    A2, deg2 = two_hop_only(A)
    print(f"[2hop]   firms w/ 2-hop-only nbrs={(deg2>0).sum():,}  "
          f"mean deg2={deg2[deg2>0].mean():.1f}")
    adoptv = align(firm["adopt_per"], nodes, NEVER).astype("int64").to_numpy()
    in_win = (adoptv >= WLO) & (adoptv <= WHI)
    print(f"[align]  in-window adopters={in_win.sum():,}  pre-window={(adoptv<WLO).sum():,}  "
          f"never={(adoptv==NEVER).sum():,}")
    obs = ri_slope(A, adoptv)
    print(f"[ri]     observed pooled within-quarter slope = {obs:+.5f}  "
          f"(expect +0.02978)")


if __name__ == "__main__":
    _selftest()
