#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "pyfixest",
#     "matplotlib",
#     "pyarrow",
# ]
# ///
"""DEPRECATED / archived — insider-sale VOLUME event-studies.

   >>> Not part of the current paper's live pipeline. Kept for provenance. <<<

History / justification for keeping this around but out of the main flow:
These staggered/Sun-Abraham event-studies on asinh(sale volume) produced the
early "-0.42, 35-45% lower insider-sale volume, robust" result. That result did
NOT survive: it was an artifact of (a) baselining against the contaminated
2023Q1 transition quarter (per==68) and (b) Goodman-Bacon/TWFE cohort-weighting.
Dropping per==68 and re-basing to rel=-2 FLATTENED the pre-trends (good) but
also COLLAPSED the effect (SA post-onset ATT -0.419 -> -0.133, mostly ns). The
design-based Horvitz-Thompson estimator (models/ht_spillover.py) is the final
arbiter on volume and returns a NULL. The paper therefore reports NO robust
volume spillover (draft section 6) and relies on ht_spillover.py, not this file.

The four blocks below are retained verbatim so the full event-study path that
led to that conclusion is reproducible:
  (A) static + plain-TWFE staggered event-study (writes event_study.png)
  (B) manual Sun-Abraham aggregated event-study (writes sunab_event_study.png)
  (C) dominant single-cohort event-study (writes event_study_domcohort.png)
  (D) re-based event-study: drop per==68, reference rel=-2 (the artifact fix)

The panel BUILDER that used to prefix this file now lives in
open_aether.data.build_panel. Run that first.
"""
import sys, os
import numpy as np, pandas as pd, pyfixest as pf
from scipy.stats import chi2

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                          else "/home/krishna/Documents/sec_data/sec_insider/extracted")
BASE = os.path.join(ROOT, "_parsed")

# ======================================================================
# (A) static c01/c10/c11 vs c00  +  plain-TWFE staggered event-study
# ======================================================================
fq = pd.read_parquet(f"{BASE}/panel_firm_quarter.parquet")
fq = fq[fq["degree"] > 0].copy()
fq["qnum"] = fq["q"].astype(str)
fq["asinh_sale"] = np.arcsinh(fq["sale_shr"])
fq["asinh_n"] = np.arcsinh(fq["n_sales"])      # de-skewed count

post = fq[fq["q"] >= pd.Period("2023Q2")].copy()
post["exposure"] = pd.Categorical(
    post["exposure"], categories=["c00_control", "c01_spill", "c10_own", "c11_both"])
m_n = pf.feols("asinh_n    ~ C(exposure) | cik + qnum", post, vcov={"CRV1": "cik"})
m_v = pf.feols("asinh_sale ~ C(exposure) | cik + qnum", post, vcov={"CRV1": "cik"})
print(pf.etable([m_n, m_v], type="md"))

first_exp = fq[fq["nbr_exposed"] == 1].groupby("cik")["q"].min()
fq = fq.merge(first_exp.rename("fe_q"), on="cik", how="left")
fq["rel"] = [(q.year - f.year) * 4 + (q.quarter - f.quarter) if pd.notna(f) else np.nan
             for q, f in zip(fq["q"], fq["fe_q"])]
fq["rel_c"] = fq["rel"].clip(-6, 6)

names, kmap = [], {}
for k in range(-6, 7):
    if k == -1:
        continue
    nm = f"ev_m{abs(k)}" if k < 0 else f"ev_p{k}"
    fq[nm] = (fq["rel_c"] == k).astype(int)
    names.append(nm); kmap[nm] = k
pre = [n for n in names if kmap[n] < 0]


def joint_pretrend(m):
    cf = m.coef(); V = pd.DataFrame(m._vcov, index=cf.index, columns=cf.index)
    b = cf.loc[pre].values; Vs = V.loc[pre, pre].values
    W = float(b @ np.linalg.inv(Vs) @ b)
    return W, chi2.sf(W, len(pre))


fits = {}
for y in ("asinh_n", "asinh_sale"):
    m = pf.feols(f"{y} ~ {' + '.join(names)} | cik + qnum", fq, vcov={"CRV1": "cik"})
    fits[y] = m
    W, p = joint_pretrend(m)
    print(f"\n=== {y} ===")
    print(m.tidy().loc[names, ["Estimate", "Std. Error", "Pr(>|t|)"]])
    print(f"joint pre-trend test (ev_m6..ev_m2 = 0):  chi2={W:.2f}, df={len(pre)}, p={p:.3f}")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for y, mk in (("asinh_sale", "o"), ("asinh_n", "s")):
        t = fits[y].tidy().loc[names]
        xs = [kmap[n] for n in names]
        ax.errorbar(xs, t["Estimate"], yerr=1.96 * t["Std. Error"],
                    marker=mk, capsize=2, label=y)
    ax.axhline(0, lw=.8, color="k"); ax.axvline(-0.5, ls="--", color="grey")
    ax.set_xlabel("quarters since neighbor's 10b5-1 adoption"); ax.set_ylabel("coef (asinh)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{BASE}/event_study.png", dpi=150)
    print(f"\nsaved {BASE}/event_study.png")
except Exception as e:
    print("plot skipped:", e)

# ======================================================================
# (B) manual Sun-Abraham aggregated event-study (asinh_sale)
# ======================================================================
fq = pd.read_parquet(f"{BASE}/panel_firm_quarter.parquet")
fq = fq[fq["degree"] > 0].copy()
fq["asinh_sale"] = np.arcsinh(fq["sale_shr"])
fq["per"] = fq["q"].apply(lambda p: (p.year - 2006) * 4 + (p.quarter - 1)).astype(int)

first_exp = fq[fq["nbr_exposed"] == 1].groupby("cik")["per"].min()
fq = fq.merge(first_exp.rename("cohort"), on="cik", how="left")  # NaN = never-exposed controls

fq["rel"] = fq["per"] - fq["cohort"]            # NaN for never-exposed
LO, HI = -6, 6
fq["relc"] = fq["rel"].clip(LO, HI)

coh_counts = fq.loc[fq["cohort"].notna()].groupby("cohort")["cik"].nunique()
good_cohorts = coh_counts[coh_counts >= 20].index
print("cohorts kept:", len(good_cohorts), "of", coh_counts.size)

terms = []                                        # (name, cohort_float, rel)
for c in good_cohorts:
    for l in range(LO, HI + 1):
        if l == -1:                               # reference period
            continue
        nm = f"g{int(c)}_r{'m' if l < 0 else 'p'}{abs(l)}"
        fq[nm] = ((fq["cohort"] == c) & (fq["relc"] == l)).astype(int)
        terms.append((nm, c, l))

rhs = " + ".join(t[0] for t in terms)
m = pf.feols(f"asinh_sale ~ {rhs} | cik + per", fq, vcov={"CRV1": "cik"})

cf = m.coef(); V = pd.DataFrame(m._vcov, index=cf.index, columns=cf.index)
wshare = coh_counts.loc[good_cohorts] / coh_counts.loc[good_cohorts].sum()

rows = []
for l in range(LO, HI + 1):
    if l == -1:
        rows.append((l, 0.0, 0.0)); continue
    nms = [nm for (nm, c, ll) in terms if ll == l and nm in cf.index]
    cs = [c for (nm, c, ll) in terms if ll == l and nm in cf.index]
    if not nms:
        rows.append((l, np.nan, np.nan)); continue
    w = wshare.loc[cs].values; w = w / w.sum()
    est = float(w @ cf.loc[nms].values)
    se = float(np.sqrt(w @ V.loc[nms, nms].values @ w))
    rows.append((l, est, se))

agg = pd.DataFrame(rows, columns=["rel", "estimate", "se"])
agg["t"] = agg["estimate"] / agg["se"].replace(0, np.nan)
print("\n=== Sun-Abraham aggregated event-study (asinh_sale) ===")
print(agg.to_string(index=False))
post = agg[agg["rel"] >= 0]
print(f"\nSA post-onset ATT (avg l>=0): {post['estimate'].mean():.3f}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(agg["rel"], agg["estimate"], yerr=1.96 * agg["se"], marker="o", capsize=2)
    ax.axhline(0, lw=.8, color="k"); ax.axvline(-0.5, ls="--", color="grey")
    ax.set_xlabel("quarters since neighbor's 10b5-1 adoption")
    ax.set_ylabel("asinh(sale volume), SA-weighted"); fig.tight_layout()
    fig.savefig(f"{BASE}/sunab_event_study.png", dpi=150)
    print(f"saved {BASE}/sunab_event_study.png")
except Exception as e:
    print("plot skipped:", e)

# ======================================================================
# (C) dominant single-cohort event-study vs never-exposed
# ======================================================================
fq = pd.read_parquet(f"{BASE}/panel_firm_quarter.parquet")
fq = fq[fq["degree"] > 0].copy()
fq["asinh_sale"] = np.arcsinh(fq["sale_shr"])
fq["per"] = fq["q"].apply(lambda p: (p.year - 2006) * 4 + (p.quarter - 1)).astype(int)

first_exp = fq[fq["nbr_exposed"] == 1].groupby("cik")["per"].min()
fq = fq.merge(first_exp.rename("cohort"), on="cik", how="left")

coh_counts = fq[fq["cohort"].notna()].groupby("cohort")["cik"].nunique()
dom = coh_counts.idxmax()
dom_q = pd.Period(year=2006 + int(dom) // 4, quarter=int(dom) % 4 + 1, freq="Q")
print(f"dominant cohort = per {int(dom)} = {dom_q}  ({coh_counts.loc[dom]} firms)")

sub = fq[(fq["cohort"] == dom) | (fq["cohort"].isna())].copy()
sub["treat"] = (sub["cohort"] == dom).astype(int)
sub["rel"] = np.where(sub["treat"] == 1, sub["per"] - dom, np.nan)
sub["relc"] = pd.Series(sub["rel"]).clip(-6, 6)

names, kmap = [], {}
for k in range(-6, 7):
    if k == -1:
        continue
    nm = f"ev_m{abs(k)}" if k < 0 else f"ev_p{k}"
    sub[nm] = ((sub["treat"] == 1) & (sub["relc"] == k)).astype(int)
    names.append(nm); kmap[nm] = k
pre = [n for n in names if kmap[n] < 0]

m = pf.feols(f"asinh_sale ~ {' + '.join(names)} | cik + per", sub, vcov={"CRV1": "cik"})
t = m.tidy().loc[names, ["Estimate", "Std. Error", "Pr(>|t|)"]]
print(t.to_string())

cf = m.coef(); V = pd.DataFrame(m._vcov, index=cf.index, columns=cf.index)
b = cf.loc[pre].values; W = float(b @ np.linalg.inv(V.loc[pre, pre].values) @ b)
print(f"\nsingle-cohort joint pre-trend test: chi2={W:.2f}, df={len(pre)}, p={chi2.sf(W, len(pre)):.3f}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xs = [kmap[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(xs, t["Estimate"], yerr=1.96 * t["Std. Error"], marker="o", capsize=2)
    ax.axhline(0, lw=.8, color="k"); ax.axvline(-0.5, ls="--", color="grey")
    ax.set_xlabel(f"quarters since neighbor adoption ({dom_q} cohort)")
    ax.set_ylabel("asinh(sale volume)"); fig.tight_layout()
    fig.savefig(f"{BASE}/event_study_domcohort.png", dpi=150)
    print(f"saved {BASE}/event_study_domcohort.png")
except Exception as e:
    print("plot skipped:", e)

# ======================================================================
# (D) re-based event-studies: drop per==68 (2023Q1), reference rel=-2
#     This is the artifact fix that flattened pre-trends AND collapsed the effect.
# ======================================================================
fq = pd.read_parquet(f"{BASE}/panel_firm_quarter.parquet")
fq = fq[fq["degree"] > 0].copy()
fq["asinh_sale"] = np.arcsinh(fq["sale_shr"])
fq["per"] = fq["q"].apply(lambda p: (p.year - 2006) * 4 + (p.quarter - 1)).astype(int)
first_exp = fq[fq["nbr_exposed"] == 1].groupby("cik")["per"].min()
fq = fq.merge(first_exp.rename("cohort"), on="cik", how="left")

DIRTY = 68  # 2023Q1 transition quarter
fq = fq[fq["per"] != DIRTY].copy()      # drop globally
REF = -2                                # new baseline (2022Q4 for the mandate cohort)


def make_dummies(df, mask_treat, rel_col, lo=-6, hi=6):
    """Batch-build clipped event-time dummies (omit REF) via concat (no fragmentation)."""
    relc = df[rel_col].clip(lo, hi)
    cols, names, kmap = {}, [], {}
    for k in range(lo, hi + 1):
        if k == REF:
            continue
        nm = f"ev_m{abs(k)}" if k < 0 else f"ev_p{k}"
        col = ((mask_treat) & (relc == k)).astype(int)
        if col.sum() == 0:            # empty (e.g. rel -1 now has no treated obs)
            continue
        cols[nm] = col; names.append(nm); kmap[nm] = k
    df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
    return df, names, kmap


def joint_pre(m, names, kmap):
    pre = [n for n in names if kmap[n] < 0]
    cf = m.coef(); V = pd.DataFrame(m._vcov, index=cf.index, columns=cf.index)
    b = cf.loc[pre].values
    W = float(b @ np.linalg.inv(V.loc[pre, pre].values) @ b)
    return W, len(pre), chi2.sf(W, len(pre)), pre


def plot(names, kmap, t, path, xlabel):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [kmap[n] for n in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(xs, t["Estimate"], yerr=1.96 * t["Std. Error"], marker="o", capsize=2)
        ax.axhline(0, lw=.8, color="k"); ax.axvline(REF + 0.5, ls="--", color="grey")
        ax.set_xlabel(xlabel); ax.set_ylabel("asinh(sale volume)"); fig.tight_layout()
        fig.savefig(path, dpi=150); print("saved", path)
    except Exception as e:
        print("plot skipped:", e)


# ---------- (D-A) single-cohort 2023Q2 vs never-exposed ----------
dom = 69
sub = fq[(fq["cohort"] == dom) | (fq["cohort"].isna())].copy()
sub["rel"] = np.where(sub["cohort"] == dom, sub["per"] - dom, -999)
sub, names, kmap = make_dummies(sub, sub["cohort"] == dom, "rel")
mA = pf.feols(f"asinh_sale ~ {' + '.join(names)} | cik + per", sub, vcov={"CRV1": "cik"})
tA = mA.tidy().loc[names, ["Estimate", "Std. Error", "Pr(>|t|)"]]
print("=== (D-A) single-cohort 2023Q2, ref=rel-2, 2023Q1 dropped ===")
print(tA.to_string())
W, df_, p, _ = joint_pre(mA, names, kmap)
print(f"joint pre-trend: chi2={W:.2f}, df={df_}, p={p:.3f}")
plot(names, kmap, tA, f"{BASE}/event_study_domcohort_rebased.png",
     "quarters since neighbor adoption (2023Q2 cohort, ref=-2)")

# ---------- (D-B) staggered Sun-Abraham, re-based ----------
fqs = fq[fq["cohort"].notna()].copy()
fqs["rel"] = (fqs["per"] - fqs["cohort"]).clip(-6, 6)
ctrl = fq[fq["cohort"].isna()].copy(); ctrl["rel"] = np.nan
coh_counts = fqs.groupby("cohort")["cik"].nunique()
keep = coh_counts[coh_counts >= 20].index
print(f"\ncohorts kept: {len(keep)}")

blocks, nms, cmap, lmap = {}, [], {}, {}
pool = pd.concat([fqs[fqs["cohort"].isin(keep)], ctrl], ignore_index=True)
for c in keep:
    for l in range(-6, 7):
        if l == REF:
            continue
        nm = f"g{int(c)}_r{'m' if l < 0 else 'p'}{abs(l)}"
        col = ((pool["cohort"] == c) & (pool["rel"] == l)).astype(int)
        if col.sum() == 0:
            continue
        blocks[nm] = col; nms.append(nm); cmap[nm] = c; lmap[nm] = l
pool = pd.concat([pool, pd.DataFrame(blocks, index=pool.index)], axis=1)
mB = pf.feols(f"asinh_sale ~ {' + '.join(nms)} | cik + per", pool, vcov={"CRV1": "cik"})
cf = mB.coef(); V = pd.DataFrame(mB._vcov, index=cf.index, columns=cf.index)
live = [n for n in nms if n in cf.index]           # survive multicollinearity drops
wshare = coh_counts.loc[keep] / coh_counts.loc[keep].sum()

rows = []
for l in range(-6, 7):
    if l == REF:
        rows.append((l, 0.0, 0.0)); continue
    sel = [n for n in live if lmap[n] == l]
    if not sel:
        continue
    cs = [cmap[n] for n in sel]
    w = wshare.loc[cs].values; w = w / w.sum()
    est = float(w @ cf.loc[sel].values)
    se = float(np.sqrt(w @ V.loc[sel, sel].values @ w))
    rows.append((l, est, se))
sa = pd.DataFrame(rows, columns=["rel", "estimate", "se"]).sort_values("rel")
sa["t"] = sa["estimate"] / sa["se"].replace(0, np.nan)
print("\n=== (D-B) Sun-Abraham aggregated, ref=rel-2, 2023Q1 dropped ===")
print(sa.to_string(index=False))
print(f"SA post-onset ATT (avg l>=0): {sa[sa.rel >= 0].estimate.mean():.3f}")
