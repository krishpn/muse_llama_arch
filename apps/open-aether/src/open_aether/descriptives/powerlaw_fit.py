#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "powerlaw",
# ]
# ///
"""Degree-distribution power-law fit for the APPENDIX (PAPER Appendix A, optional).

Reporting only — produces NO estimand and is NOT load-bearing for the homophily-vs-
contagion argument. The paper makes NO scale-free claim in the body; this module
exists so the appendix can state, honestly and quantitatively, whether a power law
even fits the interlock degree distribution better than a lognormal or exponential.

Method: Clauset-Shalizi-Newman (2009) discrete MLE — estimate the scaling exponent
alpha and the lower cutoff xmin by KS minimization, then compare the power-law fit
against lognormal and exponential alternatives by normalized log-likelihood ratio
(Vuong). A power-law claim is only credible if (i) the KS distance D is small, (ii)
a usable share of the mass sits in the tail x >= xmin, and (iii) the power law is
not beaten by lognormal (R < 0 or p large => cannot distinguish / lognormal wins).

Prefers the `powerlaw` package; falls back to a bare-bones discrete MLE (alpha, xmin,
D only) with a clear note if the package is unavailable in the active venv.

Run:  uv run python -m open_aether.descriptives.powerlaw_fit
"""
from __future__ import annotations
import numpy as np
import pandas as pd
# --- standalone bootstrap: make `open_aether` importable when run as a loose
# --- script (uv run --script); harmless no-op under `python -m open_aether...`.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from open_aether._spillover_common import load_canonical_graph, BASE


# ---- load the canonical graph degree sequence ---------------------------
nodes, A, deg = load_canonical_graph()
deg = np.asarray(deg, dtype="int64")
deg = deg[deg > 0]
N = len(deg)
print(f"networked firms {N:,}  degree: min {int(deg.min())}  median {int(np.median(deg))}  "
      f"mean {deg.mean():.1f}  p95 {int(np.quantile(deg, 0.95))}  max {int(deg.max())}")


def _discrete_mle_fallback(x: np.ndarray):
    """Minimal CSN discrete fit (alpha, xmin, D) with no package. Scans integer
    xmin candidates, uses the continuous-approx MLE for alpha, keeps the xmin with
    the smallest KS distance. Returns a dict; GoF p and LLR are left None."""
    x = np.sort(x)
    best = None
    for xmin in np.unique(x):
        if xmin < 1:
            continue
        tail = x[x >= xmin]
        n = len(tail)
        if n < 10:                       # too few points to fit a tail
            continue
        # discrete MLE (CSN eq. 3.7 approximation): alpha = 1 + n / sum ln(x/(xmin-0.5))
        alpha = 1.0 + n / np.sum(np.log(tail / (xmin - 0.5)))
        # KS distance between empirical and fitted (discrete) CDF on the tail
        ks_support = np.arange(xmin, tail.max() + 1)
        # fitted survival via Hurwitz zeta ratio, approximated by the continuous tail
        emp_cdf = np.searchsorted(tail, ks_support, side="right") / n
        fit_cdf = 1.0 - (ks_support / xmin) ** (-(alpha - 1.0))
        D = float(np.max(np.abs(emp_cdf - fit_cdf)))
        if best is None or D < best["D"]:
            best = {"alpha": float(alpha), "xmin": int(xmin), "D": D, "n_tail": int(n)}
    return best


rows = []
engine = None
try:
    import powerlaw  # type: ignore

    engine = "powerlaw"
    fit = powerlaw.Fit(deg, discrete=True, verbose=False)
    alpha = float(fit.power_law.alpha)
    xmin = int(fit.power_law.xmin)
    D = float(fit.power_law.D)
    n_tail = int((deg >= xmin).sum())
    def _verdict(R, p, alt):
        if p >= 0.05:
            return "cannot distinguish (tie)"
        return f"power law favored over {alt}" if R > 0 else f"{alt} favored over power law"

    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    print("\n[power-law fit — Clauset-Shalizi-Newman via `powerlaw`]")
    print(f"  alpha (scaling exponent)   {alpha:.3f}")
    print(f"  xmin (lower cutoff)        {xmin}")
    print(f"  KS distance D              {D:.4f}")
    print(f"  tail size (deg >= xmin)    {n_tail:,}  ({100*n_tail/N:.1f}% of nodes)")
    print(f"  power-law vs lognormal     R = {R_ln:+.2f}  p = {p_ln:.3f}  ({_verdict(R_ln, p_ln, 'lognormal')})")
    print(f"  power-law vs exponential   R = {R_exp:+.2f}  p = {p_exp:.3f}  ({_verdict(R_exp, p_exp, 'exponential')})")
    rows.append({
        "engine": engine, "alpha": alpha, "xmin": xmin, "ks_D": D,
        "n_tail": n_tail, "tail_share": n_tail / N,
        "R_vs_lognormal": R_ln, "p_vs_lognormal": p_ln,
        "R_vs_exponential": R_exp, "p_vs_exponential": p_exp,
    })
except ImportError:
    engine = "fallback"
    print("\n[power-law fit — FALLBACK discrete MLE; `powerlaw` package not installed]")
    print("  (for the lognormal/exponential comparison and GoF p, install powerlaw:")
    print("   `uv add powerlaw` in the project, or run this file via its --script shebang.)")
    res = _discrete_mle_fallback(deg)
    if res is None:
        print("  too few tail points to fit.")
    else:
        print(f"  alpha (scaling exponent)   {res['alpha']:.3f}")
        print(f"  xmin (lower cutoff)        {res['xmin']}")
        print(f"  KS distance D              {res['D']:.4f}")
        print(f"  tail size (deg >= xmin)    {res['n_tail']:,}  ({100*res['n_tail']/N:.1f}% of nodes)")
        rows.append({
            "engine": engine, "alpha": res["alpha"], "xmin": res["xmin"], "ks_D": res["D"],
            "n_tail": res["n_tail"], "tail_share": res["n_tail"] / N,
            "R_vs_lognormal": np.nan, "p_vs_lognormal": np.nan,
            "R_vs_exponential": np.nan, "p_vs_exponential": np.nan,
        })

print("\nInterpretation guide: a credible scale-free claim wants small D, a tail that")
print("is more than a trivial fraction of nodes, and R>0 with p<0.05 against lognormal.")
print("A negative R with small p means the alternative (lognormal/exponential) fits")
print("better and the power law is rejected — report the degree distribution as")
print("heavy-tailed, and do NOT assert the network is scale-free in the body.")

if rows:
    out = BASE / "desc_powerlaw.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
