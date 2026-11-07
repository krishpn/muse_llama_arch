#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Estimate Sec 1's two-way-fixed-effects DiD, event study, and RDD.

Two-way-fixed-effects DiD (primary identification), its event-study
generalization, and a local-linear RDD around the capital-floor threshold
(secondary). All take the already-computed regressors and HT weights
(mandate_design.py, spillover.py) as inputs -- none fits propensity or
exposure itself.

No statsmodels/sklearn dependency, matching this project's existing
scipy-only convention (paper1_model/pyproject.toml has no stats-package
dependency beyond scipy) -- fixed effects are swept out by exact within
(demeaning) transformation, then weighted least squares via torch.linalg.lstsq.

Inference is bank-clustered throughout: CR1 standard errors, plus wild-cluster
restricted (WCR) bootstrap p-values with Webb six-point weights -- with only
N=16 banks (clusters), CR1's asymptotic normal reference is unreliable, and
Webb weights are the standard choice below ~12-20 clusters (Webb 2014;
MacKinnon, Nielsen & Webb 2023), where Rademacher's 2^G distinct draws get thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


@dataclass
class DiDResult:
    """Hold pooled DiD coefficients with bank-clustered CR1 SEs and WCR p-values.

    Covers dose and exposure plus the control coefficients, observation/cluster
    counts, and the regressors dropped as unidentified or non-finite.
    """

    beta_dose: float
    beta_exposure: float
    beta_controls: dict[str, float]
    n_obs: int
    residual_std: float
    se_dose: float = math.nan  # bank-clustered CR1
    se_exposure: float = math.nan
    p_dose: float | None = None  # WCR wild-cluster-bootstrap p-value, None if n_boot=0
    p_exposure: float | None = None
    n_clusters: int = 0
    dropped_regressors: list[str] = field(default_factory=list)  # unidentified -> NaN, see fit_twfe
    n_cells_nonfinite_regressor: int = 0


def _within_transform(
    y: torch.Tensor, weights: torch.Tensor, rtol: float = 1e-6, max_iter: int = 10_000
) -> torch.Tensor:
    """Sweep out bank and period fixed effects by weighted two-way demeaning.

    Each pass subtracts the weighted bank mean then the weighted period mean of
    the (N, T) tensor:
        y_i,t <- y_i,t - (sum_t w_i,t y_i,t) / (sum_t w_i,t)   (bank mean)
        y_i,t <- y_i,t - (sum_i w_i,t y_i,t) / (sum_i w_i,t)   (period mean)
    iterated until both means fall below tol. Removes bank fixed effects and
    period fixed effects exactly, the standard DiD/panel-FE trick that avoids
    ever materializing N+T dummy columns. A fixed number of passes is only exact
    for balanced panels with separable weights, and HT weights / NaN-masked
    cells are neither -- hence the iteration to convergence.
    """
    w = weights
    w_bank = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    w_period = w.sum(dim=0, keepdim=True).clamp_min(1e-12)
    tol = rtol * y.abs().max().clamp_min(1.0).item()
    for _ in range(max_iter):
        bank_mean = (y * w).sum(dim=1, keepdim=True) / w_bank
        y = y - bank_mean
        period_mean = (y * w).sum(dim=0, keepdim=True) / w_period
        y = y - period_mean
        if max(bank_mean.abs().max().item(), period_mean.abs().max().item()) < tol:
            break
    return y


_WEBB = torch.tensor([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)],
                     dtype=torch.float64)


@dataclass
class TWFEFit:
    """Hold a fitted weighted two-way-FE regression and its cluster-robust inference.

    Coefficients and bank-clustered CR1 SEs per regressor, optional WCR
    bootstrap p-values, observation/cluster counts, dropped regressors, and the
    joint-test results.
    """

    names: list[str]
    beta: dict[str, float]
    se: dict[str, float]  # bank-clustered CR1
    p_wild: dict[str, float] = field(default_factory=dict)  # WCR bootstrap, only for requested names
    n_obs: int = 0
    n_clusters: int = 0
    residual_std: float = math.nan
    dropped: list[str] = field(default_factory=list)  # not identified on valid cells -> beta/se/p NaN
    n_cells_nonfinite_regressor: int = 0  # valid-outcome cells dropped for a non-finite regressor
    # Joint tests, per requested family name: {"wald": CR1 Wald statistic, "p": WCR bootstrap p,
    # "q": number of identified coefficients tested, "names": those coefficients}.
    joint: dict[str, dict] = field(default_factory=dict)


def _cluster_sum(values: torch.Tensor, cluster: torch.Tensor, n_clusters: int) -> torch.Tensor:
    """values: (n,) or (n, B) -> (G,) or (G, B), summed within cluster."""
    out = torch.zeros((n_clusters, *values.shape[1:]), dtype=values.dtype)
    return out.index_add_(0, cluster, values)


def _cr1_scale(n: int, k: int, g: int) -> float:
    return (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else math.nan


def fit_twfe(
    outcome: torch.Tensor,
    regressors: dict[str, torch.Tensor],
    weights: torch.Tensor,
    bootstrap: list[str] | tuple[str, ...] = (),
    n_boot: int = 0,
    seed: int = 0,
    joint: dict[str, list[str]] | None = None,
) -> TWFEFit:
    """Fit a weighted two-way-FE regression of outcome on regressors by WLS.

    outcome (N, T) on regressors (each (N, T)), bank + period FE swept out by
    _within_transform. Let X~, y~ be the within-demeaned regressors/outcome and
    W = diag(weights); the weighted within-OLS estimator is
        beta = (X~' W X~)^-1 X~' W y~
    (computed on sqrt(W)-scaled X~, y~ via a pseudo-inverse of X~' W X~). Cells
    with a non-finite outcome or zero weight drop out of both the demeaning and
    the regression (unbalanced panel). Clusters = banks (rows).

    Standard errors are bank-clustered CR1. With within residuals
    e_i = y~_i - x~_i' beta and per-cluster weighted score u_g = sum_{i in g} w_i x~_i e_i,
    the sandwich is
        V = c * (X~' W X~)^-1 (sum_g u_g u_g') (X~' W X~)^-1,
        c = (G / (G - 1)) * ((n - 1) / (n - k))
    the CR1 small-sample scaling (G clusters, n obs, k regressors); se[name] is
    the sqrt of the matching diagonal of V.

    For each name in `bootstrap`, a WCR wild-cluster-bootstrap p-value for
    H0: beta_name = 0 (restricted residuals, Webb weights, n_boot draws,
    symmetric |t| comparison against the CR1 t-statistic).

    For each family in `joint` (name -> coefficient names), a joint test of
    H0: all those betas = 0: CR1 Wald statistic W = b' V^-1 b (b the tested
    coefficients, V their CR1 block), WCR bootstrap p-value from residuals
    restricted under the joint null, with each draw's own CR1 V (pseudo-inverse,
    so a near-singular V can't raise). Run after the single-coefficient
    p-values, so those are unchanged by it.

    Never raises on bad data (the extracted NRB panel has known extraction
    errors, see paper2_causal_spillover.md Sec 5a): cells where any regressor is
    non-finite are dropped and counted; regressors with no variation after FE
    removal are dropped and reported as NaN; a zero/undefined SE gives a NaN
    p-value instead of dividing by it.
    """
    all_names = list(regressors)
    outcome_ok = torch.isfinite(outcome) & (weights > 0)
    reg_ok = torch.ones_like(outcome_ok)
    for r in regressors.values():
        reg_ok &= torch.isfinite(r)
    valid = outcome_ok & reg_ok
    n_nonfinite = int((outcome_ok & ~reg_ok).sum())

    def _empty(dropped: list[str]) -> TWFEFit:
        nan = {nm: math.nan for nm in all_names}
        return TWFEFit(names=all_names, beta=dict(nan), se=dict(nan),
                       p_wild={nm: math.nan for nm in bootstrap} if n_boot > 0 else {},
                       n_obs=int(valid.sum()), dropped=dropped, n_cells_nonfinite_regressor=n_nonfinite)

    if int(valid.sum()) == 0 or not all_names:
        return _empty(list(all_names))

    weights = torch.where(valid, weights, 0.0).to(torch.float64)
    outcome = torch.where(valid, outcome, torch.zeros_like(outcome)).to(torch.float64)

    n_banks = outcome.shape[0]
    mask = valid.reshape(-1)
    sqrt_w = weights.reshape(-1).sqrt()[mask]

    y = _within_transform(outcome, weights).reshape(-1)[mask] * sqrt_w
    x_all = torch.stack([
        _within_transform(torch.where(valid, regressors[nm], 0.0).to(torch.float64), weights).reshape(-1)[mask]
        for nm in all_names
    ], dim=1) * sqrt_w.unsqueeze(1)

    # A regressor that's identically zero after FE removal (no valid cells, or
    # absorbed by the bank/period FE) has no identified coefficient -- drop it and
    # report NaN rather than beta=0, se=0 (and divide by that se in the bootstrap).
    col_norm = x_all.norm(dim=0)
    keep = [c > 1e-10 * max(1.0, col_norm.max().item()) for c in col_norm.tolist()]
    names = [nm for nm, k_ in zip(all_names, keep) if k_]
    dropped = [nm for nm, k_ in zip(all_names, keep) if not k_]
    if not names:
        return _empty(dropped)
    x = x_all[:, torch.tensor(keep)]

    cluster_full = torch.arange(n_banks).unsqueeze(1).expand_as(valid).reshape(-1)[mask]
    # Re-index clusters to 0..G-1 over banks that actually contribute a cell.
    present, cluster = torch.unique(cluster_full, return_inverse=True)
    n_obs, k, g = x.shape[0], x.shape[1], present.numel()

    xtx_inv = torch.linalg.pinv(x.T @ x)
    proj = xtx_inv @ x.T  # (K, n): beta = proj @ y
    beta = proj @ y
    resid = y - x @ beta
    scale = _cr1_scale(n_obs, k, g)

    se: dict[str, float] = {}
    for j, nm in enumerate(names):
        h = x @ xtx_inv[j]
        q = _cluster_sum(h * resid, cluster, g)
        var = scale * (q ** 2).sum().item()
        se[nm] = math.sqrt(var) if var >= 0 else math.nan  # scale is NaN when g <= 1 or n <= k

    p_wild: dict[str, float] = {}
    if n_boot > 0:
        gen = torch.Generator().manual_seed(seed)
        for nm in bootstrap:
            if nm not in names or not se[nm] > 0:  # dropped, or zero/NaN SE (e.g. exact fit)
                p_wild[nm] = math.nan
                continue
            j = names.index(nm)
            t_obs = beta[j].item() / se[nm]
            x_r = torch.cat([x[:, :j], x[:, j + 1:]], dim=1)
            if x_r.shape[1] > 0:
                beta_r = torch.linalg.lstsq(x_r, y.unsqueeze(1)).solution.squeeze(1)
                fitted_r = x_r @ beta_r
            else:
                fitted_r = torch.zeros_like(y)
            resid_r = y - fitted_r
            h = x @ xtx_inv[j]
            exceed, done = 0, 0
            while done < n_boot:
                b = min(1000, n_boot - done)
                v = _WEBB[torch.randint(0, 6, (g, b), generator=gen)]
                y_star = fitted_r.unsqueeze(1) + resid_r.unsqueeze(1) * v[cluster]
                beta_star = proj @ y_star  # (K, b)
                u_star = y_star - x @ beta_star
                q_star = _cluster_sum(h.unsqueeze(1) * u_star, cluster, g)
                se_star = (scale * (q_star ** 2).sum(dim=0)).sqrt()
                # se_star = 0 -> t_star inf/NaN; NaN never counts as an exceedance.
                t_star = beta_star[j] / se_star
                exceed += int((t_star.abs() >= abs(t_obs)).sum())
                done += b
            p_wild[nm] = exceed / n_boot

    joint_out: dict[str, dict] = {}
    for fam, members in (joint or {}).items():
        idx = [names.index(nm) for nm in members if nm in names]
        tested = [names[j] for j in idx]
        res = {"wald": math.nan, "p": math.nan if n_boot > 0 else None, "q": len(idx), "names": tested}
        joint_out[fam] = res
        if not idx or not math.isfinite(scale):
            continue
        h_q = x @ xtx_inv[idx].T  # (n, q)
        s = _cluster_sum(h_q * resid.unsqueeze(1), cluster, g)  # (G, q) cluster scores
        v = scale * s.T @ s
        b_q = beta[idx]
        wald = (b_q @ torch.linalg.pinv(v) @ b_q).item()
        res["wald"] = wald
        if n_boot <= 0 or not math.isfinite(wald):
            continue
        rest = [j for j in range(k) if j not in idx]
        if rest:
            x_r = x[:, rest]
            fitted_r = x_r @ torch.linalg.lstsq(x_r, y.unsqueeze(1)).solution.squeeze(1)
        else:
            fitted_r = torch.zeros_like(y)
        resid_r = y - fitted_r
        gen_j = torch.Generator().manual_seed(seed + 1)
        exceed, done = 0, 0
        while done < n_boot:
            b = min(200, n_boot - done)  # (G, q, b) score tensor: keep the chunk small
            w = _WEBB[torch.randint(0, 6, (g, b), generator=gen_j)]
            y_star = fitted_r.unsqueeze(1) + resid_r.unsqueeze(1) * w[cluster]
            beta_star = proj @ y_star
            u_star = y_star - x @ beta_star
            s_star = _cluster_sum(h_q.unsqueeze(2) * u_star.unsqueeze(1), cluster, g)  # (G, q, b)
            v_star = scale * torch.einsum("gib,gjb->bij", s_star, s_star)
            bq_star = beta_star[idx].T  # (b, q)
            w_star = torch.einsum("bi,bij,bj->b", bq_star, torch.linalg.pinv(v_star), bq_star)
            exceed += int((w_star >= wald).sum())  # NaN never counts as an exceedance
            done += b
        res["p"] = exceed / n_boot

    beta_out = {nm: math.nan for nm in all_names} | {nm: beta[j].item() for j, nm in enumerate(names)}
    return TWFEFit(
        names=all_names,
        beta=beta_out,
        se={nm: math.nan for nm in all_names} | se,
        p_wild=p_wild,
        n_obs=n_obs,
        n_clusters=g,
        residual_std=resid.std(unbiased=True).item() if n_obs > 1 else math.nan,
        dropped=dropped,
        n_cells_nonfinite_regressor=n_nonfinite,
        joint=joint_out,
    )


def fit_did(
    outcome: torch.Tensor,
    dose: torch.Tensor,
    exposure: torch.Tensor,
    controls: dict[str, torch.Tensor],
    weights: torch.Tensor,
    n_boot: int = 0,
    seed: int = 0,
) -> DiDResult:
    """Fit the pooled treatment-intensity DiD of outcome on dose and exposure.

    Estimates the structural regression
        y_i,t = beta * dose_i,t + gamma * exposure_i,t + delta' controls_i,t
                + alpha_i + tau_t + e_i,t
    with bank fixed effects alpha_i and period fixed effects tau_t swept out via
    the weighted within-transform, then weighted OLS of the demeaned outcome on
    demeaned [dose, exposure, *controls] (no intercept -- removed by demeaning).
    outcome/dose/exposure/each controls value: (N, T). weights: (N, T),
    e.g. spillover.stabilized_ht_weights(...). In the mandate design
    (mandate_design.did_design) `dose`/`exposure` are the frozen-pre-mandate-dose
    x post interactions (dose_pre_i * post_t and exposure_i,t * post_t), so beta
    and gamma are the pre/post mandate effects. n_boot > 0 adds WCR bootstrap
    p-values for dose and exposure.
    """
    if "dose" in controls or "exposure" in controls:
        raise ValueError("control names 'dose'/'exposure' are reserved")
    fit = fit_twfe(
        outcome,
        {"dose": dose, "exposure": exposure, **controls},
        weights,
        bootstrap=("dose", "exposure"),
        n_boot=n_boot,
        seed=seed,
    )
    return DiDResult(
        beta_dose=fit.beta["dose"],
        beta_exposure=fit.beta["exposure"],
        beta_controls={nm: fit.beta[nm] for nm in controls},
        n_obs=fit.n_obs,
        residual_std=fit.residual_std,
        se_dose=fit.se["dose"],
        se_exposure=fit.se["exposure"],
        p_dose=fit.p_wild.get("dose"),
        p_exposure=fit.p_wild.get("exposure"),
        n_clusters=fit.n_clusters,
        dropped_regressors=fit.dropped,
        n_cells_nonfinite_regressor=fit.n_cells_nonfinite_regressor,
    )


@dataclass
class EventStudyResult:
    """Hold event-study dose/exposure coefficients per bin with their inference.

    Per-bin {beta, se, p} for dose and exposure (reference bin excluded),
    observation/cluster counts, dropped bins/regressors, and the joint-test
    results.
    """

    reference_bin: int
    bins: list[str]  # bin labels in display order, reference excluded
    dose: dict[str, dict[str, float | None]]  # bin -> {beta, se, p}
    exposure: dict[str, dict[str, float | None]]
    n_obs: int
    n_clusters: int
    dropped_bins: list[str] = field(default_factory=list)  # no valid outcome cell, not estimable
    dropped_regressors: list[str] = field(default_factory=list)  # unidentified -> NaN, see fit_twfe
    note: str = ""
    joint: dict[str, dict] = field(default_factory=dict)  # dose_post / exposure_post / *_lead, see fit_twfe


def fit_event_study(
    outcome: torch.Tensor,
    dose_pre: torch.Tensor,
    exposure: torch.Tensor | None,
    bins: list[int | str],
    controls: dict[str, torch.Tensor],
    weights: torch.Tensor,
    reference_bin: int = -1,
    n_boot: int = 0,
    seed: int = 0,
) -> EventStudyResult:
    """Fit the event-study generalization of the frozen-dose DiD.

    Estimates
        y_i,t = sum_{k != ref} beta_k (dose_pre_i * 1[bin_t = k])
                + sum_{k != ref} gamma_k (exposure_i,t * 1[bin_t = k])
                + delta' controls_i,t + alpha_i + tau_t + e_i,t,
    i.e. separate dose_pre_i x 1[bin_t = k] and exposure_i,t x 1[bin_t = k]
    coefficients for every event-time bin k except reference_bin (bins: one
    label per period, from mandate_design.event_bins). Pre-reference bins are
    the parallel-trends check; post bins trace the dynamic effect. `controls`
    should already include the exposure level term (mandate_design.did_design's
    "exposure_level") so the bin interactions are identified off within-bank
    changes, as in the pooled DiD. Bins with no valid outcome cell (e.g. the
    panel's first month after YTD de-cumulation) are dropped and listed in
    dropped_bins, not estimated as zero. exposure=None gives the dose-only event
    study (no exposure[k] terms, exposure result empty).
    """
    n, t = outcome.shape
    valid_t = (torch.isfinite(outcome) & (weights > 0)).any(dim=0).tolist()
    has_data = {b: any(v for bt, v in zip(bins, valid_t) if bt == b) for b in dict.fromkeys(bins)}
    if not has_data.get(reference_bin, False):
        # No normalization possible -> report nothing rather than raise mid-run.
        return EventStudyResult(reference_bin=reference_bin, bins=[], dose={}, exposure={}, n_obs=0,
                                n_clusters=0, dropped_bins=[str(b) for b in has_data],
                                note=f"reference bin {reference_bin} has no valid outcome cells")
    labels = [b for b in has_data if b != reference_bin and has_data[b]]
    dropped = [str(b) for b in has_data if not has_data[b]]

    regressors: dict[str, torch.Tensor] = {}
    for b in labels:
        ind = torch.tensor([1.0 if bt == b else 0.0 for bt in bins]).unsqueeze(0)
        regressors[f"dose[{b}]"] = dose_pre.unsqueeze(1) * ind
        if exposure is not None:
            regressors[f"exposure[{b}]"] = exposure * ind
    clash = set(regressors) & set(controls)
    if clash:
        raise ValueError(f"control names clash with event-study regressors: {clash}")
    regressors.update(controls)

    boot = [nm for nm in regressors if nm.startswith(("dose[", "exposure["))]
    # Joint tests (Sec 5b item 5): all post bins = 0, and all lead bins = 0, per coefficient.
    # The transition is in neither family: it is neither pre- nor fully post-mandate.
    post_bins = [b for b in labels if isinstance(b, int) and b >= 0]
    lead_bins = [b for b in labels if isinstance(b, int) and b < reference_bin]
    coefs = ("dose",) if exposure is None else ("dose", "exposure")
    families = {f"{coef}_{side}": [f"{coef}[{b}]" for b in bs]
                for coef in coefs for side, bs in (("post", post_bins), ("lead", lead_bins)) if bs}
    fit = fit_twfe(outcome, regressors, weights, bootstrap=boot, n_boot=n_boot, seed=seed, joint=families)

    def _entry(nm: str) -> dict[str, float | None]:
        return {"beta": fit.beta[nm], "se": fit.se[nm], "p": fit.p_wild.get(nm)}

    return EventStudyResult(
        reference_bin=reference_bin,
        bins=[str(b) for b in labels],
        dose={str(b): _entry(f"dose[{b}]") for b in labels},
        exposure={} if exposure is None else {str(b): _entry(f"exposure[{b}]") for b in labels},
        n_obs=fit.n_obs,
        n_clusters=fit.n_clusters,
        dropped_bins=dropped,
        dropped_regressors=fit.dropped,
        joint=fit.joint,
    )


@dataclass
class RDDResult:
    """Hold a local-linear RDD fit: left/right threshold intercepts and their jump.

    beta_left / beta_right are the fitted outcomes at dose=0 from each side,
    discontinuity their difference, with the bandwidth and cell/bank counts.
    """

    beta_left: float
    beta_right: float
    discontinuity: float
    bandwidth: float
    n_left: int
    n_right: int
    n_banks_left: int = 0
    n_banks_right: int = 0


def fit_rdd(
    outcome: torch.Tensor,
    dose: torch.Tensor,
    period_idx: int | list[int],
    bandwidth: float,
) -> RDDResult:
    """Estimate a sharp local-linear RDD at the dose=0 capital-floor threshold.

    Fits separate local-linear regressions of outcome on dose on each side,
        m_side(x) = a_side + b_side * x   (x = dose, side in {left, right}),
    and reports the discontinuity at the threshold
        discontinuity = m_right(0) - m_left(0) = a_right - a_left,
    the jump in the fitted outcome as dose crosses 0 (dose = capital_floor -
    capital_ratio; dose>0 means below the floor). Restricted to the given
    period(s) within +/- bandwidth of the threshold. A list of periods pools
    those cross-sections (bank-months, so n_left/n_right count cells, not
    banks -- n_banks_* is the number of distinct banks, the real effective
    sample with N=16). Same convention as the standard Imbens-Lemieux
    local-linear RDD estimator.
    """
    cols = [period_idx] if isinstance(period_idx, int) else list(period_idx)
    y = outcome[:, cols]
    x = dose[:, cols]
    bank = torch.arange(outcome.shape[0]).unsqueeze(1).expand_as(y)

    within = (x.abs() <= bandwidth) & torch.isfinite(y) & torch.isfinite(x)
    left_mask = within & (x < 0)
    right_mask = within & (x >= 0)

    def _local_linear(mask: torch.Tensor) -> float:
        xs, ys = x[mask], y[mask]
        if xs.numel() < 2:
            return float("nan")
        design = torch.stack([torch.ones_like(xs), xs], dim=1)
        beta = torch.linalg.lstsq(design, ys.unsqueeze(1)).solution.squeeze(1)
        return beta[0].item()  # intercept = fitted value at dose=0 from this side

    beta_left = _local_linear(left_mask)
    beta_right = _local_linear(right_mask)

    return RDDResult(
        beta_left=beta_left,
        beta_right=beta_right,
        discontinuity=beta_right - beta_left,
        bandwidth=bandwidth,
        n_left=int(left_mask.sum()),
        n_right=int(right_mask.sum()),
        n_banks_left=int(bank[left_mask].unique().numel()),
        n_banks_right=int(bank[right_mask].unique().numel()),
    )
