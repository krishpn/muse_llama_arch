#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Tests for the DiD, RDD, TWFE, and event-study estimators in estimator.py."""

import math
import sys
from pathlib import Path

import torch

try:
    from open_nrb_causal.models.estimator import fit_did, fit_event_study, fit_rdd, fit_twfe
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from estimator import fit_did, fit_event_study, fit_rdd, fit_twfe  # type: ignore[no-redef]


def fit_did_recovers_known_coefficients_case() -> bool:
    """Assert fit_did recovers the true beta_dose/beta_exposure/beta_control.

    Synthetic panel with known betas plus bank and period fixed effects and
    noise; fit_did's weighted within-transform recovers the true betas up to
    sampling noise (each error < 0.1), with weights=1 -- ordinary two-way FE
    OLS, the uniform-weight special case of HT weighting.
    """
    torch.manual_seed(0)
    n, t = 20, 15
    true_beta_dose, true_beta_exposure, true_beta_control = 1.5, -0.8, 0.3

    bank_fe = torch.randn(n, 1) * 2.0
    period_fe = torch.randn(1, t) * 2.0
    dose = torch.randn(n, t)
    exposure = torch.randn(n, t)
    control = torch.randn(n, t)
    noise = torch.randn(n, t) * 0.05

    outcome = (
        true_beta_dose * dose
        + true_beta_exposure * exposure
        + true_beta_control * control
        + bank_fe
        + period_fe
        + noise
    )
    weights = torch.ones(n, t)

    result = fit_did(outcome, dose, exposure, {"control": control}, weights)

    dose_err = abs(result.beta_dose - true_beta_dose)
    exposure_err = abs(result.beta_exposure - true_beta_exposure)
    control_err = abs(result.beta_controls["control"] - true_beta_control)
    passed = dose_err < 0.1 and exposure_err < 0.1 and control_err < 0.1
    print(f"[fit_did_recovers_known_coefficients_case] dose_err={dose_err:.4f} "
          f"exposure_err={exposure_err:.4f} control_err={control_err:.4f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_rdd_recovers_known_discontinuity_case() -> bool:
    """Assert fit_rdd recovers a known discontinuity at dose=0.

    Synthetic cross-section with a known jump at dose=0; fit_rdd's local-linear
    fit on each side recovers it within the bandwidth (jump_err < 0.1).
    """
    torch.manual_seed(1)
    n = 400
    dose_full = torch.linspace(-0.5, 0.5, n)
    true_jump = 2.0
    outcome_full = 1.0 + 0.5 * dose_full + true_jump * (dose_full >= 0).float() + torch.randn(n) * 0.02

    # fit_rdd indexes a single period column -- build a (N, 1) tensor.
    outcome = outcome_full.unsqueeze(1)
    dose = dose_full.unsqueeze(1)

    result = fit_rdd(outcome, dose, period_idx=0, bandwidth=0.2)

    jump_err = abs(result.discontinuity - true_jump)
    passed = jump_err < 0.1
    print(f"[fit_rdd_recovers_known_discontinuity_case] discontinuity={result.discontinuity:.4f} "
          f"true={true_jump} err={jump_err:.4f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_rdd_too_few_points_returns_nan_case() -> bool:
    """Assert fit_rdd returns NaN betas when a side has too few points.

    With only one point per side inside the bandwidth, n_left and n_right stay
    <= 1 and beta_left/beta_right are NaN (detected via x != x).
    """
    outcome = torch.tensor([[1.0], [2.0]])
    dose = torch.tensor([[-0.9], [0.9]])  # only 1 point per side within a tiny bandwidth
    result = fit_rdd(outcome, dose, period_idx=0, bandwidth=0.05)
    passed = result.n_left <= 1 and result.n_right <= 1 and (
        result.beta_left != result.beta_left or result.beta_right != result.beta_right
    )
    print(f"[fit_rdd_too_few_points_returns_nan_case] n_left={result.n_left} "
          f"n_right={result.n_right} -> {'PASS' if passed else 'FAIL'}")
    return passed


def cr1_se_matches_full_dummy_sandwich_case() -> bool:
    """Assert fit_twfe's CR1 SE equals the bank-clustered dummy-regression sandwich.

    The within-transformed CR1 SE must equal the regressor block of the
    bank-clustered sandwich from an explicit bank+period dummy regression (FWL),
    with the same CR1 scaling scale = (G/(G-1)) * ((N-1)/(N-k)), where G = n
    clusters, N = n*t observations, and k = 2 non-FE regressors; equality holds
    to 1e-4 relative.
    """
    torch.manual_seed(3)
    n, t = 8, 6
    x1, x2 = torch.randn(n, t), torch.randn(n, t)
    y = 0.7 * x1 - 0.2 * x2 + torch.randn(n, 1) + torch.randn(1, t) + torch.randn(n, t)
    w = torch.rand(n, t) + 0.5
    fit = fit_twfe(y, {"x1": x1, "x2": x2}, w)

    d = torch.float64
    bank = torch.arange(n).unsqueeze(1).expand(n, t).reshape(-1)
    period = torch.arange(t).unsqueeze(0).expand(n, t).reshape(-1)
    full = torch.cat([
        x1.reshape(-1, 1), x2.reshape(-1, 1),
        torch.nn.functional.one_hot(bank, n).float(),
        torch.nn.functional.one_hot(period, t).float()[:, 1:],
    ], dim=1).to(d)
    sw = w.reshape(-1).to(d).sqrt()
    xw, yw = full * sw.unsqueeze(1), y.reshape(-1).to(d) * sw
    xtx_inv = torch.linalg.pinv(xw.T @ xw)
    u = yw - xw @ (xtx_inv @ xw.T @ yw)
    scores = torch.zeros(n, full.shape[1], dtype=d).index_add_(0, bank, xw * u.unsqueeze(1))
    v = xtx_inv @ (scores.T @ scores) @ xtx_inv
    scale = (n / (n - 1)) * ((n * t - 1) / (n * t - 2))
    manual = [math.sqrt(scale * v[0, 0].item()), math.sqrt(scale * v[1, 1].item())]

    # 1e-4 relative: _within_transform stops at rtol=1e-6, not exact convergence.
    passed = all(abs(a - b) < 1e-4 * max(1.0, b) for a, b in zip([fit.se["x1"], fit.se["x2"]], manual))
    print(f"[cr1_se_matches_full_dummy_sandwich_case] fit={[fit.se['x1'], fit.se['x2']]} manual={manual} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def wild_bootstrap_rejects_real_effect_not_null_case() -> bool:
    """Assert the wild cluster bootstrap rejects a real effect but not a null one.

    16 clusters (same as the real panel) with bank-level random-effect noise: a
    large true effect gets a small WCR p-value (< 0.01), a zero effect does not
    (> 0.05), and n_clusters == 16.
    """
    torch.manual_seed(4)
    n, t = 16, 20
    x_real, x_null = torch.randn(n, t), torch.randn(n, t)
    noise = torch.randn(n, t) + 0.5 * torch.randn(n, 1) * torch.randn(1, t)
    y = 1.0 * x_real + 0.0 * x_null + torch.randn(n, 1) + torch.randn(1, t) + noise
    fit = fit_twfe(y, {"real": x_real, "null": x_null}, torch.ones(n, t),
                   bootstrap=("real", "null"), n_boot=1999, seed=0)
    passed = fit.p_wild["real"] < 0.01 and fit.p_wild["null"] > 0.05 and fit.n_clusters == n
    print(f"[wild_bootstrap_rejects_real_effect_not_null_case] p={fit.p_wild} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_rdd_pools_periods_and_counts_banks_case() -> bool:
    """Assert fit_rdd pools period columns and counts distinct banks per side.

    Pooling three period columns yields n_left == n_right == 3 cells, each drawn
    from a single bank per side (n_banks_left == n_banks_right == 1).
    """
    outcome = torch.tensor([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [5.0, 5.1, 5.2]])
    dose = torch.tensor([[-0.01, -0.012, -0.011], [0.01, 0.011, 0.012], [0.5, 0.5, 0.5]])
    result = fit_rdd(outcome, dose, period_idx=[0, 1, 2], bandwidth=0.02)
    passed = (result.n_left == 3 and result.n_right == 3
              and result.n_banks_left == 1 and result.n_banks_right == 1)
    print(f"[fit_rdd_pools_periods_and_counts_banks_case] cells={result.n_left}/{result.n_right} "
          f"banks={result.n_banks_left}/{result.n_banks_right} -> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_event_study_recovers_dynamic_effect_case() -> bool:
    """Assert fit_event_study recovers the dynamic dose path with no pre-trend.

    The dose_pre effect switches on at bin 0 and doubles at bin 1; pre bin -2 is
    ~0 (no pre-trend), bins 0/1/transition recover 1.0/2.0/0.5, and the omitted
    reference bin -1 is absent from the output.
    """
    torch.manual_seed(5)
    n = 16
    bins: list[int | str] = [-2] * 4 + [-1] * 4 + ["transition"] * 2 + [0] * 4 + [1] * 4
    t = len(bins)
    dose_pre = torch.randn(n)
    exposure = torch.randn(n, t)
    effect = torch.tensor([{0: 1.0, 1: 2.0}.get(b, 0.0) if isinstance(b, int) else 0.5 for b in bins])
    y = (dose_pre.unsqueeze(1) * effect.unsqueeze(0) + 0.3 * exposure
         + torch.randn(n, 1) + torch.randn(1, t) + 0.01 * torch.randn(n, t))
    es = fit_event_study(y, dose_pre, exposure, bins, {"exposure_level": exposure}, torch.ones(n, t))
    passed = (
        abs(es.dose["-2"]["beta"]) < 0.05
        and abs(es.dose["0"]["beta"] - 1.0) < 0.05
        and abs(es.dose["1"]["beta"] - 2.0) < 0.05
        and abs(es.dose["transition"]["beta"] - 0.5) < 0.05
        and "-1" not in es.dose
    )
    print(f"[fit_event_study_recovers_dynamic_effect_case] "
          f"{ {k: round(v['beta'], 3) for k, v in es.dose.items()} } -> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_event_study_drops_bin_with_no_valid_cells_case() -> bool:
    """Assert fit_event_study drops a bin that has no valid cells.

    Real-run regression (2026-11-03): YTD de-cumulation NaNs the panel's first
    month, which was the whole of bin -2 -> zero regressor columns -> se=0 ->
    ZeroDivisionError in the bootstrap. The bin must be dropped instead
    (dropped_bins == ["-2"], only bin "0" survives with a non-None p).
    """
    torch.manual_seed(6)
    n = 16
    bins: list[int | str] = [-2] + [-1] * 4 + [0] * 4
    t = len(bins)
    exposure = torch.randn(n, t)
    y = torch.randn(n, t)
    y[:, 0] = float("nan")
    es = fit_event_study(y, torch.randn(n), exposure, bins, {"exposure_level": exposure},
                         torch.ones(n, t), n_boot=99)
    passed = es.dropped_bins == ["-2"] and list(es.dose) == ["0"] and es.dose["0"]["p"] is not None
    print(f"[fit_event_study_drops_bin_with_no_valid_cells_case] dropped={es.dropped_bins} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_twfe_drops_degenerate_regressor_case() -> bool:
    """Assert fit_twfe drops a bank-constant regressor absorbed by the bank FE.

    A bank-constant regressor is absorbed by the bank fixed effect: reported as
    NaN and listed in `dropped`, while the identified regressor is still
    estimated (finite beta) and nothing is raised.
    """
    torch.manual_seed(7)
    n, t = 6, 5
    fit = fit_twfe(torch.randn(n, t), {"x": torch.randn(n, t), "bank_constant": torch.randn(n, 1).expand(n, t)},
                   torch.ones(n, t), bootstrap=("x", "bank_constant"), n_boot=99)
    passed = (fit.dropped == ["bank_constant"] and math.isnan(fit.beta["bank_constant"])
              and math.isnan(fit.p_wild["bank_constant"]) and math.isfinite(fit.beta["x"]))
    print(f"[fit_twfe_drops_degenerate_regressor_case] dropped={fit.dropped} -> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_twfe_exact_fit_no_zero_division_case() -> bool:
    """Assert fit_twfe handles an exact fit without a ZeroDivisionError.

    y is explained exactly by x + FE, so residuals and the CR1 SE are ~0 (only
    _within_transform's rtol-level error remains); the bootstrap returns a valid
    or NaN p, never raising ZeroDivisionError on beta/se.
    """
    torch.manual_seed(8)
    n, t = 6, 5
    x = torch.randn(n, t)
    y = 2.0 * x + torch.randn(n, 1) + torch.randn(1, t)
    fit = fit_twfe(y, {"x": x}, torch.ones(n, t), bootstrap=("x",), n_boot=99)
    passed = abs(fit.beta["x"] - 2.0) < 1e-3 and (fit.se["x"] < 1e-3) and (
        math.isnan(fit.p_wild["x"]) or 0.0 <= fit.p_wild["x"] <= 1.0)
    print(f"[fit_twfe_exact_fit_no_zero_division_case] beta={fit.beta['x']:.4f} se={fit.se['x']:.2e} "
          f"p={fit.p_wild['x']} -> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_twfe_drops_nonfinite_regressor_cells_case() -> bool:
    """Assert fit_twfe drops cells with a non-finite regressor and counts them."""
    torch.manual_seed(9)
    n, t = 8, 6
    x = torch.randn(n, t)
    y = 0.5 * x + torch.randn(n, t)
    x[0, 0], x[1, 2] = float("nan"), float("inf")
    fit = fit_twfe(y, {"x": x}, torch.ones(n, t))
    passed = fit.n_cells_nonfinite_regressor == 2 and fit.n_obs == n * t - 2 and math.isfinite(fit.beta["x"])
    print(f"[fit_twfe_drops_nonfinite_regressor_cells_case] dropped_cells={fit.n_cells_nonfinite_regressor} "
          f"n_obs={fit.n_obs} -> {'PASS' if passed else 'FAIL'}")
    return passed


def joint_wald_rejects_real_effect_not_null_case() -> bool:
    """Assert the joint WCR Wald test rejects a real family but not a null one.

    A family with one real coefficient among three must reject (p < 0.01), an
    all-null family must not (p > 0.05, q == 3); for a single-coefficient family
    (q=1) the Wald statistic equals the CR1 t-squared: wald == t^2 = (beta/se)^2.
    """
    torch.manual_seed(6)
    n, t = 16, 20
    xs = {f"x{i}": torch.randn(n, t) for i in range(6)}
    noise = torch.randn(n, t) + 0.5 * torch.randn(n, 1) * torch.randn(1, t)
    y = 1.0 * xs["x0"] + torch.randn(n, 1) + torch.randn(1, t) + noise
    fit = fit_twfe(y, xs, torch.ones(n, t), n_boot=1999, seed=0,
                   joint={"real": ["x0", "x1", "x2"], "null": ["x3", "x4", "x5"], "single": ["x3"]})
    t2 = (fit.beta["x3"] / fit.se["x3"]) ** 2
    passed = (fit.joint["real"]["p"] < 0.01 and fit.joint["null"]["p"] > 0.05
              and fit.joint["real"]["q"] == 3 and abs(fit.joint["single"]["wald"] - t2) < 1e-6 * max(1.0, t2))
    print(f"[joint_wald_rejects_real_effect_not_null_case] p_real={fit.joint['real']['p']} "
          f"p_null={fit.joint['null']['p']} wald_single={fit.joint['single']['wald']:.4f} t2={t2:.4f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


CASES = [
    fit_did_recovers_known_coefficients_case,
    fit_rdd_recovers_known_discontinuity_case,
    fit_rdd_too_few_points_returns_nan_case,
    cr1_se_matches_full_dummy_sandwich_case,
    wild_bootstrap_rejects_real_effect_not_null_case,
    fit_rdd_pools_periods_and_counts_banks_case,
    fit_event_study_recovers_dynamic_effect_case,
    fit_event_study_drops_bin_with_no_valid_cells_case,
    fit_twfe_drops_degenerate_regressor_case,
    fit_twfe_exact_fit_no_zero_division_case,
    fit_twfe_drops_nonfinite_regressor_cells_case,
    joint_wald_rejects_real_effect_not_null_case,
]


def test_fit_event_study_drops_bin_with_no_valid_cells():
    """Assert fit_event_study_drops_bin_with_no_valid_cells passes."""
    assert fit_event_study_drops_bin_with_no_valid_cells_case()


def test_fit_twfe_drops_degenerate_regressor():
    """Assert fit_twfe_drops_degenerate_regressor passes."""
    assert fit_twfe_drops_degenerate_regressor_case()


def test_fit_twfe_exact_fit_no_zero_division():
    """Assert fit_twfe_exact_fit_no_zero_division passes."""
    assert fit_twfe_exact_fit_no_zero_division_case()


def test_fit_twfe_drops_nonfinite_regressor_cells():
    """Assert fit_twfe_drops_nonfinite_regressor_cells passes."""
    assert fit_twfe_drops_nonfinite_regressor_cells_case()


def test_joint_wald_rejects_real_effect_not_null():
    """Assert joint_wald_rejects_real_effect_not_null passes."""
    assert joint_wald_rejects_real_effect_not_null_case()


def test_cr1_se_matches_full_dummy_sandwich():
    """Assert cr1_se_matches_full_dummy_sandwich passes."""
    assert cr1_se_matches_full_dummy_sandwich_case()


def test_wild_bootstrap_rejects_real_effect_not_null():
    """Assert wild_bootstrap_rejects_real_effect_not_null passes."""
    assert wild_bootstrap_rejects_real_effect_not_null_case()


def test_fit_rdd_pools_periods_and_counts_banks():
    """Assert fit_rdd_pools_periods_and_counts_banks passes."""
    assert fit_rdd_pools_periods_and_counts_banks_case()


def test_fit_event_study_recovers_dynamic_effect():
    """Assert fit_event_study_recovers_dynamic_effect passes."""
    assert fit_event_study_recovers_dynamic_effect_case()


def test_fit_did_recovers_known_coefficients():
    """Assert fit_did_recovers_known_coefficients passes."""
    assert fit_did_recovers_known_coefficients_case()


def test_fit_rdd_recovers_known_discontinuity():
    """Assert fit_rdd_recovers_known_discontinuity passes."""
    assert fit_rdd_recovers_known_discontinuity_case()


def test_fit_rdd_too_few_points_returns_nan():
    """Assert fit_rdd_too_few_points_returns_nan passes."""
    assert fit_rdd_too_few_points_returns_nan_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
