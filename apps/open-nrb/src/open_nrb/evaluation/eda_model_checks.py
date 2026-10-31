#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "matplotlib",
#     "pyyaml",
# ]
# ///
"""Reviewer pre-mortem EDA (design doc Sec 10a) plus standard network-reconstruction/
systemic-risk literature checks (Squartini & Garlaschelli 2011; Garlaschelli & Loffredo 2004;
Cimini et al. 2015; Battiston et al. 2012) -- descriptive-only, no new training. Reads only
from artifacts run_real_experiment.py/run_synthetic_experiment.py already save -- never
retrains.

Section A runs off the raw panel alone (data_loader.load_panel) -- always available once the
real CSV exists:
  5. discard-sample bias: first-through-fourth moments (mean/var/skew/excess-kurtosis) of bank
     size and capital ratio, included (complete-case) vs. excluded banks, own-active-window
     basis -- extends eda_survivorship.py's mean-only comparison.
  6. parameter count vs. effective real-data size (N x T), no training needed.
  (basic) coverage-count time series and aggregate interbank-marginal (r_t, c_t) time series,
     each with its own moments, over the full real panel.

Section B needs a trained artifact from run_real_experiment.py (results/real_run_seed{seed}.pt):
  1. smoothness-penalty circularity check: lag-1 autocorrelation of raw edge values, model vs.
     classical baseline -- if the baseline is already highly autocorrelated for free, the
     model's stability margin is weaker evidence of learned structure than it looks.
  2. cap-binding fraction: how often does $\\hat W_t$ actually hit the regulatory cap it was
     given and the baseline wasn't.
  per-period edge-weight moments of $\\hat W_t$ and the baseline, as a time series.

Section C needs the same real artifact plus (independently) run_synthetic_experiment.py's
artifact (results/seed_{seed}.pt) -- whether a reconstruction preserves the empirical strength
sequence/density/reciprocity structure (not just Frobenius accuracy), and whether it preserves
the systemic-risk-relevant structure (centrality ranking, DebtRank contagion estimate) that's
the actual reason a reconstructed interbank network is useful:
  marginal-preservation error against the real row/col targets; density and weighted
  reciprocity vs. the model/baseline/(synthetic) ground truth; eigenvector-centrality rank
  agreement; DebtRank mean/worst single-bank-shock systemic impact.

Explicitly out of scope: formal core-periphery block-fitting. Eigenvector centrality + DebtRank
already answer "does reconstruction preserve who's systemically important" without it.

Usage: uv run eda_model_checks.py [--csv PATH] [--seed N] [--config PATH] [--synthetic-config PATH] [--tol F]
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import torch

try:
    from ..models.config import EncoderDecoderConfig
    from ..models.data_loader import COVARIATE_ORDER, Panel, filter_commercial_banks, filter_from_period, load_panel
    from ..models.experiment_config import ModelConfig, load_real_experiment_config, load_synthetic_experiment_config
    from ..models.train import ReconstructionModel
    from .report_io import captured_output, save_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from config import EncoderDecoderConfig
    from data_loader import COVARIATE_ORDER, Panel, filter_commercial_banks, filter_from_period, load_panel
    from experiment_config import ModelConfig, load_real_experiment_config, load_synthetic_experiment_config
    from train import ReconstructionModel
    from report_io import captured_output, save_report

DEFAULT_CSV = Path("src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv")
# results/ and experiments/ are both siblings of this script's parent dir (evaluation/) under
# src/open_nrb/, same __file__-relative convention as run_real_experiment.py's RESULTS_DIR/
# DEFAULT_CONFIG -- this script reads real_run_seed{seed}.pt/seed_{seed}.pt from the same place
# those scripts wrote them, and loads the same baseline.yaml/synthetic_baseline.yaml by default
# so param_count_vs_data_size below matches whatever architecture that artifact was actually
# trained with.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "baseline.yaml"
DEFAULT_SYNTHETIC_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "synthetic_baseline.yaml"

D = len(COVARIATE_ORDER)


def moments(x: torch.Tensor) -> dict[str, float]:
    """Population (not sample) first-through-fourth moments -- mean, variance, std, skewness,
    excess kurtosis -- over x's finite entries. Population (divide by n, not n-1) because the
    goal here is describing the actual observed set (e.g. "the 17 excluded banks"), not
    estimating a population parameter from a sample of a larger one.
    """
    finite = x[~x.isnan()] if x.dtype.is_floating_point else x.float()
    n = finite.numel()
    if n == 0:
        return {"n": 0, "mean": math.nan, "var": math.nan, "std": math.nan, "skew": math.nan, "kurtosis_excess": math.nan}
    mean = finite.mean()
    var = ((finite - mean) ** 2).mean()
    std = var.sqrt()
    if std.item() == 0:
        skew, kurt = math.nan, math.nan
    else:
        skew = (((finite - mean) ** 3).mean() / std**3).item()
        kurt = (((finite - mean) ** 4).mean() / std**4).item() - 3.0
    return {"n": n, "mean": mean.item(), "var": var.item(), "std": std.item(), "skew": skew, "kurtosis_excess": kurt}


def _print_moments(label: str, m: dict[str, float]) -> None:
    print(
        f"  {label:28s} n={m['n']:4d}  mean={m['mean']:12.4f}  std={m['std']:12.4f}  "
        f"skew={m['skew']:7.4f}  excess_kurt={m['kurtosis_excess']:8.4f}"
    )


# ---------------------------------------------------------------------------
# Section A -- raw panel only
# ---------------------------------------------------------------------------


def _complete_case_mask(panel: Panel) -> torch.Tensor:
    """Same NaN definition as run_real_experiment.py's _filter_complete_banks, over the full
    period range (no START_PERIOD/END_PERIOD narrowing -- that's a separate per-run knob)."""
    return ~(panel.X.isnan().any(dim=(1, 2)) | panel.r.isnan().any(dim=1) | panel.c.isnan().any(dim=1) | panel.K.isnan().any(dim=1))


def _own_window_covariate_mean(panel: Panel, covariate_idx: int) -> torch.Tensor:
    """(N,) per-bank mean of covariate `covariate_idx` over that bank's own non-NaN periods --
    doesn't penalize an excluded bank for periods it never had any chance of reporting."""
    col = panel.X[:, :, covariate_idx]
    present = ~col.isnan()
    out = torch.full((col.shape[0],), math.nan)
    for i in range(col.shape[0]):
        vals = col[i, present[i]]
        if vals.numel():
            out[i] = vals.mean()
    return out


def discard_bias_moments(panel: Panel) -> dict:
    print("\n=== Item 5: discard-sample bias (included vs. excluded, own-active-window basis) ===")
    included = _complete_case_mask(panel)
    excluded = ~included
    print(f"included (complete-case): {int(included.sum())}   excluded: {int(excluded.sum())}")
    result = {"n_included": int(included.sum()), "n_excluded": int(excluded.sum()), "by_covariate": {}}
    for name, idx in COVARIATE_NAME_INDEX.items():
        size_like = _own_window_covariate_mean(panel, idx)
        print(f"\n covariate: {name}")
        included_moments = moments(size_like[included])
        excluded_moments = moments(size_like[excluded])
        _print_moments("included", included_moments)
        _print_moments("excluded", excluded_moments)
        result["by_covariate"][name] = {"included": included_moments, "excluded": excluded_moments}
    return result


COVARIATE_NAME_INDEX = {"size": 0, "capital_ratio": 1}


def coverage_time_series(panel: Panel) -> list[int]:
    """Per-period count of banks with >=1 non-NaN covariate that period -- a covariate-presence
    proxy, not the literal raw-row presence eda_survivorship.py's has_rows computes; good enough
    for a coverage-over-time description, not a substitute for that script's entry/exit detail.
    """
    present = ~panel.X.isnan().all(dim=2)  # (N, T)
    counts = present.sum(dim=0)  # (T,)
    print("\n=== Coverage-count time series (covariate-presence proxy) ===")
    _print_moments("banks present per period", moments(counts.float()))
    print(f"  min={int(counts.min())}  max={int(counts.max())}")
    return counts.tolist()


def marginal_time_series(panel: Panel) -> tuple[list[float], list[float]]:
    """Aggregate (sum over included banks) interbank lending/borrowing per period, complete-case
    subset only -- same subset run_real_experiment.py actually trains on."""
    included = _complete_case_mask(panel)
    r_t = panel.r[included].sum(dim=0)
    c_t = panel.c[included].sum(dim=0)
    print("\n=== Aggregate marginal time series (complete-case banks only) ===")
    _print_moments("total r_t (lending)", moments(r_t))
    _print_moments("total c_t (borrowing)", moments(c_t))
    return r_t.tolist(), c_t.tolist()


def param_count_vs_data_size(n_banks: int, n_periods: int, model_config: ModelConfig) -> dict:
    enc_config = EncoderDecoderConfig(
        input_dim=D,
        hidden_dim=model_config.hidden_dim,
        num_blocks=model_config.num_blocks,
        decoder_hidden_dim=model_config.decoder_hidden_dim,
    )
    model = ReconstructionModel(enc_config)
    n_params = sum(p.numel() for p in model.parameters())
    n_obs = n_banks * n_periods
    ratio = n_params / n_obs if n_obs else None
    print("\n=== Item 6: parameter count vs. effective real-data size ===")
    print(f"trainable parameters: {n_params}")
    print(f"N x T (effective real observations): {n_banks} x {n_periods} = {n_obs}")
    print(f"params / (N x T): {ratio:.3f}" if ratio is not None else "N x T is zero -- cannot ratio")
    return {"n_params": n_params, "n_banks": n_banks, "n_periods": n_periods, "n_obs": n_obs, "params_per_obs": ratio}


def _plot_section_a(coverage: list[int], r_t: list[float], c_t: list[float], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(coverage)
    axes[0].set_title("Banks present per period")
    axes[0].set_xlabel("period index")
    axes[0].set_ylabel("count")
    axes[1].plot(r_t, label="total r_t (lending)")
    axes[1].plot(c_t, label="total c_t (borrowing)")
    axes[1].set_title("Aggregate interbank marginals (complete-case banks)")
    axes[1].set_xlabel("period index")
    axes[1].legend()
    fig.suptitle("Reviewer pre-mortem EDA -- Section A (raw panel)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eda_model_checks_section_a.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {path}")


# ---------------------------------------------------------------------------
# Section B -- needs a trained real run (results/real_run_seed{seed}.pt)
# ---------------------------------------------------------------------------


def _off_diagonal(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool)
    return w[..., mask]


def edge_moments_over_time(W_seq: torch.Tensor, label: str) -> dict:
    print(f"\n=== Edge-weight moments over time: {label} ===")
    per_period = [moments(_off_diagonal(W_seq[t])) for t in range(W_seq.shape[0])]
    means = [m["mean"] for m in per_period]
    stds = [m["std"] for m in per_period]
    skews = [m["skew"] for m in per_period]
    kurts = [m["kurtosis_excess"] for m in per_period]
    print(f"  mean of per-period means:  {sum(v for v in means if v == v) / max(1, sum(1 for v in means if v == v)):.4f}")
    means_of_means = moments(torch.tensor(means))
    means_of_stds = moments(torch.tensor(stds))
    means_of_skews = moments(torch.tensor(skews))
    means_of_kurts = moments(torch.tensor(kurts))
    _print_moments("std-of-means (drift)", means_of_means)
    _print_moments("mean-of-stds", means_of_stds)
    _print_moments("mean-of-skews", means_of_skews)
    _print_moments("mean-of-excess-kurt", means_of_kurts)
    return {
        "std_of_means": means_of_means,
        "mean_of_stds": means_of_stds,
        "mean_of_skews": means_of_skews,
        "mean_of_excess_kurt": means_of_kurts,
    }


def _valid_period_mask(W_hat: torch.Tensor, W_baseline: torch.Tensor) -> torch.Tensor:
    bad_hat = W_hat.isnan().any(dim=(1, 2))
    bad_base = W_baseline.isnan().any(dim=(1, 2))
    return ~(bad_hat | bad_base)


def lag1_autocorrelation(W_seq: torch.Tensor, valid: torch.Tensor) -> float:
    """Mean per-edge lag-1 autocorrelation across the valid (finite, non-degenerate) periods
    only -- answers Sec 10a item 1: is the classical baseline already autocorrelated "for free",
    which would make the model's smoothness-penalized stability margin weaker evidence of
    learned structure than it looks.
    """
    seq = W_seq[valid]
    if seq.shape[0] < 3:
        return math.nan
    n = seq.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool)
    flat = seq[:, mask]  # (T_valid, n*(n-1))
    x0, x1 = flat[:-1], flat[1:]
    x0c, x1c = x0 - x0.mean(dim=0), x1 - x1.mean(dim=0)
    num = (x0c * x1c).sum(dim=0)
    denom = (x0c.pow(2).sum(dim=0).sqrt() * x1c.pow(2).sum(dim=0).sqrt()).clamp_min(1e-12)
    per_edge = num / denom
    return per_edge[per_edge.isfinite()].mean().item() if per_edge.isfinite().any() else math.nan


def cap_binding_fraction(W_hat: torch.Tensor, cap: torch.Tensor, valid: torch.Tensor, tol: float = 1e-3) -> float:
    """Fraction of off-diagonal (bank, counterparty, period) cells where W_hat actually sits
    within `tol` (relative) of its row's regulatory cap -- Sec 10a item 2. cap is (N, T) (one
    bound per bank per period, broadcast across that bank's row); W_hat is (T, N, N).
    """
    n = W_hat.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool)
    total, binding = 0, 0
    for t in range(W_hat.shape[0]):
        if not bool(valid[t]):
            continue
        w_t = W_hat[t]
        cap_t = cap[:, t].unsqueeze(1).expand(n, n)  # row i's cap applied across row i
        close = (w_t >= cap_t * (1 - tol)) & mask
        total += int(mask.sum())
        binding += int(close.sum())
    return binding / total if total else math.nan


def run_section_b(seed: int, config_path: Path) -> dict | None:
    # Namespaced by the config file's stem, same convention run_real_experiment.py writes
    # under -- must match the --config that script was actually run with.
    artifact_path = RESULTS_DIR / config_path.stem / f"real_run_seed{seed}.pt"
    if not artifact_path.exists():
        print(
            f"\n=== Section B skipped: {artifact_path} not found ===\n"
            f"Run `uv run run_real_experiment.py --config {config_path}` with seed: {seed} in "
            "that config first to produce it."
        )
        return None

    data = torch.load(artifact_path, weights_only=False)
    W_hat, W_baseline, cap = data["W_hat"], data["W_baseline"], data["cap"]
    valid = _valid_period_mask(W_hat, W_baseline)
    print(f"\n=== Section B: trained-run checks (seed={seed}, {int(valid.sum())}/{len(valid)} valid periods) ===")

    model_edge_moments = edge_moments_over_time(W_hat, "model (W_hat)")
    baseline_edge_moments = edge_moments_over_time(W_baseline, "classical baseline (RAS/Sinkhorn)")

    print("\n=== Item 1: smoothness-penalty circularity -- lag-1 autocorrelation of raw edges ===")
    ac_model = lag1_autocorrelation(W_hat, valid)
    ac_baseline = lag1_autocorrelation(W_baseline, valid)
    print(f"  model autocorrelation:    {ac_model:.4f}")
    print(f"  baseline autocorrelation: {ac_baseline:.4f}")
    if ac_baseline == ac_baseline and ac_baseline > 0.5:
        print(
            "  NOTE: baseline is already substantially autocorrelated without any smoothness "
            "penalty -- the model's stability margin over RAS is weaker evidence of learned "
            "structure than the headline number alone suggests (Sec 10a item 1)."
        )

    print("\n=== Item 2: cap-binding fraction ===")
    frac = cap_binding_fraction(W_hat, cap, valid)
    print(f"  fraction of off-diagonal cells at/near the regulatory cap: {frac:.4f}")
    if frac == frac and frac < 0.01:
        print(
            "  NOTE: the cap almost never binds -- the model's cap-awareness may not be doing "
            "real work on this panel, independent of the stability-comparison fairness question "
            "(Sec 10a item 2)."
        )

    return {
        "n_valid_periods": int(valid.sum()),
        "n_periods": len(valid),
        "edge_moments": {"model": model_edge_moments, "baseline": baseline_edge_moments},
        "lag1_autocorrelation": {"model": ac_model, "baseline": ac_baseline},
        "cap_binding_fraction": frac,
    }


# ---------------------------------------------------------------------------
# Section C -- network-structure/systemic-risk literature checks. Pure torch,
# NaN/valid-mask-aware, same style as Section B's cap_binding_fraction/lag1_autocorrelation.
# Runs on the real trained artifact (results/real_run_seed{seed}.pt) and, independently, on the
# synthetic artifact (results/seed_{seed}.pt).
# ---------------------------------------------------------------------------


def marginal_preservation_error(
    W: torch.Tensor, r_target: torch.Tensor, c_target: torch.Tensor, valid: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Relative error between W's achieved row/col sums and the empirical target marginals the
    projection was supposed to hit -- the standard maxent/RAS sanity check this codebase never
    reported: row sum should equal the real measured interbank lending, column sum the measured
    borrowing, not just an aggregate Frobenius accuracy number. r_target/c_target are (T, N),
    same period-major convention as W (T, N, N).
    """
    row_err = (W.sum(dim=2) - r_target).abs() / r_target.abs().clamp_min(1e-8)
    col_err = (W.sum(dim=1) - c_target).abs() / c_target.abs().clamp_min(1e-8)
    return {"row": moments(row_err[valid].flatten()), "col": moments(col_err[valid].flatten())}


def density(W: torch.Tensor, valid: torch.Tensor, tol: float = 1e-3) -> dict[str, float]:
    """Fraction of off-diagonal cells >= tol per period -- same tolerance convention
    cap_binding_fraction already uses. A classical RAS/maxent baseline is well known (Cimini et
    al. 2015) to over-densify relative to the sparse true network; comparing this across
    W_hat/W_baseline/W_true is the standard check for that.
    """
    n = W.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool, device=W.device)
    per_period = []
    for t in range(W.shape[0]):
        if not bool(valid[t]):
            continue
        nonzero = (W[t].abs() >= tol) & mask
        per_period.append(nonzero.sum().item() / mask.sum().item())
    return moments(torch.tensor(per_period) if per_period else torch.tensor([]))


def weighted_reciprocity(W: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Garlaschelli-Loffredo (2004) weighted reciprocity per period:
    sum(min(w_ij, w_ji)) / sum(w_ij) over off-diagonal cells. A pure maxent/RAS reconstruction
    from marginals alone has no mechanism to prefer reciprocated exposures, so this is the
    standard check for whether a model is (or isn't) recovering real reciprocity structure
    beyond what the marginals alone would imply.
    """
    n = W.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool, device=W.device)
    per_period = []
    for t in range(W.shape[0]):
        if not bool(valid[t]):
            continue
        w = W[t]
        recip = torch.minimum(w, w.T)[mask].sum().item()
        total = w[mask].sum().item()
        per_period.append(recip / total if total > 0 else math.nan)
    return moments(torch.tensor(per_period) if per_period else torch.tensor([]))


def eigenvector_centrality(W: torch.Tensor, n_iters: int = 200, tol: float = 1e-8) -> torch.Tensor:
    """Power iteration on (W + W^T)/2 per period -- symmetrizing avoids picking an arbitrary
    convention between in-/out-centrality for a directed weighted network, and is the usual
    simplification when all that's actually needed is a single systemic-importance ranking to
    compare across W_hat/W_baseline/W_true (not a claim about directionality itself). Returns
    (T, N) centrality scores, L2-normalized per period. Invalid (NaN-containing) periods produce
    a NaN row here -- harmless, since every caller filters by a `valid` mask before aggregating.
    """
    t_len, n, _ = W.shape
    scores = torch.zeros(t_len, n, device=W.device)
    for t in range(t_len):
        A = (W[t] + W[t].T) / 2
        v = torch.ones(n, device=W.device) / n**0.5
        for _ in range(n_iters):
            v_new = A @ v
            norm = v_new.norm()
            if norm < tol:
                break
            v_new = v_new / norm
            converged = (v_new - v).norm() < tol
            v = v_new
            if converged:
                break
        scores[t] = v
    return scores


def rank_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation, hand-rolled as Pearson-on-ranks (double argsort) to avoid
    adding scipy to this script -- used to compare two centrality rankings (model vs baseline,
    or vs ground truth) without caring about the two scores' absolute scale.
    """
    def _rank(x: torch.Tensor) -> torch.Tensor:
        return x.argsort().argsort().float()

    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return (ra @ rb / denom).item() if denom > 0 else math.nan


def centrality_rank_agreement(scores_a: torch.Tensor, scores_b: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Per-period Spearman agreement between two centrality rankings, summarized across periods
    -- answers "does the reconstruction preserve WHO the systemically important banks are", the
    standard practical bar in this literature when exact edge recovery is hard but identifying
    key players still matters (Cimini et al. 2015; Battiston et al. 2012).
    """
    per_period = [rank_correlation(scores_a[t], scores_b[t]) for t in range(scores_a.shape[0]) if bool(valid[t])]
    return moments(torch.tensor(per_period) if per_period else torch.tensor([]))


def debtrank(W: torch.Tensor, equity: torch.Tensor, shocked: int, h_cap: float = 1.0, n_iters: int = 100, tol: float = 1e-6) -> float:
    """Battiston et al. (2012) DebtRank for a single period: shocking bank `shocked` to full
    distress (h=1) and propagating via exposure-weighted impact A[i, j] = min(1, W[i, j] /
    equity_i) -- how much of creditor i's equity a full default by debtor j would wipe out
    (W[i, j] is i's lending exposure to j, per data_loader.Panel's row=lending/col=borrowing
    convention) -- iterating h_i <- min(1, h_i + sum_j A[i,j] * h_j * (1 - h_i)) to convergence.
    Returns total systemic impact sum(h) at convergence, the actual contagion-estimate output
    this literature cares about, not just whether edges/weights match.
    """
    n = W.shape[-1]
    A = (W / equity.clamp_min(1e-8).unsqueeze(1)).clamp(max=h_cap)
    h = torch.zeros(n, device=W.device)
    h[shocked] = h_cap
    for _ in range(n_iters):
        increment = (A * h.unsqueeze(0)).sum(dim=1) * (1 - h)
        h_new = (h + increment).clamp(max=h_cap)
        converged = (h_new - h).abs().max().item() < tol
        h = h_new
        if converged:
            break
    return h.sum().item()


def aggregate_debtrank(W: torch.Tensor, equity: torch.Tensor, valid: torch.Tensor) -> dict[str, dict[str, float]]:
    """Mean and worst-case (over which single bank is shocked) total DebtRank impact, averaged
    across valid periods -- the standard way this literature reports systemic risk from a
    reconstructed network: not one global number, but the distribution of impact under each
    possible single-bank shock. `equity` is (N, T) (real data's per-period cap) or (N,)
    (synthetic's time-invariant scenario.cap, broadcast across all periods).
    """
    n = W.shape[-1]
    per_period_mean, per_period_worst = [], []
    for t in range(W.shape[0]):
        if not bool(valid[t]):
            continue
        equity_t = equity[:, t] if equity.dim() == 2 else equity
        impacts = [debtrank(W[t], equity_t, i) for i in range(n)]
        per_period_mean.append(sum(impacts) / len(impacts))
        per_period_worst.append(max(impacts))
    empty = torch.tensor([])
    return {
        "mean_shock_impact": moments(torch.tensor(per_period_mean) if per_period_mean else empty),
        "worst_shock_impact": moments(torch.tensor(per_period_worst) if per_period_worst else empty),
    }


def run_marginal_and_density_checks(
    label: str,
    W: torch.Tensor,
    valid: torch.Tensor,
    r_target: torch.Tensor | None,
    c_target: torch.Tensor | None,
    tol: float,
) -> dict:
    print(f"\n=== Marginal preservation & density: {label} ===")
    result: dict = {}
    if r_target is not None and c_target is not None:
        mp = marginal_preservation_error(W, r_target, c_target, valid)
        _print_moments("row-sum rel. error", mp["row"])
        _print_moments("col-sum rel. error", mp["col"])
        result["marginal_preservation"] = mp
    density_result = density(W, valid, tol)
    reciprocity_result = weighted_reciprocity(W, valid)
    _print_moments("density (frac. off-diag >= tol)", density_result)
    _print_moments("weighted reciprocity", reciprocity_result)
    result["density"] = density_result
    result["weighted_reciprocity"] = reciprocity_result
    return result


def run_centrality_check(label_a: str, scores_a: torch.Tensor, label_b: str, scores_b: torch.Tensor, valid: torch.Tensor) -> dict:
    print(f"\n=== Eigenvector-centrality rank agreement: {label_a} vs {label_b} ===")
    agreement = centrality_rank_agreement(scores_a, scores_b, valid)
    _print_moments("Spearman rank correlation", agreement)
    if agreement["mean"] == agreement["mean"] and agreement["mean"] < 0.5:
        print(
            f"  NOTE: {label_a} and {label_b} disagree substantially on which banks are most "
            "systemically central -- a low Frobenius error does not guarantee the reconstruction "
            "identifies the same key players."
        )
    return {"rank_correlation": agreement}


def run_debtrank_check(label: str, W: torch.Tensor, equity: torch.Tensor, valid: torch.Tensor) -> dict:
    print(f"\n=== DebtRank systemic impact: {label} ===")
    result = aggregate_debtrank(W, equity, valid)
    _print_moments("mean single-bank-shock impact", result["mean_shock_impact"])
    _print_moments("worst single-bank-shock impact", result["worst_shock_impact"])
    return result


def run_section_c_real(seed: int, config_path: Path, tol: float) -> dict | None:
    real_path = RESULTS_DIR / config_path.stem / f"real_run_seed{seed}.pt"
    if not real_path.exists():
        print(
            f"\nskip real-data network-structure checks: {real_path} not found\n"
            f"Run `uv run run_real_experiment.py --config {config_path}` with seed {seed} first."
        )
        return None

    data = torch.load(real_path, weights_only=False)
    W_hat, W_baseline, cap = data["W_hat"], data["W_baseline"], data["cap"]
    # Saved as (N, T) (Panel's own convention); transpose to (T, N) to match W_hat/
    # W_baseline's period-major (T, N, N) layout.
    r_target = data["r"].T if "r" in data else None
    c_target = data["c"].T if "c" in data else None
    valid = _valid_period_mask(W_hat, W_baseline)

    print(f"\n########## Section C: REAL DATA network-structure checks (config={config_path.stem}, seed={seed}) ##########")
    model_md = run_marginal_and_density_checks("model (W_hat)", W_hat, valid, r_target, c_target, tol)
    baseline_md = run_marginal_and_density_checks("classical baseline", W_baseline, valid, r_target, c_target, tol)

    centrality_hat = eigenvector_centrality(W_hat)
    centrality_base = eigenvector_centrality(W_baseline)
    centrality = run_centrality_check("model", centrality_hat, "baseline", centrality_base, valid)

    model_debtrank = run_debtrank_check("model (W_hat)", W_hat, cap, valid)
    baseline_debtrank = run_debtrank_check("classical baseline", W_baseline, cap, valid)

    return {
        "marginal_and_density": {"model": model_md, "baseline": baseline_md},
        "centrality_rank_agreement": centrality,
        "debtrank": {"model": model_debtrank, "baseline": baseline_debtrank},
    }


def run_section_c_synthetic(seed: int, synthetic_config_path: Path, tol: float) -> dict | None:
    synth_path = RESULTS_DIR / synthetic_config_path.stem / f"seed_{seed}.pt"
    if not synth_path.exists():
        print(
            f"\nskip synthetic network-structure checks: {synth_path} not found\n"
            f"Run `uv run run_synthetic_experiment.py --config {synthetic_config_path}` first."
        )
        return None

    cfg = load_synthetic_experiment_config(synthetic_config_path)
    data = torch.load(synth_path, weights_only=False)
    W_true, W_hat, W_baseline = data["W_true"], data["W_hat_final"], data["W_baseline"]
    # scenario.cap is a single scalar applied uniformly to every bank and period (see
    # run_synthetic_experiment.py's _build_panel) -- there's no separate equity concept in
    # the synthetic DGP, so this is the stand-in for equity in DebtRank, same role cap plays
    # for real data (cap = kappa * K, proportional to actual capital).
    equity = torch.full((cfg.scenario.n_banks,), cfg.scenario.cap)
    valid = _valid_period_mask(W_hat, W_baseline) & _valid_period_mask(W_hat, W_true)
    r_target, c_target = W_true.sum(dim=2), W_true.sum(dim=1)

    print(f"\n########## Section C: SYNTHETIC network-structure checks (config={synthetic_config_path.stem}, seed={seed}) ##########")
    model_md = run_marginal_and_density_checks("model (W_hat)", W_hat, valid, r_target, c_target, tol)
    baseline_md = run_marginal_and_density_checks("classical baseline", W_baseline, valid, r_target, c_target, tol)
    true_md = run_marginal_and_density_checks("ground truth (W_true)", W_true, valid, r_target, c_target, tol)

    centrality_hat = eigenvector_centrality(W_hat)
    centrality_base = eigenvector_centrality(W_baseline)
    centrality_true = eigenvector_centrality(W_true)
    centrality_model_vs_true = run_centrality_check("model", centrality_hat, "ground truth", centrality_true, valid)
    centrality_baseline_vs_true = run_centrality_check("baseline", centrality_base, "ground truth", centrality_true, valid)

    model_debtrank = run_debtrank_check("model (W_hat)", W_hat, equity, valid)
    baseline_debtrank = run_debtrank_check("classical baseline", W_baseline, equity, valid)
    true_debtrank = run_debtrank_check("ground truth (W_true)", W_true, equity, valid)

    return {
        "marginal_and_density": {"model": model_md, "baseline": baseline_md, "ground_truth": true_md},
        "centrality_rank_agreement": {"model_vs_true": centrality_model_vs_true, "baseline_vs_true": centrality_baseline_vs_true},
        "debtrank": {"model": model_debtrank, "baseline": baseline_debtrank, "ground_truth": true_debtrank},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seed", type=int, default=0)
    # Which variant's architecture to assume for param_count_vs_data_size below -- must match
    # whatever --config run_real_experiment.py was actually run with to produce the
    # real_run_seed{seed}.pt artifact read in Sections B and C.
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--synthetic-config", type=Path, default=DEFAULT_SYNTHETIC_CONFIG)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    eval_dir = RESULTS_DIR / args.config.stem / "evaluation"

    with captured_output() as buf:
        panel = load_panel(args.csv)
        panel = filter_commercial_banks(panel)
        panel = filter_from_period(panel)

        discard_bias = discard_bias_moments(panel)
        coverage = coverage_time_series(panel)
        r_t, c_t = marginal_time_series(panel)
        included = _complete_case_mask(panel)
        cfg = load_real_experiment_config(args.config)
        param_count = param_count_vs_data_size(int(included.sum()), len(panel.periods), cfg.model)
        _plot_section_a(coverage, r_t, c_t, eval_dir)

        section_b = run_section_b(args.seed, args.config)
        section_c_real = run_section_c_real(args.seed, args.config, args.tol)

    metrics = {
        "discard_bias": discard_bias,
        "coverage_counts": coverage,
        "r_t": r_t,
        "c_t": c_t,
        "param_count_vs_data_size": param_count,
        "section_b": section_b,
        "section_c_real": section_c_real,
    }
    save_report(eval_dir, "eda_model_checks", buf.getvalue(), metrics)

    synth_path = RESULTS_DIR / args.synthetic_config.stem / f"seed_{args.seed}.pt"
    if synth_path.exists():
        with captured_output() as buf:
            section_c_synthetic = run_section_c_synthetic(args.seed, args.synthetic_config, args.tol)
        save_report(
            RESULTS_DIR / args.synthetic_config.stem / "evaluation",
            "eda_model_checks",
            buf.getvalue(),
            {"section_c_synthetic": section_c_synthetic},
        )
    else:
        print(
            f"\nskip synthetic checks: {synth_path} not found\n"
            f"Run `uv run run_synthetic_experiment.py --config {args.synthetic_config}` first."
        )


if __name__ == "__main__":
    main()
