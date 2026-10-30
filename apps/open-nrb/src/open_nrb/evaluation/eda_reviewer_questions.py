#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "matplotlib",
#     "pyyaml",
# ]
# ///
"""Reviewer pre-mortem EDA (design doc Sec 10a) -- descriptive-only, no new training.

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

Usage: uv run eda_reviewer_questions.py [--csv PATH] [--seed N]
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
    from ..models.experiment_config import ModelConfig, load_real_experiment_config
    from ..models.train import ReconstructionModel
    from .report_io import captured_output, save_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from config import EncoderDecoderConfig
    from data_loader import COVARIATE_ORDER, Panel, filter_commercial_banks, filter_from_period, load_panel
    from experiment_config import ModelConfig, load_real_experiment_config
    from train import ReconstructionModel
    from report_io import captured_output, save_report

DEFAULT_CSV = Path("src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv")
# results/ and experiments/ are both siblings of this script's parent dir (evaluation/) under
# src/open_nrb/, same __file__-relative convention as run_real_experiment.py's RESULTS_DIR/
# DEFAULT_CONFIG -- this script reads real_run_seed{seed}.pt from the same place that script
# wrote it, and loads the same baseline.yaml by default so param_count_vs_data_size below
# matches whatever architecture that artifact was actually trained with.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "baseline.yaml"

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
    path = out_dir / "eda_reviewer_questions_section_a.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {path}")


# ---------------------------------------------------------------------------
# Section B -- needs a trained run (results/real_run_seed{seed}.pt)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seed", type=int, default=0)
    # Which variant's architecture to assume for param_count_vs_data_size below -- must match
    # whatever --config run_real_experiment.py was actually run with to produce the
    # real_run_seed{seed}.pt artifact read in Section B.
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_real_experiment_config(args.config)

    eval_dir = RESULTS_DIR / args.config.stem / "evaluation"

    with captured_output() as buf:
        panel = load_panel(args.csv)
        panel = filter_commercial_banks(panel)
        panel = filter_from_period(panel)

        discard_bias = discard_bias_moments(panel)
        coverage = coverage_time_series(panel)
        r_t, c_t = marginal_time_series(panel)
        included = _complete_case_mask(panel)
        param_count = param_count_vs_data_size(int(included.sum()), len(panel.periods), cfg.model)
        _plot_section_a(coverage, r_t, c_t, eval_dir)

        section_b = run_section_b(args.seed, args.config)

    metrics = {
        "discard_bias": discard_bias,
        "coverage_counts": coverage,
        "r_t": r_t,
        "c_t": c_t,
        "param_count_vs_data_size": param_count,
        "section_b": section_b,
    }
    save_report(eval_dir, "eda_reviewer_questions", buf.getvalue(), metrics)


if __name__ == "__main__":
    main()
