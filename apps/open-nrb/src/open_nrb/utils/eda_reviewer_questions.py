#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "matplotlib",
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
    from ..models.train import ReconstructionModel
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from config import EncoderDecoderConfig
    from data_loader import COVARIATE_ORDER, Panel, filter_commercial_banks, filter_from_period, load_panel
    from train import ReconstructionModel

DEFAULT_CSV = Path("src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv")
# evaluation/ is a sibling of results/ under src/open_nrb/, so parent.parent lands on
# src/open_nrb/ regardless of where the repo is cloned -- same location run_real_experiment.py's
# RESULTS_DIR writes to.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

FIGURES_DIR = RESULTS_DIR / "figures"

# Same architecture hyperparameters as run_real_experiment.py -- kept in sync manually rather
# than imported, since importing them would also import torch's CUDA-dependent DEVICE selection
# that this read-only script has no other reason to touch.
D = len(COVARIATE_ORDER)
K_BLOCKS = 2
HIDDEN, DECODER_HIDDEN = 16, 16


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


def discard_bias_moments(panel: Panel) -> None:
    print("\n=== Item 5: discard-sample bias (included vs. excluded, own-active-window basis) ===")
    included = _complete_case_mask(panel)
    excluded = ~included
    print(f"included (complete-case): {int(included.sum())}   excluded: {int(excluded.sum())}")
    for name, idx in COVARIATE_NAME_INDEX.items():
        size_like = _own_window_covariate_mean(panel, idx)
        print(f"\n covariate: {name}")
        _print_moments("included", moments(size_like[included]))
        _print_moments("excluded", moments(size_like[excluded]))


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


def param_count_vs_data_size(n_banks: int, n_periods: int) -> None:
    enc_config = EncoderDecoderConfig(input_dim=D, hidden_dim=HIDDEN, num_blocks=K_BLOCKS, decoder_hidden_dim=DECODER_HIDDEN)
    model = ReconstructionModel(enc_config)
    n_params = sum(p.numel() for p in model.parameters())
    n_obs = n_banks * n_periods
    print("\n=== Item 6: parameter count vs. effective real-data size ===")
    print(f"trainable parameters: {n_params}")
    print(f"N x T (effective real observations): {n_banks} x {n_periods} = {n_obs}")
    print(f"params / (N x T): {n_params / n_obs:.3f}" if n_obs else "N x T is zero -- cannot ratio")


def _plot_section_a(coverage: list[int], r_t: list[float], c_t: list[float]) -> None:
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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "eda_reviewer_questions_section_a.png"
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


def edge_moments_over_time(W_seq: torch.Tensor, label: str) -> None:
    print(f"\n=== Edge-weight moments over time: {label} ===")
    per_period = [moments(_off_diagonal(W_seq[t])) for t in range(W_seq.shape[0])]
    means = [m["mean"] for m in per_period]
    stds = [m["std"] for m in per_period]
    skews = [m["skew"] for m in per_period]
    kurts = [m["kurtosis_excess"] for m in per_period]
    print(f"  mean of per-period means:  {sum(v for v in means if v == v) / max(1, sum(1 for v in means if v == v)):.4f}")
    _print_moments("std-of-means (drift)", moments(torch.tensor(means)))
    _print_moments("mean-of-stds", moments(torch.tensor(stds)))
    _print_moments("mean-of-skews", moments(torch.tensor(skews)))
    _print_moments("mean-of-excess-kurt", moments(torch.tensor(kurts)))


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


def run_section_b(seed: int) -> None:
    artifact_path = RESULTS_DIR / f"real_run_seed{seed}.pt"
    if not artifact_path.exists():
        print(
            f"\n=== Section B skipped: {artifact_path} not found ===\n"
            f"Run `uv run run_real_experiment.py --seed {seed}` first to produce it."
        )
        return

    data = torch.load(artifact_path, weights_only=False)
    W_hat, W_baseline, cap = data["W_hat"], data["W_baseline"], data["cap"]
    valid = _valid_period_mask(W_hat, W_baseline)
    print(f"\n=== Section B: trained-run checks (seed={seed}, {int(valid.sum())}/{len(valid)} valid periods) ===")

    edge_moments_over_time(W_hat, "model (W_hat)")
    edge_moments_over_time(W_baseline, "classical baseline (RAS/Sinkhorn)")

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    panel = load_panel(args.csv)
    panel = filter_commercial_banks(panel)
    panel = filter_from_period(panel)

    discard_bias_moments(panel)
    coverage = coverage_time_series(panel)
    r_t, c_t = marginal_time_series(panel)
    included = _complete_case_mask(panel)
    param_count_vs_data_size(int(included.sum()), len(panel.periods))
    _plot_section_a(coverage, r_t, c_t)

    run_section_b(args.seed)


if __name__ == "__main__":
    main()
