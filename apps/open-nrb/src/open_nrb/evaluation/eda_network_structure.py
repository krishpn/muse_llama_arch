#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
# ]
# ///
"""Standard network-reconstruction/systemic-risk literature checks (Squartini & Garlaschelli
2011; Garlaschelli & Loffredo 2004; Cimini et al. 2015; Battiston et al. 2012) that
eda_reviewer_questions.py doesn't cover -- whether a reconstruction preserves the empirical
strength sequence/density/reciprocity structure (not just Frobenius accuracy), and whether it
preserves the systemic-risk-relevant structure (centrality ranking, DebtRank contagion estimate)
that's the actual reason a reconstructed interbank network is useful. Reads only from artifacts
run_real_experiment.py/run_synthetic_experiment.py already save -- never retrains.

Explicitly out of scope here: formal core-periphery block-fitting. Eigenvector centrality +
DebtRank already answer "does reconstruction preserve who's systemically important" without it.

Usage: uv run eda_network_structure.py [--seed N] [--config PATH] [--synthetic-config PATH] [--tol F]
"""

import argparse
import math
import sys
from pathlib import Path

import torch

try:
    from .eda_reviewer_questions import _print_moments, _valid_period_mask, moments
    from ..models.experiment_config import load_synthetic_experiment_config
    from .report_io import captured_output, save_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from eda_reviewer_questions import _print_moments, _valid_period_mask, moments
    from experiment_config import load_synthetic_experiment_config
    from report_io import captured_output, save_report

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "baseline.yaml"
DEFAULT_SYNTHETIC_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "synthetic_baseline.yaml"


# ---------------------------------------------------------------------------
# Metrics -- pure torch, NaN/valid-mask-aware, same style as eda_reviewer_questions.py's
# cap_binding_fraction/lag1_autocorrelation.
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


# ---------------------------------------------------------------------------
# Section runners -- compute + print + flag-if-notable, same shape as
# eda_reviewer_questions.py's discard_bias_moments/run_section_b.
# ---------------------------------------------------------------------------


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--synthetic-config", type=Path, default=DEFAULT_SYNTHETIC_CONFIG)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    real_path = RESULTS_DIR / args.config.stem / f"real_run_seed{args.seed}.pt"
    if real_path.exists():
        with captured_output() as buf:
            data = torch.load(real_path, weights_only=False)
            W_hat, W_baseline, cap = data["W_hat"], data["W_baseline"], data["cap"]
            # Saved as (N, T) (Panel's own convention); transpose to (T, N) to match W_hat/
            # W_baseline's period-major (T, N, N) layout.
            r_target = data["r"].T if "r" in data else None
            c_target = data["c"].T if "c" in data else None
            valid = _valid_period_mask(W_hat, W_baseline)

            print(f"\n########## REAL DATA (config={args.config.stem}, seed={args.seed}) ##########")
            model_md = run_marginal_and_density_checks("model (W_hat)", W_hat, valid, r_target, c_target, args.tol)
            baseline_md = run_marginal_and_density_checks("classical baseline", W_baseline, valid, r_target, c_target, args.tol)

            centrality_hat = eigenvector_centrality(W_hat)
            centrality_base = eigenvector_centrality(W_baseline)
            centrality = run_centrality_check("model", centrality_hat, "baseline", centrality_base, valid)

            model_debtrank = run_debtrank_check("model (W_hat)", W_hat, cap, valid)
            baseline_debtrank = run_debtrank_check("classical baseline", W_baseline, cap, valid)

        metrics = {
            "marginal_and_density": {"model": model_md, "baseline": baseline_md},
            "centrality_rank_agreement": centrality,
            "debtrank": {"model": model_debtrank, "baseline": baseline_debtrank},
        }
        save_report(RESULTS_DIR / args.config.stem / "evaluation", "eda_network_structure", buf.getvalue(), metrics)
    else:
        print(
            f"\nskip real-data checks: {real_path} not found\n"
            f"Run `uv run run_real_experiment.py --config {args.config}` with seed {args.seed} first."
        )

    synth_path = RESULTS_DIR / args.synthetic_config.stem / f"seed_{args.seed}.pt"
    if synth_path.exists():
        with captured_output() as buf:
            cfg = load_synthetic_experiment_config(args.synthetic_config)
            data = torch.load(synth_path, weights_only=False)
            W_true, W_hat, W_baseline = data["W_true"], data["W_hat_final"], data["W_baseline"]
            # scenario.cap is a single scalar applied uniformly to every bank and period (see
            # run_synthetic_experiment.py's _build_panel) -- there's no separate equity concept in
            # the synthetic DGP, so this is the stand-in for equity in DebtRank, same role cap plays
            # for real data (cap = kappa * K, proportional to actual capital).
            equity = torch.full((cfg.scenario.n_banks,), cfg.scenario.cap)
            valid = _valid_period_mask(W_hat, W_baseline) & _valid_period_mask(W_hat, W_true)
            r_target, c_target = W_true.sum(dim=2), W_true.sum(dim=1)

            print(f"\n########## SYNTHETIC (config={args.synthetic_config.stem}, seed={args.seed}) ##########")
            model_md = run_marginal_and_density_checks("model (W_hat)", W_hat, valid, r_target, c_target, args.tol)
            baseline_md = run_marginal_and_density_checks("classical baseline", W_baseline, valid, r_target, c_target, args.tol)
            true_md = run_marginal_and_density_checks("ground truth (W_true)", W_true, valid, r_target, c_target, args.tol)

            centrality_hat = eigenvector_centrality(W_hat)
            centrality_base = eigenvector_centrality(W_baseline)
            centrality_true = eigenvector_centrality(W_true)
            centrality_model_vs_true = run_centrality_check("model", centrality_hat, "ground truth", centrality_true, valid)
            centrality_baseline_vs_true = run_centrality_check("baseline", centrality_base, "ground truth", centrality_true, valid)

            model_debtrank = run_debtrank_check("model (W_hat)", W_hat, equity, valid)
            baseline_debtrank = run_debtrank_check("classical baseline", W_baseline, equity, valid)
            true_debtrank = run_debtrank_check("ground truth (W_true)", W_true, equity, valid)

        metrics = {
            "marginal_and_density": {"model": model_md, "baseline": baseline_md, "ground_truth": true_md},
            "centrality_rank_agreement": {"model_vs_true": centrality_model_vs_true, "baseline_vs_true": centrality_baseline_vs_true},
            "debtrank": {"model": model_debtrank, "baseline": baseline_debtrank, "ground_truth": true_debtrank},
        }
        save_report(RESULTS_DIR / args.synthetic_config.stem / "evaluation", "eda_network_structure", buf.getvalue(), metrics)
    else:
        print(
            f"\nskip synthetic checks: {synth_path} not found\n"
            f"Run `uv run run_synthetic_experiment.py --config {args.synthetic_config}` first."
        )


if __name__ == "__main__":
    main()
