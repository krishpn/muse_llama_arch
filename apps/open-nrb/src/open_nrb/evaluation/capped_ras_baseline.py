#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "scipy",
#     "pyyaml",
# ]
# ///
"""Decompose the synthetic reconstruction win into cap vs learned-prior contributions.

Capped-RAS baseline (reviewer check 2, 2026-11-06): how much of the synthetic
reconstruction win over the classical uncapped Sinkhorn/RAS baseline (0.43 model vs 0.57 RAS in
sweep_lambda_mi_0) is the per-obligor CAP, and how much is the covariate-informed learned prior?

run_synthetic_experiment.py's RAS baseline (its _baseline_errors) balances a FLAT uniform
off-diagonal prior onto the observed marginals with *no* cap -- so the model enjoys two
advantages at once: the cap AND a learned, covariate-informed prior. This script adds the
missing middle rung: run the SAME capacitated_projection (Algorithm 1, the model's own
constraint layer -- row marginal + col marginal + per-entry cap) on the SAME flat uniform prior
RAS uses. That gives the baseline the cap but still no learned prior, so:

    uncapped RAS   : flat prior, no cap          (already in seed_{seed}.pt as baseline_mean)
    capped RAS     : flat prior, cap             (this script)
    model          : learned prior, cap          (already in seed_{seed}.pt as model_mean)

capped_RAS - uncapped_RAS  isolates the cap's contribution.
model      - capped_RAS    isolates the learned prior's contribution, cap held fixed.

Runs purely post-hoc from the saved seed_{seed}.pt artifacts (W_true, from which the achieved
marginals r=W_true.sum(1), c=W_true.sum(0) are exactly the marginals the model/RAS were projected
onto) plus the scenario cap from the config -- NO retraining. Point it at whichever config
produced the headline number (default: sweep_lambda_mi_0.yaml).

Usage: uv run -m open_nrb.evaluation.capped_ras_baseline [--synthetic-config PATH]
"""

import argparse
import sys
from pathlib import Path

import torch
from scipy import stats

try:
    from ..models.capacity_projection import capacitated_projection
    from ..models.config import CapacityProjectionConfig
    from ..models.experiment_config import load_synthetic_experiment_config
    from .report_io import captured_output, save_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capacity_projection import capacitated_projection
    from config import CapacityProjectionConfig
    from experiment_config import load_synthetic_experiment_config
    from report_io import captured_output, save_report

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_SYNTHETIC_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "sweep_lambda_mi_0.yaml"


def _capped_ras_error_for_period(W_true_t: torch.Tensor, cap_value: float, proj_config: CapacityProjectionConfig) -> float:
    """Compute the relative Frobenius error of the capped-RAS reconstruction for one period.

    Balances a flat uniform off-diagonal prior (ones - I) onto the achieved marginals
    r = W_true_t.sum(1), c = W_true_t.sum(0) under a uniform per-obligor cap via
    Algorithm 1, then scores against W_true -- same metric
    run_synthetic_experiment.py's _relative_frobenius_errors uses, so the number is
    directly comparable to the stored model_mean/baseline_mean:
        W_capped = capacitated_projection(ones - I, r, c, cap)
        error    = ||W_capped - W_true_t||_F / ||W_true_t||_F
    """
    n = W_true_t.shape[0]
    r = W_true_t.sum(dim=1)
    c = W_true_t.sum(dim=0)
    cap = torch.full((n,), cap_value, dtype=W_true_t.dtype, device=W_true_t.device)
    prior = torch.ones(n, n, dtype=W_true_t.dtype, device=W_true_t.device) - torch.eye(
        n, dtype=W_true_t.dtype, device=W_true_t.device
    )
    W_capped = capacitated_projection(prior, r, c, cap, proj_config)
    denom = W_true_t.norm().item()
    return (W_capped - W_true_t).norm().item() / denom if denom > 0 else float("nan")


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, var**0.5


def _paired_report(label: str, a: list[float], b: list[float]) -> dict:
    """Run a paired t-test on the per-seed gap and report the mean paired difference.

    Mirrors seed_significance.py's _paired_report (same n<=6 Wilcoxon floor caveat
    applies -- the mean paired difference is the number that carries information at 5
    seeds). For paired per-seed vectors a, b:
        d_i       = a_i - b_i
        mean_diff = (1/n) * sum_i d_i
        (t_stat, p) = scipy.stats.ttest_rel(a, b)
    and for n <= 6 a signed-rank test's minimum achievable two-sided p-value is
    1 / 2^(n-1), so it cannot reach p < 0.05; read the effect size instead.
    """
    diffs = [x - y for x, y in zip(a, b)]
    mean_diff = sum(diffs) / len(diffs)
    t_stat, t_p = stats.ttest_rel(a, b)
    print(f"\n=== {label} ===")
    print(f"  n_seeds={len(a)}  mean paired difference (a - b): {mean_diff:+.6f}")
    print(f"  paired t-test: statistic={t_stat:.4f}  p={t_p:.4f}")
    n = len(a)
    note = None
    if n <= 6:
        note = 1 / (2 ** (n - 1))
        print(f"  NOTE: at n={n} seeds a signed-rank test cannot reach p<0.05; read the effect size.")
    return {
        "n_seeds": n,
        "mean_paired_difference": mean_diff,
        "ttest_rel": {"statistic": float(t_stat), "p_value": float(t_p)},
        "wilcoxon_min_achievable_p": note,
    }


def main() -> None:
    """Score capped-RAS per seed and report the cap and learned-prior contributions.

    Loads each seed's W_true artifact, computes the capped-RAS error per period, then
    compares three post-hoc reconstructions (uncapped RAS, capped RAS, model) so that
    capped_RAS - uncapped_RAS isolates the cap and model - capped_RAS isolates the
    covariate-informed learned prior, each via a paired t-test across seeds.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-config", type=Path, default=DEFAULT_SYNTHETIC_CONFIG)
    args = parser.parse_args()
    cfg = load_synthetic_experiment_config(args.synthetic_config)
    stem = args.synthetic_config.stem
    config_results_dir = RESULTS_DIR / stem
    proj_config = CapacityProjectionConfig()

    with captured_output() as buf:
        print(f"\n########## Capped-RAS baseline: {stem} (cap={cfg.scenario.cap}, {len(cfg.seeds)} seeds) ##########")
        per_seed = []
        for seed in cfg.seeds:
            artifact_path = config_results_dir / f"seed_{seed}.pt"
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"{artifact_path} not found -- run `uv run -m open_nrb.models.run_synthetic_experiment "
                    f"--config {args.synthetic_config}` first so the per-seed W_true artifacts exist."
                )
            data = torch.load(artifact_path, map_location="cpu")
            W_true = data["W_true"]  # (N_PERIODS, N, N)
            period_errors = [
                _capped_ras_error_for_period(W_true[t], cfg.scenario.cap, proj_config) for t in range(W_true.shape[0])
            ]
            finite = [e for e in period_errors if e == e]
            capped_mean = sum(finite) / len(finite)
            per_seed.append(
                {
                    "seed": int(seed),
                    "capped_ras_mean": capped_mean,
                    "uncapped_ras_mean": float(data["baseline_mean"]),
                    "model_mean": float(data["model_mean"]),
                }
            )
            print(
                f"seed {seed}: capped_RAS={capped_mean:.4f}  "
                f"uncapped_RAS={data['baseline_mean']:.4f}  model={data['model_mean']:.4f}"
            )

        capped = [r["capped_ras_mean"] for r in per_seed]
        uncapped = [r["uncapped_ras_mean"] for r in per_seed]
        model = [r["model_mean"] for r in per_seed]

        print("\n--- aggregates (mean relative Frobenius error, lower is better) ---")
        for name, xs in [("capped_RAS", capped), ("uncapped_RAS", uncapped), ("model", model)]:
            m, s = _mean_std(xs)
            print(f"  {name:14s}: mean={m:.4f}  std={s:.4f}")

        cap_effect = _paired_report("cap contribution: capped_RAS vs uncapped_RAS", capped, uncapped)
        prior_effect = _paired_report("learned-prior contribution: model vs capped_RAS", model, capped)

        print(
            "\nreading: a negative 'capped_RAS vs uncapped_RAS' mean difference = the cap alone already "
            "improves RAS; a negative 'model vs capped_RAS' mean difference = the covariate-informed "
            "learned prior adds reconstruction accuracy BEYOND the cap. If model vs capped_RAS is ~0, "
            "the synthetic win is essentially all cap, and Section 5.2/5.4 must say so."
        )

    metrics = {
        "config": str(args.synthetic_config),
        "cap": cfg.scenario.cap,
        "per_seed": per_seed,
        "aggregates": {
            "capped_ras": dict(zip(("mean", "std"), _mean_std(capped))),
            "uncapped_ras": dict(zip(("mean", "std"), _mean_std(uncapped))),
            "model": dict(zip(("mean", "std"), _mean_std(model))),
        },
        "cap_contribution_capped_vs_uncapped": cap_effect,
        "learned_prior_contribution_model_vs_capped": prior_effect,
    }
    save_report(config_results_dir / "evaluation", "capped_ras_baseline", buf.getvalue(), metrics)


if __name__ == "__main__":
    main()
