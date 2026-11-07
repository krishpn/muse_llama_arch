#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
# ]
# ///
"""Score a reprojected-persistence baseline against the model on the real panel.

Persistence baseline on REAL data (reviewer check 1, 2026-11-06): is the model's temporal
stability edge in Section 6.5 (model mean rel. change ~1.16 vs classical RAS ~2.32) real added
value, or could a trivial "carry last month forward" baseline look just as stable?

IMPORTANT -- why this is a REAL-data check, not synthetic. On the synthetic panel the generator
(run_synthetic_experiment.py::_build_panel) redraws the block labels z AND the covariates X
independently every period; sample_synthetic_network has no time index at all. So consecutive
W_true_t are i.i.d. draws with no temporal dependence, and a persistence baseline Ŵ_t := Ŵ_{t-1}
there is tested against an independent target -- it is guaranteed to look terrible and says
NOTHING about the real-data smoothness claim. The persistence concern only bites on the real
panel, where months genuinely are autocorrelated. Hence this script reads real_run_seed{seed}.pt.

A raw Ŵ_t := Ŵ_{t-1} is not even a valid reconstruction of month t (it violates month t's
observed marginals), so the honest persistence baseline is REPROJECTED persistence: take last
month's model reconstruction and re-impose THIS month's marginals + cap via the same Algorithm 1
the model uses:
    W_persist_t = capacitated_projection(W_hat_{t-1}, r_t, c_t, cap_t)

Two numbers come out, both computed post-hoc from the saved artifact (W_hat, r, c, cap all in
real NPR-million units) -- NO retraining:

  persistence_stability  = ||W_persist_t - W_hat_{t-1}|| / ||W_persist_t||
      the stability FLOOR: how much reprojecting last month onto this month's marginals alone
      moves the matrix. You cannot be more temporally stable than this while still fitting each
      month's marginals. Compare against the stored model_stability (~1.16) and baseline
      RAS stability (~2.32).

  model_vs_persistence   = ||W_hat_t - W_persist_t|| / ||W_hat_t||
      how far the model's OWN reconstruction departs from carry-forward-and-reproject. Large =>
      the model is genuinely covariate-driven, not just inertia; ~0 => it is essentially
      persistence and Section 6.5 should say the stability edge is really "RAS over-reacts to
      marginal noise," not "the model adds smoothness."

Usage: uv run -m open_nrb.evaluation.persistence_baseline [--config PATH]   (real config, e.g. baseline.yaml)
"""

import argparse
import sys
from pathlib import Path

import torch

try:
    from ..models.capacity_projection import capacitated_projection
    from ..models.config import CapacityProjectionConfig
    from ..models.experiment_config import load_real_experiment_config
    from .report_io import captured_output, save_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capacity_projection import capacitated_projection
    from config import CapacityProjectionConfig
    from experiment_config import load_real_experiment_config
    from report_io import captured_output, save_report

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "baseline.yaml"


def _mean(xs: list[float]) -> float:
    finite = [x for x in xs if x == x]  # NaN != NaN
    return sum(finite) / len(finite) if finite else float("nan")


def main() -> None:
    """Load the real reconstruction artifact and report persistence stability vs the model.

    For each consecutive transition t-1 -> t (skipping non-finite/degenerate periods),
    reprojects last month's reconstruction onto this month's marginals and cap via
    Algorithm 1, then reports two Frobenius-norm ratios averaged over scored transitions:
        W_persist_t           = capacitated_projection(W_hat_{t-1}, r_t, c_t, cap_t)
        persistence_stability = ||W_persist_t - W_hat_{t-1}|| / ||W_persist_t||
        model_vs_persistence  = ||W_hat_t - W_persist_t|| / ||W_hat_t||
    plus a verdict comparing the stored model stability to the persistence floor.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_real_experiment_config(args.config)
    seed = cfg.seed
    stem = args.config.stem
    config_results_dir = RESULTS_DIR / stem
    artifact_path = config_results_dir / f"real_run_seed{seed}.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"{artifact_path} not found -- run `uv run -m open_nrb.models.run_real_experiment "
            f"--config {args.config}` first so the real reconstruction artifact exists."
        )

    proj_config = CapacityProjectionConfig()
    data = torch.load(artifact_path, map_location="cpu")
    W_hat = data["W_hat"]  # (T, N, N), real NPR-million units
    r = data["r"]  # (N, T), real units
    c = data["c"]  # (N, T)
    cap = data["cap"]  # (N, T)
    periods = data["periods"]
    t_total = W_hat.shape[0]
    bad = set(data.get("non_finite_periods", [])) | set(data.get("degenerate_periods", []))

    with captured_output() as buf:
        print(f"\n########## Persistence baseline (real): {stem} seed {seed} (T={t_total} periods) ##########")
        print(f"excluded periods (non-finite model output or ~zero interbank activity): {sorted(bad)}")

        persistence_stability: list[float] = []
        model_vs_persistence: list[float] = []
        n_scored = 0
        for t in range(1, t_total):
            if t in bad or (t - 1) in bad:
                persistence_stability.append(float("nan"))
                model_vs_persistence.append(float("nan"))
                continue
            W_persist = capacitated_projection(W_hat[t - 1], r[:, t], c[:, t], cap[:, t], proj_config)
            denom_p = W_persist.norm().item()
            persistence_stability.append((W_persist - W_hat[t - 1]).norm().item() / denom_p if denom_p > 0 else float("nan"))
            denom_m = W_hat[t].norm().item()
            model_vs_persistence.append((W_hat[t] - W_persist).norm().item() / denom_m if denom_m > 0 else float("nan"))
            n_scored += 1

        model_stability_stored = _mean(data.get("model_stability", []))
        baseline_stability_stored = _mean(data.get("baseline_stability", []))

        print(f"\nscored {n_scored}/{t_total - 1} consecutive-period transitions")
        print("\n--- temporal stability (mean rel. change month-to-month, lower = more stable) ---")
        print(f"  reprojected-persistence stability FLOOR : {_mean(persistence_stability):.4f}")
        print(f"  model stability (stored, Section 6.5)   : {model_stability_stored:.4f}")
        print(f"  classical RAS stability (stored)        : {baseline_stability_stored:.4f}")
        print(f"\n  model departure from persistence (||W_hat - W_persist|| / ||W_hat||): {_mean(model_vs_persistence):.4f}")

        floor = _mean(persistence_stability)
        within_15pct = floor == floor and floor > 0 and abs(model_stability_stored - floor) / floor < 0.15
        if model_stability_stored == model_stability_stored and floor == floor and model_stability_stored < floor and not within_15pct:
            verdict = (
                "model is MORE stable than the persistence floor -- it is over-smoothing relative to what "
                "marginal-fitting forces; Section 6.5/7 should flag this as possible over-regularization."
            )
        elif within_15pct:
            verdict = (
                "model stability ~ persistence floor (within 15%) -- the stability edge over RAS is really "
                "'RAS over-reacts to marginal noise', not 'the model adds smoothness'. State it that way in 6.5."
            )
        else:
            verdict = (
                "model moves meaningfully more than the persistence floor while staying far below RAS -- "
                "consistent with a genuinely covariate-driven (not inertial) reconstruction; Section 6.5 holds."
            )
        print(f"\nverdict: {verdict}")

    metrics = {
        "config": str(args.config),
        "seed": seed,
        "n_scored_transitions": n_scored,
        "excluded_periods": sorted(bad),
        "persistence_stability_floor_mean": _mean(persistence_stability),
        "model_stability_stored_mean": model_stability_stored,
        "baseline_ras_stability_stored_mean": baseline_stability_stored,
        "model_vs_persistence_mean": _mean(model_vs_persistence),
        "persistence_stability_per_transition": persistence_stability,
        "model_vs_persistence_per_transition": model_vs_persistence,
    }
    save_report(config_results_dir / "evaluation", "persistence_baseline", buf.getvalue(), metrics)


if __name__ == "__main__":
    main()
