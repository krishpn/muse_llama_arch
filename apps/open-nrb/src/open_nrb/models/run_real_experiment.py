#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///



from __future__ import annotations

import csv, re, json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import itertools
import argparse
import json
from pathlib import Path

"""Real-data validation experiment (design doc Sec 8/12): train the full
reconstruction model (Algorithm 3) against the real NRB balance-sheet panel
(data_loader.load_panel) and report the indirect evidence Sec 6/12 commit to
-- temporal stability of W_hat_t and a descriptive comparison against the
classical uncapped Sinkhorn/RAS baseline. There is no W_true here, so unlike
run_synthetic_experiment.py this script can never report a reconstruction- or
block-recovery-ACCURACY number; it reports what the real data actually
supports and nothing more.

Usage: uv run run_real_experiment.py [--csv PATH]
"""

import torch
import torch.nn.functional as F


try:
    from .config import CapacityProjectionConfig, EncoderDecoderConfig, ReparamConfig, SinkhornConfig, TrainStepConfig
    from .data_loader import COVARIATE_ORDER, Panel, load_panel
    from .run_synthetic_experiment import _forward_pass_components, _kl_beta_schedule, KL_BETA_MAX, KL_WARMUP_EPOCHS
    from .sinkhorn import sinkhorn_projection
    from .train import Period, ReconstructionModel, train_step
except ImportError:
    from config import CapacityProjectionConfig, EncoderDecoderConfig, ReparamConfig, SinkhornConfig, TrainStepConfig
    from data_loader import COVARIATE_ORDER, Panel, load_panel
    from run_synthetic_experiment import _forward_pass_components, _kl_beta_schedule, KL_BETA_MAX, KL_WARMUP_EPOCHS
    from sinkhorn import sinkhorn_projection
    from train import Period, ReconstructionModel, train_step


# N, K, D = 24, 2, 5  # banks, SBM blocks, covariate dim (matches COVARIATE_ORDER, Sec 8e)
# HIDDEN, DECODER_HIDDEN = 16, 16
# N_PERIODS = 16
# EPOCHS = 3000
# EVAL_EVERY = 100  # epochs between relative-error checkpoints, not just loss
# KL_WARMUP_EPOCHS = 300  # Sec 10: epochs over which Eq. 9's beta_t ramps 0 -> KL_BETA_MAX
# KL_BETA_MAX = 0.15  # Sec 10: beta_t's ceiling, NOT 1.0 -- see _kl_beta_schedule docstring
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RESULTS_DIR = Path(__file__).resolve().parent / "results"  # saved per-seed artifacts for later
#                                                             # plotting, so a figure never requires
#                                                             # re-running GPU training

# Relative to CWD at invocation (run from the repo root, not from inside paper1_model/) --
# same convention as verify_real_csv.py's default_path.
DEFAULT_CSV = Path("src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv")

D = len(COVARIATE_ORDER)
K = 2  # SBM blocks -- core/periphery, same convention as run_synthetic_experiment.py
HIDDEN, DECODER_HIDDEN = 16, 16
KAPPA = 0.05  # NRB's single-obligor exposure cap multiplier (confirmed by user, Sec 11)
EPOCHS = 3000  # first-pass pipeline smoke test, NOT a tuned final run -- real T is likely much
              # bigger than the synthetic script's 16, so per-epoch cost is unknown until this
              # runs once; bump after seeing real wall-clock
EVAL_EVERY = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Narrows the period window used for the complete-case bank filter below. Default is the full
# range load_panel finds; retune after seeing the real missingness printout if the full-range
# filter keeps too few banks (Nepal's BFI sector has a long merger history, Sec 8 note).
START_PERIOD: tuple[int, str] | None = None
END_PERIOD: tuple[int, str] | None = None

# Narrows the period window used for the complete-case bank filter below. Default is the full
# range load_panel finds; retune after seeing the real missingness printout if the full-range
# filter keeps too few banks (Nepal's BFI sector has a long merger history, Sec 8 note).
START_PERIOD: tuple[int, str] | None = None
END_PERIOD: tuple[int, str] | None = None


def _select_period_window(panel: Panel) -> list[int]:
    start_idx = 0 if START_PERIOD is None else panel.periods.index(START_PERIOD)
    end_idx = len(panel.periods) - 1 if END_PERIOD is None else panel.periods.index(END_PERIOD)
    return list(range(start_idx, end_idx + 1))


def _filter_complete_banks(
    panel: Panel,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, str]]]:
    """Keep only banks with zero NaN across the full period window on X/r/c/K (Sec 8 note:
    load_panel leaves NaN for (bank, period) cells where that bank has no rows that period --
    not chartered yet, exited, or merged away -- rather than raising). train_step's GRU needs a
    fixed N across the whole batch, so this filtering has to happen here, not inside train_step.

    Prints missingness diagnostics BEFORE filtering so a too-aggressive filter is visible
    immediately rather than silently shrinking N.
    """
    period_idx = _select_period_window(panel)
    n_total = len(panel.bfi_names)

    nan_any = (
        panel.X[:, period_idx, :].isnan().any(dim=(1, 2))
        | panel.r[:, period_idx].isnan().any(dim=1)
        | panel.c[:, period_idx].isnan().any(dim=1)
        | panel.K[:, period_idx].isnan().any(dim=1)
    )
    complete_mask = ~nan_any

    print(f"\n=== missingness diagnostics (period window: {len(period_idx)} periods) ===")
    print(f"banks total: {n_total}  complete (no NaN in window): {int(complete_mask.sum())}")
    per_period_missing = panel.X[:, period_idx, :].isnan().any(dim=2).float().mean(dim=0)
    worst = sorted(
        zip((panel.periods[i] for i in period_idx), per_period_missing.tolist()), key=lambda x: -x[1]
    )[:5]
    print("periods with most missing banks (top 5): " + ", ".join(f"{p}={frac:.2f}" for p, frac in worst))

    kept_names = [name for name, keep in zip(panel.bfi_names, complete_mask.tolist()) if keep]
    X = panel.X[complete_mask][:, period_idx, :]
    r = panel.r[complete_mask][:, period_idx]
    c = panel.c[complete_mask][:, period_idx]
    K_raw = panel.K[complete_mask][:, period_idx]
    period_labels = [panel.periods[i] for i in period_idx]
    return kept_names, X, r, c, K_raw, period_labels


def _build_real_periods(X: torch.Tensor, r: torch.Tensor, c: torch.Tensor, K_raw: torch.Tensor) -> list[Period]:
    n, t, _ = X.shape
    cap = KAPPA * K_raw
    return [
        Period(X=X[:, time, :], r=r[:, time], c=c[:, time], cap=cap[:, time], W_true=None)
        for time in range(t)
    ]


def _temporal_stability(W_seq: list[torch.Tensor]) -> list[float]:
    """||W_t - W_{t-1}||_F / ||W_t||_F for consecutive periods -- smaller means a more
    temporally stable reconstruction. No ground truth exists for real data, so this (and the
    baseline comparison below) stand in for the accuracy numbers run_synthetic_experiment.py
    reports (Sec 6/12).
    """
    changes = []
    for prev, curr in zip(W_seq[:-1], W_seq[1:]):
        denom = curr.norm().item()
        changes.append((curr - prev).norm().item() / denom if denom > 0 else float("nan"))
    return changes


def _baseline_sequence(periods: list[Period]) -> list[torch.Tensor]:
    """Classical uncapped Sinkhorn/RAS baseline (Sec 7) applied per period against the same
    observed marginals the model gets -- no covariates, no cap, matching what the classical
    method actually is.
    """
    sinkhorn_config = SinkhornConfig()
    baseline = []
    for period in periods:
        n = period.X.shape[0]
        prior = torch.ones(n, n, device=period.X.device) - torch.eye(n, device=period.X.device)
        baseline.append(sinkhorn_projection(prior, period.r, period.c, sinkhorn_config))
    return baseline


def _relative_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> list[float]:
    diffs = []
    for x, y in zip(a, b):
        denom = x.norm().item()
        diffs.append((x - y).norm().item() / denom if denom > 0 else float("nan"))
    return diffs


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _top_banks_by_block(bfi_names: list[str], q_seq: list[torch.Tensor], k: int, top_n: int = 10) -> dict[int, list]:
    """Descriptive, qualitative check only: there is no ground truth z for real data, so this
    is NOT a block-recovery-accuracy claim (contrast run_synthetic_experiment.py's
    _block_recovery_accuracy) -- just the model's own mean posterior per bank, for a human to
    sanity-check cluster identity against known bank names.
    """
    q_mean = torch.stack(q_seq).mean(dim=0)  # (N, K), time-averaged posterior-predictive mean
    report = {}
    for block in range(k):
        scores = q_mean[:, block].tolist()
        ranked = sorted(zip(bfi_names, scores), key=lambda x: -x[1])[:top_n]
        report[block] = ranked
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    print(f"device: {DEVICE}")
    panel = load_panel(args.csv)
    bfi_names, X, r, c, K_raw, period_labels = _filter_complete_banks(panel)
    n, t = len(bfi_names), len(period_labels)
    print(f"\ntraining panel: N={n} banks, T={t} periods")
    if n == 0 or t < 2:
        raise RuntimeError(
            f"complete-case filter left N={n} banks, T={t} periods -- too small to train "
            "(adjust START_PERIOD/END_PERIOD and retry)"
        )

    X, r, c, K_raw = X.to(DEVICE), r.to(DEVICE), c.to(DEVICE), K_raw.to(DEVICE)

    # Real NRB values are in NPR millions, a wildly different scale than the synthetic
    # experiment's convenient O(1-10) scale the model's hyperparameters (lambda_smooth, lr,
    # HIDDEN) were tuned for -- confirmed live: loss stuck flat (24,650,148 -> 24,649,156 over
    # 300 epochs) with kl_beta ramped all the way to its ceiling, i.e. nothing was learning.
    # Standardizing X per covariate and rescaling r/c/K (hence cap=kappa*K) by one global scalar
    # brings everything to a comparable training scale. This is lossless: capacitated_projection
    # is scale-equivariant (scaling r,c,cap by s scales the feasible W by s too), so W_hat/
    # W_baseline are multiplied back by SCALE before persistence/reporting, restoring real
    # NPR-million units. The temporal-stability/baseline-diff metrics are relative (ratios), so
    # unaffected either way.
    X_mean = X.mean(dim=(0, 1), keepdim=True)
    X_std = X.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    X = (X - X_mean) / X_std

    SCALE = torch.cat([r.flatten(), c.flatten()]).mean().clamp_min(1e-6)
    print(f"rescaling r/c/K by SCALE={SCALE.item():.4f} (real NPR-million magnitudes -> training scale)")
    r, c, K_raw = r / SCALE, c / SCALE, K_raw / SCALE

    periods = _build_real_periods(X, r, c, K_raw)

    torch.manual_seed(0)
    enc_config = EncoderDecoderConfig(input_dim=D, hidden_dim=HIDDEN, num_blocks=K, decoder_hidden_dim=DECODER_HIDDEN)
    model = ReconstructionModel(enc_config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    prior_probs = (torch.ones(K) / K).to(DEVICE)
    projection_config = CapacityProjectionConfig()
    reparam_config = ReparamConfig()
    train_config = TrainStepConfig(lambda_smooth=0.05, lambda_reg=0.001)
    generator = torch.Generator(device=DEVICE).manual_seed(0)

    losses = []
    # (epoch, kl, smooth_weighted, reg_weighted) -- recon is omitted, it's always exactly 0 here
    # (W_true=None every real Period, Sec 8). Needed to tell a benign converged fixed point apart
    # from e.g. KL already collapsed to the uniform prior well before loss itself visibly
    # flattens -- the only two terms with ANY gradient signal for q_psi(z)/the decoder on real
    # data are this kl term and smooth_weighted, so if both are themselves flat here, the loss
    # plateau really is the whole story, not something hidden by the single scalar loss.
    components_trend = []
    for epoch in range(EPOCHS):
        train_config.kl_beta = _kl_beta_schedule(epoch, KL_WARMUP_EPOCHS, KL_BETA_MAX)
        loss = train_step(
            model, optimizer, periods, prior_probs, projection_config, reparam_config, train_config,
            generator=generator,
        )
        losses.append(loss.item())
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            _, _, components = _forward_pass_components(model, periods, prior_probs, projection_config, train_config)
            components_trend.append(
                (epoch, components["kl"], components["smooth_weighted"], components["reg_weighted"])
            )
            print(
                f"epoch {epoch:5d}  loss={loss.item():10.3f}  kl_beta={train_config.kl_beta:.3f}  "
                f"kl={components['kl']:8.3f}  smooth_w={components['smooth_weighted']:8.3f}  "
                f"reg_w={components['reg_weighted']:7.4f}"
            )

    W_hats, qs, _ = _forward_pass_components(model, periods, prior_probs, projection_config, train_config)
    W_baseline = _baseline_sequence(periods)
    W_hats = [w * SCALE for w in W_hats]
    W_baseline = [w * SCALE for w in W_baseline]

    model_stability = _temporal_stability(W_hats)
    baseline_stability = _temporal_stability(W_baseline)
    baseline_diff = _relative_diff(W_hats, W_baseline)
    top_banks = _top_banks_by_block(bfi_names, qs, K)

    print(f"\n=== REAL-DATA VALIDATION (Sec 8/12) ===")
    print(f"loss: first={losses[0]:.3f} last={losses[-1]:.3f}")
    print(f"model temporal stability (mean rel. change, lower=more stable): {_mean(model_stability):.4f}")
    print(f"baseline temporal stability (mean rel. change, lower=more stable): {_mean(baseline_stability):.4f}")
    print(
        "model is MORE temporally stable than classical baseline"
        if _mean(model_stability) < _mean(baseline_stability)
        else "model is NOT more temporally stable than classical baseline -- report honestly (Sec 7)"
    )
    print(f"mean relative difference from classical baseline (descriptive, no ground truth): "
          f"{_mean(baseline_diff):.4f}")
    for block, ranked in top_banks.items():
        print(f"\ntop banks, block {block} (mean posterior over all periods):")
        for name, score in ranked:
            print(f"  {name:40s} {score:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = RESULTS_DIR / "real_run.pt"
    torch.save(
        {
            "bfi_names": bfi_names,
            "periods": period_labels,
            "W_hat": torch.stack(W_hats).cpu(),
            "W_baseline": torch.stack(W_baseline).cpu(),
            "q": torch.stack(qs).cpu(),
            "losses": losses,
            "components_trend": components_trend,
            "model_stability": model_stability,
            "baseline_stability": baseline_stability,
            "baseline_diff": baseline_diff,
        },
        artifact_path,
    )
    print(f"\nsaved: {artifact_path}")

    summary_path = RESULTS_DIR / "real_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "csv": str(args.csv),
                "n_banks": n,
                "n_periods": t,
                "epochs": EPOCHS,
                "kl_beta_max": KL_BETA_MAX,
                "kl_warmup_epochs": KL_WARMUP_EPOCHS,
                "kappa": KAPPA,
                "model_stability_mean": _mean(model_stability),
                "baseline_stability_mean": _mean(baseline_stability),
                "baseline_diff_mean": _mean(baseline_diff),
            },
            f,
            indent=2,
        )
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()