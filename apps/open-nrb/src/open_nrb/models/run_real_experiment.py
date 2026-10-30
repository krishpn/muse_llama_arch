#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
# ]
# ///
"""Real-data validation experiment (design doc Sec 8/12): train the full
reconstruction model (Algorithm 3) against the real NRB balance-sheet panel
(data_loader.load_panel) and report the indirect evidence Sec 6/12 commit to
-- temporal stability of W_hat_t and a descriptive comparison against the
classical uncapped Sinkhorn/RAS baseline. There is no W_true here, so unlike
run_synthetic_experiment.py this script can never report a reconstruction- or
block-recovery-ACCURACY number; it reports what the real data actually
supports and nothing more.

Usage: uv run run_real_experiment.py [--csv PATH] [--config PATH]
"""

import argparse
import json
import warnings
from pathlib import Path

import torch

try:
    from .config import CapacityProjectionConfig, EncoderDecoderConfig, ReparamConfig, SinkhornConfig, TrainStepConfig
    from .data_loader import COVARIATE_ORDER, Panel, filter_commercial_banks, filter_from_period, load_panel
    from .experiment_config import load_real_experiment_config
    from .run_synthetic_experiment import _forward_pass_components, _kl_beta_schedule
    from .sinkhorn import sinkhorn_projection
    from .train import Period, ReconstructionModel, train_step
except ImportError:
    from config import CapacityProjectionConfig, EncoderDecoderConfig, ReparamConfig, SinkhornConfig, TrainStepConfig
    from data_loader import COVARIATE_ORDER, Panel, filter_commercial_banks, filter_from_period, load_panel
    from experiment_config import load_real_experiment_config
    from run_synthetic_experiment import _forward_pass_components, _kl_beta_schedule
    from sinkhorn import sinkhorn_projection
    from train import Period, ReconstructionModel, train_step


# Relative to CWD at invocation (run from the repo root, not from inside src/open_nrb/) --
# same convention as verify_real_csv.py's default_path.
DEFAULT_CSV = Path("src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv")

D = len(COVARIATE_ORDER)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# results/ and experiments/ are both siblings of this script's parent dir (models/) under
# src/open_nrb/ -- data/evaluation/models/results/tests/utils/experiments all live at the same
# depth, same __file__-relative convention as evaluation/make_plots.py's RESULTS_DIR. Safe
# under a real install too: both live INSIDE the installable package (src/open_nrb/), so this
# still resolves correctly from site-packages/open_nrb/... after `pip install`/`uv run` as a
# module, not just as a script.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "baseline.yaml"

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


def _build_real_periods(
    X: torch.Tensor, r: torch.Tensor, c: torch.Tensor, K_raw: torch.Tensor, kappa: float
) -> list[Period]:
    n, t, _ = X.shape
    cap = kappa * K_raw
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


def _baseline_sequence(periods: list[Period]) -> tuple[list[torch.Tensor], list[int]]:
    """Classical uncapped Sinkhorn/RAS baseline (Sec 7) applied per period against the same
    observed marginals the model gets -- no covariates, no cap, matching what the classical
    method actually is.

    Also returns the list of degenerate-marginal period indices (not just a count): a period
    with ~zero total interbank lending/borrowing has no real network to speak of, so "relative
    change in network structure" around that period is meaningless for the MODEL too, not just
    the baseline -- the caller excludes W_hat at these same indices from temporal-stability/
    diff aggregates, for symmetry with how the baseline already treats them (see main()).
    """
    sinkhorn_config = SinkhornConfig()
    baseline = []
    degenerate_periods: list[int] = []
    for t, period in enumerate(periods):
        n = period.X.shape[0]
        r, c = period.r, period.c
        r_total, c_total = r.sum(), c.sum()

        # Sinkhorn/IPF's multiplicative row/col balancing (u=r/Kv, v=c/K^Tu) is only well-posed
        # when sum(r) == sum(c) -- a feasible row/col-marginal transportation polytope exists.
        # Unlike capacitated_projection's additive Dykstra projections (always bounded, converges
        # to a least-violation compromise when infeasible), a sum(r) != sum(c) mismatch here
        # makes the u/v iterates diverge to inf/nan with no warning. Two distinct cases, confirmed
        # live on the real panel (baseline_stability/baseline_diff came back nan both times):
        #
        # 1. Both totals positive but unequal: each complete-case bank's counterparty can be one
        # of the OTHER real banks excluded by the complete-case filter, so this subset's total
        # lending need not equal its total borrowing in a given period (unlike the synthetic
        # experiment, where sample_synthetic_network guarantees balanced totals by construction).
        # Rescaling c to match r's total before projecting is the standard RAS/IPF fix for
        # mismatched control totals (not a hack specific to this codebase) -- the two marginals
        # are already scale-free row/col TARGETS being fit, not an externally-fixed ground
        # truth, so rescaling one to match the other's total is exactly the degree of freedom
        # IPF expects a caller to resolve before balancing.
        #
        # 2. EITHER total is exactly zero (confirmed still possible after fix 1, still produced
        # nan): for a nonnegative matrix, row sums summing to zero forces every row to be
        # all-zero, which forces every column sum to zero too -- so a zero r_total with a
        # positive c_total (or vice versa) is a genuine infeasibility, not a magnitude mismatch
        # rescaling can resolve. Plausible cause here: a bulletin extraction where the "a.
        # Interbank Lending"/"b. Interbank Borrowing" line simply didn't match for ANY bank in
        # one whole period, so _lookup_field_or_zero silently defaulted everyone to 0 that
        # period (Sec 9.4's leaf-omission convention, at panel-wide scale instead of per-bank).
        # The only matrix consistent with an all-zero side is the all-zero matrix, so return
        # that directly rather than calling sinkhorn_projection into a divide-by-zero; counted
        # and warned below rather than silently absorbed, since "how many periods" matters for
        # whether this is a handful of genuine zero-activity months or a bigger extraction gap.
        # Threshold is 1e-8, not exactly 0: r/c here are already SCALE-rescaled to ~O(1)
        # (main()), so a total that's tiny-but-not-exactly-zero (e.g. one bank with a
        # rounding-level residual while every other entry is exactly 0) can still blow up
        # c*(r_total/c_total) or Sinkhorn's internal u/v iterates into numerical garbage even
        # though it technically passes a strict ">0" check.
        if r_total <= 1e-8 or c_total <= 1e-8:
            degenerate_periods.append(t)
            baseline.append(torch.zeros(n, n, device=r.device, dtype=r.dtype))
            continue

        prior = torch.ones(n, n, device=period.X.device) - torch.eye(n, device=period.X.device)
        c = c * (r_total / c_total)
        baseline.append(sinkhorn_projection(prior, r, c, sinkhorn_config))
    if degenerate_periods:
        warnings.warn(
            f"{len(degenerate_periods)}/{len(periods)} periods have zero total interbank lending "
            "or borrowing across the complete-case bank subset -- baseline set to the all-zero "
            "matrix for those periods instead of calling sinkhorn_projection into a "
            "divide-by-zero",
            stacklevel=2,
        )
    return baseline, degenerate_periods


def _relative_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> list[float]:
    diffs = []
    for x, y in zip(a, b):
        denom = x.norm().item()
        diffs.append((x - y).norm().item() / denom if denom > 0 else float("nan"))
    return diffs


def _mean(xs: list[float]) -> float:
    """Skips NaN entries rather than letting one poison the whole aggregate (both callers --
    _temporal_stability, _relative_diff -- already emit NaN for individual degenerate periods
    deliberately, e.g. a period whose own W_hat/W_baseline norm is exactly 0, rather than
    guessing a substitute; a single such period shouldn't blank out all 76 others). Printed
    by the caller alongside the count skipped, since a handful of genuine zero-activity
    periods is very different from most of the panel being degenerate.
    """
    finite = [x for x in xs if x == x]  # x != x only for NaN, no math import needed
    return sum(finite) / len(finite) if finite else float("nan")


def _top_banks_by_block(bfi_names: list[str], q_seq: list[torch.Tensor], k: int, top_n: int = 10) -> dict[int, list]:
    """Descriptive, qualitative check only: there is no ground truth z for real data, so this
    is NOT a block-recovery-accuracy claim (contrast run_synthetic_experiment.py's
    _block_recovery_accuracy) -- just the model's own mean posterior per bank, for a human to
    sanity-check cluster identity against known bank names.
    """
    # nanmean, not mean: a period NaN'd out by _check_finite below would otherwise poison every
    # bank's block-posterior ranking for the whole run, not just that one period.
    q_mean = torch.stack(q_seq).nanmean(dim=0)  # (N, K), time-averaged posterior-predictive mean
    report = {}
    for block in range(k):
        scores = q_mean[:, block].tolist()
        ranked = sorted(zip(bfi_names, scores), key=lambda x: -x[1])[:top_n]
        report[block] = ranked
    return report


def _check_finite(W_hats: list[torch.Tensor], qs: list[torch.Tensor], period_labels: list[tuple[int, str]]) -> list[int]:
    """Flags periods where the MODEL's own W_hat/q is non-finite -- distinct from
    _baseline_sequence's degenerate-marginal case above (a well-understood, benign per-period
    data gap). A non-finite model output means something went numerically wrong in training or
    in that period's specific forward pass.

    A handful of isolated bad periods is reported and excluded from downstream aggregates, same
    count-and-warn convention as everywhere else here. But if MOST periods are affected, that's
    weight corruption (e.g. an exploding gradient or a log(0) in the KL/lognormal terms) -- every
    forward pass downstream of a NaN weight is NaN too, so this isn't a per-period gap to skip
    around. Continuing to report stability/top-bank numbers from a model that broken would be
    actively misleading for anything consuming this run (Sec 8/12's downstream systemic-risk
    analysis), so that case raises instead of warning.
    """
    bad = [t for t, (w, q) in enumerate(zip(W_hats, qs)) if not (w.isfinite().all() and q.isfinite().all())]
    if len(bad) > len(W_hats) // 2:
        raise RuntimeError(
            f"model produced non-finite W_hat/q for {len(bad)}/{len(W_hats)} periods (majority) "
            "-- this is weight corruption, not a benign per-period gap. Rerun with a lower lr / "
            "tighter KL_BETA_MAX / gradient clipping rather than excluding periods here -- a "
            "model this broken shouldn't feed downstream analysis at all."
        )
    if bad:
        warnings.warn(
            f"{len(bad)}/{len(W_hats)} periods have a non-finite model W_hat/q "
            f"(periods: {[period_labels[t] for t in bad]}) -- excluded from stability/diff/"
            "top-bank aggregates below, same convention as the classical baseline's degenerate "
            "periods",
            stacklevel=2,
        )
    return bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    # Picks the experiment variant (baseline, an ablation, ...) -- training hyperparameters,
    # model dims, kappa, and seed all come from this file now, not module constants or CLI
    # flags. A variant/ablation/sweep is a new YAML file, not a code or command-line change --
    # e.g. the posterior-collapse ablation (Sec 10's open item) is
    # experiments/configs/ablation_no_kl_pressure.yaml, and a multi-seed rerun (reviewer
    # pre-mortem item 1, design doc Sec 10a) is one YAML per seed.
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_real_experiment_config(args.config)
    seed = cfg.seed

    print(f"device: {DEVICE}")
    panel = load_panel(args.csv)
    panel = filter_commercial_banks(panel)
    panel = filter_from_period(panel)
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
    # hidden_dim) were tuned for -- confirmed live: loss stuck flat (24,650,148 -> 24,649,156 over
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

    periods = _build_real_periods(X, r, c, K_raw, cfg.kappa)

    torch.manual_seed(seed)
    enc_config = EncoderDecoderConfig(
        input_dim=D,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks,
        decoder_hidden_dim=cfg.model.decoder_hidden_dim,
    )
    model = ReconstructionModel(enc_config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    prior_probs = (torch.ones(cfg.model.num_blocks) / cfg.model.num_blocks).to(DEVICE)
    projection_config = CapacityProjectionConfig()
    reparam_config = ReparamConfig()
    train_config = TrainStepConfig(lambda_smooth=cfg.training.lambda_smooth, lambda_reg=cfg.training.lambda_reg)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    losses = []
    # (epoch, kl, smooth_weighted, reg_weighted) -- recon is omitted, it's always exactly 0 here
    # (W_true=None every real Period, Sec 8). Needed to tell a benign converged fixed point apart
    # from e.g. KL already collapsed to the uniform prior well before loss itself visibly
    # flattens -- the only two terms with ANY gradient signal for q_psi(z)/the decoder on real
    # data are this kl term and smooth_weighted, so if both are themselves flat here, the loss
    # plateau really is the whole story, not something hidden by the single scalar loss.
    components_trend = []
    for epoch in range(cfg.training.epochs):
        train_config.kl_beta = _kl_beta_schedule(epoch, cfg.training.kl_warmup_epochs, cfg.training.kl_beta_max)
        loss = train_step(
            model, optimizer, periods, prior_probs, projection_config, reparam_config, train_config,
            generator=generator,
        )
        losses.append(loss.item())
        if epoch % cfg.training.eval_every == 0 or epoch == cfg.training.epochs - 1:
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
    W_baseline, degenerate_periods = _baseline_sequence(periods)

    bad_periods = _check_finite(W_hats, qs, period_labels)
    for t in bad_periods:
        W_hats[t] = torch.full_like(W_hats[t], float("nan"))
        qs[t] = torch.full_like(qs[t], float("nan"))
    # Degenerate-marginal periods (~zero total interbank activity) have no real network to
    # compare transitions against for the MODEL either, not just the baseline -- NaN only
    # W_hat (not qs: block posterior comes from X covariates via the encoder, independent of
    # that period's r/c marginals, so it's still meaningful there) so _temporal_stability/
    # _relative_diff skip these the same way _baseline_sequence's own all-zero matrix does.
    for t in degenerate_periods:
        if t not in bad_periods:
            W_hats[t] = torch.full_like(W_hats[t], float("nan"))

    W_hats = [w * SCALE for w in W_hats]
    W_baseline = [w * SCALE for w in W_baseline]

    model_stability = _temporal_stability(W_hats)
    baseline_stability = _temporal_stability(W_baseline)
    baseline_diff = _relative_diff(W_hats, W_baseline)
    top_banks = _top_banks_by_block(bfi_names, qs, cfg.model.num_blocks)

    def _nan_count(xs: list[float]) -> int:
        return sum(1 for x in xs if x != x)

    print(f"\n=== REAL-DATA VALIDATION (Sec 8/12) ===")
    print(f"loss: first={losses[0]:.3f} last={losses[-1]:.3f}")
    print(f"non-finite model periods: {len(bad_periods)}/{len(W_hats)} (excluded from aggregates below, NaN-filled)")
    print(
        f"degenerate-marginal periods (~zero interbank activity): {len(degenerate_periods)}/{len(W_hats)} "
        "(W_hat NaN-filled for these too, excluded from model/baseline stability comparisons)"
    )
    print(
        f"model temporal stability (mean rel. change, lower=more stable): {_mean(model_stability):.4f} "
        f"({_nan_count(model_stability)}/{len(model_stability)} periods skipped, NaN)"
    )
    print(
        f"baseline temporal stability (mean rel. change, lower=more stable): {_mean(baseline_stability):.4f} "
        f"({_nan_count(baseline_stability)}/{len(baseline_stability)} periods skipped, NaN)"
    )
    print(
        "model is MORE temporally stable than classical baseline"
        if _mean(model_stability) < _mean(baseline_stability)
        else "model is NOT more temporally stable than classical baseline -- report honestly (Sec 7)"
    )
    print(
        f"mean relative difference from classical baseline (descriptive, no ground truth): "
        f"{_mean(baseline_diff):.4f} ({_nan_count(baseline_diff)}/{len(baseline_diff)} periods skipped, NaN)"
    )
    for block, ranked in top_banks.items():
        print(f"\ntop banks, block {block} (mean posterior over all periods):")
        for name, score in ranked:
            print(f"  {name:40s} {score:.4f}")

    # Namespaced by the config file's stem (e.g. results/baseline/, results/
    # ablation_no_kl_pressure/), not flat under RESULTS_DIR -- two configs can share the same
    # seed (baseline.yaml and ablation_no_kl_pressure.yaml both use seed: 0), and a flat
    # real_run_seed{seed}.pt would let the second run silently overwrite the first's artifact.
    config_results_dir = RESULTS_DIR / args.config.stem
    config_results_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = config_results_dir / f"real_run_seed{seed}.pt"
    # cap is saved in real (NPR-million) units, matching W_hat/W_baseline above -- K_raw was
    # divided by SCALE earlier in main(), so multiplying back here undoes that the same way
    # W_hats/W_baseline already do. Needed by eda_reviewer_questions.py's cap-binding-fraction
    # check (Sec 10a item 2) without that script having to re-derive KAPPA/SCALE itself.
    cap_real_units = (cfg.kappa * K_raw * SCALE).cpu()
    # r/c restored to real NPR-million units the same way cap_real_units is -- the empirical
    # target marginals W_hat/W_baseline were actually projected onto, needed by
    # eda_network_structure.py's marginal-preservation check (this script only ever uses r/c
    # in training-rescaled form in-memory, never persisted them before now).
    torch.save(
        {
            "bfi_names": bfi_names,
            "periods": period_labels,
            "W_hat": torch.stack(W_hats).cpu(),
            "W_baseline": torch.stack(W_baseline).cpu(),
            "q": torch.stack(qs).cpu(),
            "cap": cap_real_units,
            "r": (r * SCALE).cpu(),
            "c": (c * SCALE).cpu(),
            "losses": losses,
            "components_trend": components_trend,
            "non_finite_periods": bad_periods,
            "degenerate_periods": degenerate_periods,
            "model_stability": model_stability,
            "baseline_stability": baseline_stability,
            "baseline_diff": baseline_diff,
            "seed": seed,
        },
        artifact_path,
    )
    print(f"\nsaved: {artifact_path}")

    summary_path = config_results_dir / f"real_run_summary_seed{seed}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "csv": str(args.csv),
                "config": str(args.config),
                "seed": seed,
                "n_banks": n,
                "n_periods": t,
                "epochs": cfg.training.epochs,
                "kl_beta_max": cfg.training.kl_beta_max,
                "kl_warmup_epochs": cfg.training.kl_warmup_epochs,
                "kappa": cfg.kappa,
                "non_finite_periods": len(bad_periods),
                "degenerate_periods": len(degenerate_periods),
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
