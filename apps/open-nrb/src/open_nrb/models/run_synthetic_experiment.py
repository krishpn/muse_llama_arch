#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
# ]
# ///
"""Synthetic validation experiment (design doc Sec 6/12): train the full
reconstruction model (Algorithm 3) against a known-ground-truth SBM panel
(Algorithm 2) and compare its recovered W_t to the classical uncapped
Sinkhorn/RAS baseline -- Sec 7's mandatory comparison, reported honestly
either way, not just when the model wins.

Usage: uv run run_synthetic_experiment.py
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .capacity_projection import capacitated_projection
    from .config import (
        CapacityProjectionConfig,
        EncoderDecoderConfig,
        ReparamConfig,
        SinkhornConfig,
        SyntheticConfig,
        TrainStepConfig,
    )
    from .experiment_config import SyntheticExperimentConfig, SyntheticScenarioConfig, load_synthetic_experiment_config
    from .sinkhorn import sinkhorn_projection
    from .synthetic import sample_synthetic_network
    from .train import Period, ReconstructionModel, _categorical_kl, _zero_inflated_lognormal_nll, train_step
except ImportError:
    from capacity_projection import capacitated_projection
    from config import (
        CapacityProjectionConfig,
        EncoderDecoderConfig,
        ReparamConfig,
        SinkhornConfig,
        SyntheticConfig,
        TrainStepConfig,
    )
    from experiment_config import SyntheticExperimentConfig, SyntheticScenarioConfig, load_synthetic_experiment_config
    from sinkhorn import sinkhorn_projection
    from synthetic import sample_synthetic_network
    from train import Period, ReconstructionModel, _categorical_kl, _zero_inflated_lognormal_nll, train_step


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# results/ and experiments/ are both siblings of this script's parent dir (models/) under
# src/open_nrb/ -- data/evaluation/models/results/tests/utils/experiments all live at the same
# depth, same __file__-relative convention as run_real_experiment.py's RESULTS_DIR/
# DEFAULT_CONFIG (and evaluation/make_plots.py's RESULTS_DIR). Safe under a real install too:
# both live INSIDE the installable package (src/open_nrb/), so this still resolves correctly
# from site-packages/open_nrb/... as a module, not just as a standalone script.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "synthetic_baseline.yaml"


def _kl_beta_schedule(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """Linear KL annealing (beta-VAE warm-up), ramping 0 -> beta_max rather than
    0 -> 1. Motivated by two successive findings (Sec 10): (1) under the
    original unweighted (beta=1 always) KL, kl collapsed to ~0 by epoch 100 and
    block_acc fell BELOW the trivial majority-guess baseline -- training was
    actively destroying whatever block signal existed at initialization. (2)
    after fixing the encoder to also see r_t, c_t (this made real block
    recovery possible -- block_acc peaked at 0.85 while beta_t was still small),
    annealing all the way to beta_t=1 reproduced the same collapse on a delay:
    block_acc decayed back toward/below the trivial baseline in lockstep with
    beta_t's climb, well before warmup even finished. So the problem isn't ramp
    SPEED, it's the ceiling -- beta_t=1 always eventually out-competes whatever
    reconstruction-gradient signal keeps q_psi(z) informative, regardless of how
    slowly it gets there. Capping beta_t's asymptote below 1 keeps the KL term
    as a regularizer instead of letting it fully win.
    """
    if warmup_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * epoch / warmup_epochs)


def _trivial_majority_baseline(zs: list[torch.Tensor], k: int) -> float:
    """The accuracy of just always guessing the single most common block label
    -- the real floor block_acc must clear to mean anything. Best-of-permutation
    accuracy against an IMBALANCED pi (0.3/0.7 in this scenario, not 0.5/0.5) is
    >= max(pi) for free, not 1/K: always predicting the majority label alone
    already scores max(pi) under the permutation-matched metric.
    """
    all_z = torch.cat(zs)
    counts = torch.bincount(all_z, minlength=k).float()
    return (counts.max() / counts.sum()).item()


def _build_panel(
    generator: torch.Generator, device: torch.device, scenario: SyntheticScenarioConfig
) -> tuple[list[Period], list[torch.Tensor]]:
    """A fixed core/periphery SBM scenario: Lambda's core-core cell (2.5) set well
    above core-periphery (0.6) and periphery-periphery (0.2) -- the textbook
    core/periphery pattern this method is meant to recover. eta_block mirrors that same
    block structure for the EXISTENCE gate (core-core almost certainly trades,
    periphery-periphery almost never does, core-periphery in between) -- an earlier
    version of this scenario fed sample_synthetic_network an (N, N) eta drawn as pure
    i.i.d. noise unrelated to z, which made the existence channel of W_true completely
    unrecoverable from any input the model observes and left only the log-normal
    magnitude term carrying a learnable block-dependent signal. z is resampled
    independently each period: Algorithm 2 (Sec 11) is a one-shot sampler with no
    time index (Eq. 1 has no t subscript on z_i either way), and giving it a
    truly time-invariant z would need a bigger change to sample_synthetic_network's
    API than this experiment script should make -- flagged here, not silently
    assumed away.

    Returns (periods, zs): zs is the true per-period block assignment (N,), kept
    separate from Period rather than added as a field on it, since z is a synthetic-
    validation-only quantity (train_step's real Period/training path never has it) --
    needed to check whether q_psi(z) (Eq. 7) actually recovers block structure, not
    just whether W_hat matches W_true, since those can diverge (Sec 10's open item on
    KL collapse).
    """
    pi = torch.tensor(scenario.pi, device=device)
    Lambda = torch.tensor(scenario.lambda_matrix, device=device)
    eta_block = torch.tensor(scenario.eta_block, device=device)
    w = torch.tensor(scenario.w, device=device)
    sigma2 = torch.tensor(scenario.sigma2, device=device)
    cap = torch.full((scenario.n_banks,), scenario.cap, device=device)
    proj_config = CapacityProjectionConfig()
    syn_config = SyntheticConfig()

    periods = []
    zs = []
    for _ in range(scenario.n_periods):
        X = torch.randn(scenario.n_banks, scenario.covariate_dim, generator=generator, device=device) * 0.5
        W_true, r, c, z = sample_synthetic_network(
            X, pi, Lambda, w, sigma2, eta_block, cap, proj_config, syn_config, generator=generator
        )
        periods.append(Period(X=X, r=r, c=c, cap=cap, W_true=W_true))
        zs.append(z)
    return periods, zs


def _forward_pass_components(
    model: ReconstructionModel,
    batch: list[Period],
    prior_probs: torch.Tensor,
    projection_config: CapacityProjectionConfig,
    train_config: TrainStepConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    """Mirrors train_step's full forward pass (steps 2-18) under no_grad, but uses the
    posterior-predictive MEAN at every stochastic step -- soft block assignment q itself
    (no Gumbel-softmax draw), existence probability sigmoid(eta) (no Gumbel-sigmoid draw),
    and the log-normal mean lam * exp(sigma2/2) (no reparameterized draw) -- instead of a
    single noisy sample. A single stochastic draw carries irreducible sampling variance
    even once (eta, lambda) are well-calibrated to W_true's likelihood, which is why
    model_rel_error was flat while recon (the likelihood, sample-independent) improved
    steadily: this point estimate minimizes expected squared error under the model rather
    than reporting one noisy realization of it. Returns the per-period W_t estimates, the
    per-period soft block posterior q (so callers can check block recovery against the
    synthetic ground-truth z independently of W_t accuracy -- Sec 10's open item on whether
    low reconstruction error implies real block recovery or just a good fit via h alone),
    AND the four loss terms (recon, kl, lambda_smooth*smooth, lambda_reg*reg) SEPARATELY --
    train_step only returns their weighted sum (Eq. 9) by design. Reuses train_step's own
    private _categorical_kl/_zero_inflated_lognormal_nll helpers rather than re-deriving
    the formulas here, so this can't silently drift from what's actually trained.
    """
    device, dtype = batch[0].X.device, batch[0].X.dtype
    n = batch[0].X.shape[0]
    h = model.init_hidden(n, device=device, dtype=dtype)
    mask = 1.0 - torch.eye(n, device=device, dtype=dtype)

    total_recon = torch.zeros((), device=device, dtype=dtype)
    total_kl = torch.zeros((), device=device, dtype=dtype)
    smooth_terms = []
    W_prev = None
    W_hats = []
    qs = []

    with torch.no_grad():
        for period in batch:
            h = model.encoder(period.X, period.r, period.c, h)
            q = F.softmax(model.posterior(h), dim=-1)
            qs.append(q)
            eta, lam = model.decoder(h, q)
            p_exist = torch.sigmoid(eta)
            w_mean = lam * torch.exp(model.sigma2 / 2)
            W_tilde = p_exist * w_mean * mask
            W_t = capacitated_projection(W_tilde, period.r, period.c, period.cap, projection_config)
            W_hats.append(W_t)

            if period.W_true is not None:
                total_recon = total_recon + _zero_inflated_lognormal_nll(
                    period.W_true, eta, lam, model.sigma2, mask, train_config
                )
            total_kl = total_kl + _categorical_kl(q, prior_probs, train_config.delta)

            if W_prev is not None:
                smooth_terms.append((W_t - W_prev).pow(2).sum())
            W_prev = W_t

    L_smooth = torch.stack(smooth_terms).sum() if smooth_terms else torch.zeros((), device=device, dtype=dtype)
    L_reg = model.Lambda.pow(2).sum() + model.w.pow(2).sum()

    components = {
        "recon": total_recon.item(),
        "kl": total_kl.item(),
        "smooth_weighted": (train_config.lambda_smooth * L_smooth).item(),
        "reg_weighted": (train_config.lambda_reg * L_reg).item(),
    }
    return W_hats, qs, components


def _block_recovery_accuracy(qs: list[torch.Tensor], zs: list[torch.Tensor], k: int) -> float:
    """argmax(q) vs the true z, best accuracy over all K! label permutations. Block labels
    are unidentifiable (Eq. 1's z_i ~ Categorical(pi) has no canonical ordering -- nothing
    ties label "0" to "core" rather than "periphery"), so comparing raw label values
    directly would wrongly score a perfectly-recovered partition near 0% if it just picked
    the other labeling. K=2 here (2 permutations), written generally in case K changes.
    """
    preds = torch.cat([q.argmax(dim=-1) for q in qs])
    trues = torch.cat(zs)
    best = 0.0
    for perm in itertools.permutations(range(k)):
        perm_t = torch.tensor(perm, device=preds.device)
        best = max(best, (perm_t[preds] == trues).float().mean().item())
    return best


def _relative_frobenius_errors(W_hats: list[torch.Tensor], periods: list[Period]) -> list[float]:
    errors = []
    for W_hat, period in zip(W_hats, periods):
        denom = period.W_true.norm().item()
        num = (W_hat - period.W_true).norm().item()
        errors.append(num / denom if denom > 0 else float("nan"))
    return errors


def _baseline_errors(periods: list[Period]) -> tuple[list[float], list[torch.Tensor]]:
    """Gandy-Veraart/RAS baseline (Sec 7's mandatory comparison): uncapped
    Sinkhorn/IPF balancing of a uniform off-diagonal prior onto the SAME observed
    marginals the model gets -- no covariates, no cap, matching what the classical
    method actually is. Returns the per-period matrices too (not just the error
    scalars) so eda_network_structure.py can run the same structural checks on the
    baseline that it runs on W_hat, instead of only having a single error number.
    """
    sinkhorn_config = SinkhornConfig()
    errors = []
    baseline_matrices = []
    for period in periods:
        n = period.X.shape[0]
        prior = torch.ones(n, n, device=period.X.device) - torch.eye(n, device=period.X.device)
        W_baseline = sinkhorn_projection(prior, period.r, period.c, sinkhorn_config)
        denom = period.W_true.norm().item()
        num = (W_baseline - period.W_true).norm().item()
        errors.append(num / denom if denom > 0 else float("nan"))
        baseline_matrices.append(W_baseline)
    return errors, baseline_matrices


def _mean_std(xs: list[float]) -> tuple[float, float]:
    """Population mean/std (ddof=0) -- fine for eyeballing a handful of seeds,
    not a formal CI.
    """
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, var**0.5


def _run_once(seed: int, cfg: SyntheticExperimentConfig, results_dir: Path) -> dict:
    torch.manual_seed(seed)  # governs nn.Module weight init (CPU global RNG, before .to(DEVICE))
    print(f"\n--- seed {seed} ---")
    print(f"device: {DEVICE}")
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    periods, zs = _build_panel(generator, DEVICE, cfg.scenario)

    enc_config = EncoderDecoderConfig(
        input_dim=cfg.scenario.covariate_dim,
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

    baseline_errors, baseline_matrices = _baseline_errors(periods)
    baseline_mean = sum(baseline_errors) / len(baseline_errors)
    print(f"RAS/Sinkhorn baseline mean rel. error (fixed, doesn't depend on training): {baseline_mean:.4f}\n")

    losses = []
    error_trend = []  # (epoch, model_mean_rel_error) -- is the gap to baseline closing, not just loss?
    block_acc_trend = []  # (epoch, block_acc) -- same cadence as error_trend, for later plotting
    for epoch in range(cfg.training.epochs):
        train_config.kl_beta = _kl_beta_schedule(epoch, cfg.training.kl_warmup_epochs, cfg.training.kl_beta_max)
        loss = train_step(
            model,
            optimizer,
            periods,
            prior_probs,
            projection_config,
            reparam_config,
            train_config,
            generator=generator,
        )
        losses.append(loss.item())
        if epoch % cfg.training.eval_every == 0 or epoch == cfg.training.epochs - 1:
            W_hats, qs, components = _forward_pass_components(
                model, periods, prior_probs, projection_config, train_config
            )
            model_errors = _relative_frobenius_errors(W_hats, periods)
            model_mean = sum(model_errors) / len(model_errors)
            block_acc = _block_recovery_accuracy(qs, zs, cfg.model.num_blocks)
            error_trend.append((epoch, model_mean))
            block_acc_trend.append((epoch, block_acc))
            print(
                f"epoch {epoch:5d}  loss={loss.item():10.3f}  "
                f"model_rel_error={model_mean:.4f}  (baseline={baseline_mean:.4f})  "
                f"block_acc={block_acc:.4f}  kl_beta={train_config.kl_beta:.3f}  "
                f"recon={components['recon']:10.3f}  kl={components['kl']:8.3f}  "
                f"smooth_w={components['smooth_weighted']:8.3f}  reg_w={components['reg_weighted']:7.4f}"
            )

    W_hats, qs, _ = _forward_pass_components(
        model, periods, prior_probs, projection_config, train_config
    )
    model_errors = _relative_frobenius_errors(W_hats, periods)
    model_mean = sum(model_errors) / len(model_errors)
    block_acc = _block_recovery_accuracy(qs, zs, cfg.model.num_blocks)
    trivial_baseline = _trivial_majority_baseline(zs, cfg.model.num_blocks)

    print(f"\n=== SYNTHETIC VALIDATION (Sec 6/12) -- seed {seed} ===")
    print(f"loss: first={losses[0]:.3f} last={losses[-1]:.3f}")
    print("error trend (epoch, model_rel_error): " + ", ".join(f"({e}, {v:.4f})" for e, v in error_trend))
    print(f"model mean relative Frobenius error:   {model_mean:.4f}")
    print(f"RAS/Sinkhorn baseline mean rel. error:  {baseline_mean:.4f}")
    print(
        f"block recovery accuracy (best label permutation): {block_acc:.4f}  "
        f"(trivial majority-guess baseline={trivial_baseline:.4f})"
    )
    print(
        "model beats classical baseline"
        if model_mean < baseline_mean
        else "model does NOT beat classical baseline -- report honestly (Sec 7)"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = results_dir / f"seed_{seed}.pt"
    torch.save(
        {
            "seed": seed,
            "zs": torch.stack(zs).cpu(),  # (N_PERIODS, N) -- ground truth, needed for any
                                          # later block-recovery plot/confusion matrix
            "W_true": torch.stack([p.W_true for p in periods]).cpu(),  # (N_PERIODS, N, N)
            "W_hat_final": torch.stack(W_hats).cpu(),  # (N_PERIODS, N, N), post-training
            "W_baseline": torch.stack(baseline_matrices).cpu(),  # (N_PERIODS, N, N), fixed
            "q_final": torch.stack(qs).cpu(),  # (N_PERIODS, N, K), post-training posterior
            "losses": losses,
            "error_trend": error_trend,
            "block_acc_trend": block_acc_trend,
            "baseline_mean": baseline_mean,
            "model_mean": model_mean,
            "block_acc": block_acc,
            "trivial_baseline": trivial_baseline,
        },
        artifact_path,
    )
    print(f"saved: {artifact_path}")

    return {
        "seed": seed,
        "baseline_mean": baseline_mean,
        "model_mean": model_mean,
        "block_acc": block_acc,
        "trivial_baseline": trivial_baseline,
    }


def main() -> None:
    """Loops _run_once over cfg.seeds (Sec 10 open item): a single seed's block_acc
    margin over the trivial baseline could be a real, repeatable effect of the
    r_t/c_t encoder fix, or just that one seed's draw of z/X/noise landing
    favorably. Report mean +/- std of the margin across seeds so that question
    has an actual answer before investing in more model capacity or a lower
    beta_max to try to widen it further.
    """
    parser = argparse.ArgumentParser()
    # Picks the experiment variant (baseline, an ablation, a sweep) -- the ground-truth SBM
    # scenario, training hyperparameters, and seed list all come from this file now, not
    # module constants.
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_synthetic_experiment_config(args.config)
    # Namespaced by the config file's stem (e.g. results/synthetic_baseline/) -- two configs
    # can share overlapping seeds, and a flat seed_{seed}.pt/multiseed_summary.json would let a
    # second variant's run silently overwrite the first's artifacts.
    config_results_dir = RESULTS_DIR / args.config.stem

    results = [_run_once(seed, cfg, config_results_dir) for seed in cfg.seeds]

    print("\n=== MULTI-SEED SUMMARY (Sec 10: is the block_acc margin real or single-seed noise?) ===")
    for r in results:
        gap = r["block_acc"] - r["trivial_baseline"]
        print(
            f"seed {r['seed']}: model_rel_error={r['model_mean']:.4f}  "
            f"block_acc={r['block_acc']:.4f}  trivial_baseline={r['trivial_baseline']:.4f}  gap={gap:+.4f}"
        )

    model_means = [r["model_mean"] for r in results]
    baseline_means = [r["baseline_mean"] for r in results]
    block_accs = [r["block_acc"] for r in results]
    trivial_baselines = [r["trivial_baseline"] for r in results]
    gaps = [r["block_acc"] - r["trivial_baseline"] for r in results]

    print()
    for name, xs in [
        ("model_rel_error", model_means),
        ("RAS/Sinkhorn baseline_rel_error", baseline_means),
        ("block_acc", block_accs),
        ("trivial_baseline", trivial_baselines),
        ("gap (block_acc - trivial_baseline)", gaps),
    ]:
        m, s = _mean_std(xs)
        print(f"{name}: mean={m:.4f}  std={s:.4f}")

    n_beats_baseline = sum(1 for r in results if r["model_mean"] < r["baseline_mean"])
    n_beats_trivial = sum(1 for g in gaps if g > 0)
    print(f"\nseeds where model beats classical baseline: {n_beats_baseline}/{len(results)}")
    print(f"seeds where block_acc beats trivial majority-guess: {n_beats_trivial}/{len(results)}")

    model_mean_agg, model_std_agg = _mean_std(model_means)
    baseline_mean_agg, baseline_std_agg = _mean_std(baseline_means)
    block_acc_mean_agg, block_acc_std_agg = _mean_std(block_accs)
    trivial_mean_agg, trivial_std_agg = _mean_std(trivial_baselines)
    gap_mean_agg, gap_std_agg = _mean_std(gaps)

    config_results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config_results_dir / "multiseed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": str(args.config),
                "seeds": cfg.seeds,
                "epochs": cfg.training.epochs,
                "kl_beta_max": cfg.training.kl_beta_max,
                "kl_warmup_epochs": cfg.training.kl_warmup_epochs,
                "per_seed": results,
                "aggregates": {
                    "model_rel_error": {"mean": model_mean_agg, "std": model_std_agg},
                    "baseline_rel_error": {"mean": baseline_mean_agg, "std": baseline_std_agg},
                    "block_acc": {"mean": block_acc_mean_agg, "std": block_acc_std_agg},
                    "trivial_baseline": {"mean": trivial_mean_agg, "std": trivial_std_agg},
                    "gap": {"mean": gap_mean_agg, "std": gap_std_agg},
                    "n_beats_baseline": n_beats_baseline,
                    "n_beats_trivial": n_beats_trivial,
                    "n_seeds": len(results),
                },
            },
            f,
            indent=2,
        )
    print(f"\nsaved: {summary_path}")
    print(f"saved per-seed artifacts (zs, W_true, W_hat, q, trajectories) under: {config_results_dir}")


if __name__ == "__main__":
    main()
