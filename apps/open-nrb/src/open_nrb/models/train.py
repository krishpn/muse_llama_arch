#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .capacity_projection import capacitated_projection
    from .config import CapacityProjectionConfig, EncoderDecoderConfig, ReparamConfig, TrainStepConfig
    from .encoder_decoder import BlockPosteriorMLP, PairwiseDecoder, TemporalEncoder
    from .reparam import gumbel_sigmoid_sample, gumbel_softmax_sample, lognormal_rsample
except ImportError:
    from capacity_projection import capacitated_projection
    from config import CapacityProjectionConfig, EncoderDecoderConfig, ReparamConfig, TrainStepConfig
    from encoder_decoder import BlockPosteriorMLP, PairwiseDecoder, TemporalEncoder
    from reparam import gumbel_sigmoid_sample, gumbel_softmax_sample, lognormal_rsample


class ReconstructionModel(nn.Module):
    """Bundles every learnable piece Algorithm 3 updates: theta (GRU), psi (block
    posterior + pairwise decoder), and sigma^2 (log-normal dispersion, Sec 10 -- a
    single global scalar here, not per-block, since that choice isn't settled
    yet and global is the simpler default). model.parameters() goes straight
    into the caller's optimizer -- train_step never touches an optimizer's
    internals directly.

    No Lambda (Eq. 3's block-pair intensities) or w (Eq. 2's fitness weights) here:
    Algorithm 3 (Sec 11) explicitly replaces the literal Eq. 3-4 computation with
    Decoder_psi(h_i, h_j, z_i, z_j), so the trained model never consumes them --
    they're generative-scenario parameters for Algorithm 2's synthetic sampler
    (SyntheticConfig.lambda_matrix/.w, caller-supplied), not something this class
    learns. Lambda/w fields and the matching lambda_reg term used to exist on this
    class/in train_step's loss with nothing feeding them a gradient signal besides
    the regularizer itself, so they just decayed to ~0 -- removed instead of wired
    in, since wiring them into the decoder would mean re-adding the literal Eq. 3-4
    formula Algorithm 3 deliberately supersedes.
    """

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.encoder = TemporalEncoder(config)
        self.posterior = BlockPosteriorMLP(config)
        self.decoder = PairwiseDecoder(config)
        self._raw_sigma2 = nn.Parameter(torch.tensor(0.0))

    @property
    def sigma2(self) -> torch.Tensor:
        return F.softplus(self._raw_sigma2) + 1e-6

    def init_hidden(self, n: int, device=None, dtype=None) -> torch.Tensor:
        return self.encoder.init_hidden(n, device=device, dtype=dtype)


@dataclass
class Period:
    """One time period's worth of inputs to train_step -- a data container, not a
    tunable hyperparameter set, so it lives here rather than in config.py.
    """

    X: torch.Tensor  # (N, d) covariates
    r: torch.Tensor  # (N,) observed row marginal
    c: torch.Tensor  # (N,) observed column marginal
    cap: torch.Tensor  # (N,) = kappa * K_t, caller's job to build (Sec 11)
    W_true: torch.Tensor | None = None  # (N, N), synthetic branch only (Eq. 9)


def _categorical_kl(q: torch.Tensor, prior_probs: torch.Tensor, delta: float, free_bits: float = 0.0) -> torch.Tensor:
    """KL(q_psi(z_i) || p(z)) per node, MEANED over nodes (not summed). q: (N, K);
    prior_probs: (K,), broadcast across nodes -- same prior for every node (mean-field,
    i.i.d. across i, matching Eq. 1's generative assumption).

    free_bits clamps each node's own KL from below before the mean (Kingma et al.'s
    "free bits"): once a node's KL is at or below free_bits, it stops contributing
    gradient, so q is free to retain that much block-membership information without
    kl_beta pulling it all the way to the uniform prior. free_bits=0.0 reproduces the
    original unfloored term.

    Mean, not sum: train_step/_forward_pass_components also mean this over periods
    (see total_kl's final division by len(batch)), so the aggregate total_kl is the
    MEAN per-(node, period) KL, not a quantity that scales with N*T. Confirmed live:
    with the old sum-based version, the SAME free_bits=0.05 produced a floor of 19.2
    on a synthetic panel (N=24, T=16) vs. 106.4 on a real panel (N=16, T=133) -- a
    ~5.5x spread purely from panel shape, not a modeling choice, that silently
    changed kl_beta_max's effective relative weight against lambda_smooth/
    lambda_mi every time the complete-case filter produced a different N or T. Mean
    aggregation makes free_bits (and therefore kl_beta_max) comparable across
    experiments with different panel shapes -- it does NOT, by itself, change
    whether any individual node's KL is floored (that's still the same per-node
    free_bits threshold either way).
    """
    log_q = torch.log(q + delta)
    log_p = torch.log(prior_probs + delta)
    per_node_kl = (q * (log_q - log_p)).sum(dim=-1)
    return per_node_kl.clamp_min(free_bits).mean()


def _mutual_info_regularizer(q: torch.Tensor, delta: float) -> torch.Tensor:
    """-I(bank; block) for one period's q (N, K): mean per-bank entropy minus the
    entropy of the banks' mean block distribution. Minimizing this maximizes I,
    rewarding confident (low mean-entropy) assignments that still use more than one
    block across banks (high marginal entropy) -- IIC/IMSAT-style clustering loss,
    not derivable from kl_free_bits alone since that only floors KL, never rewards
    confidence (Sec 10).
    """
    log_q = torch.log(q + delta)
    cond_entropy = -(q * log_q).sum(dim=-1).mean()
    q_bar = q.mean(dim=0)
    marg_entropy = -(q_bar * torch.log(q_bar + delta)).sum()
    return cond_entropy - marg_entropy


def _zero_inflated_lognormal_nll(
    W_true: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    sigma2: torch.Tensor,
    mask: torch.Tensor,
    config: TrainStepConfig,
) -> torch.Tensor:
    """-log p(W_true | lambda, eta, sigma^2), Eq. 9's first term: a zero-inflated
    log-normal likelihood (Eq. 4's gate, Eq. 3's intensity). `mask` excludes
    i == j (no self-exposure, same convention as synthetic.py). Uses
    config.recon_zero_tol rather than exact equality to decide which entries
    count as "zero", since W_true is the output of capacitated_projection
    (Algorithm 1), whose row/column projection steps generally do not preserve
    the raw sample's exact sparsity pattern.
    """
    is_zero = W_true.abs() < config.recon_zero_tol
    log_p_zero = F.logsigmoid(-eta)

    safe_w = W_true.clamp_min(config.recon_zero_tol)
    loc = torch.log(lam + config.delta)
    scale = torch.sqrt(sigma2)
    log_normal_logpdf = torch.distributions.LogNormal(loc, scale).log_prob(safe_w)
    log_p_nonzero = F.logsigmoid(eta) + log_normal_logpdf

    log_p = torch.where(is_zero, log_p_zero, log_p_nonzero)
    return -(log_p * mask).sum()


def train_step(
    model: ReconstructionModel,
    optimizer: torch.optim.Optimizer,
    batch: list[Period],
    prior_probs: torch.Tensor,
    projection_config: CapacityProjectionConfig,
    reparam_config: ReparamConfig,
    train_config: TrainStepConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Algorithm 3 (paper1_network_reconstruction.md Sec 11, Eq. 6-9): one
    amortized-inference pass over `batch` (in temporal order) plus one optimizer
    step.

    N must be the same across every period in `batch` (same bank panel) since
    the GRU hidden state carries across periods and GRUCell requires a fixed
    batch size; K/d/hidden_dim are fixed at model-construction time via
    EncoderDecoderConfig, not here.

    Args:
        model: ReconstructionModel bundling theta, psi, Lambda, w, sigma^2.
        optimizer: Caller-constructed optimizer over model.parameters() -- this
            function doesn't choose the optimizer type or learning rate.
        batch: Periods in temporal order; GRU hidden state carries across them.
        prior_probs: (K,) p(z) for the KL term, e.g. torch.ones(K)/K for a
            uniform prior -- this function stays agnostic to whether the
            intended prior is uniform or the generative pi (Sec 10 open item).
        projection_config, reparam_config, train_config: hyperparameters for
            Algorithm 1, the reparameterized samplers, and
            lambda_smooth/recon_zero_tol/delta/kl_beta respectively.
            train_config.kl_beta (Eq. 9's beta_t) is read fresh on every call, so
            a caller implementing KL annealing (Sec 10) just mutates
            train_config.kl_beta between calls rather than passing a schedule in.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Scalar loss (Eq. 9), computed before backprop; by the time this
        returns, the optimizer step has already been applied (step 19 happens
        before step 20's return, same order as the pseudocode).
    """
    device = batch[0].X.device
    dtype = batch[0].X.dtype
    n = batch[0].X.shape[0]
    h = model.init_hidden(n, device=device, dtype=dtype)
    mask = 1.0 - torch.eye(n, device=device, dtype=dtype)

    total_recon = torch.zeros((), device=device, dtype=dtype)
    total_kl = torch.zeros((), device=device, dtype=dtype)
    total_mi_reg = torch.zeros((), device=device, dtype=dtype)
    smooth_terms = []
    W_prev = None

    for period in batch:
        # Steps 2-3: temporal encoder (no GNN refinement -- optional, Sec 8c, not built)
        h = model.encoder(period.X, period.r, period.c, h)

        # Steps 4-5: block-assignment posterior + reparameterized sample (Eq. 7)
        q = F.softmax(model.posterior(h), dim=-1)
        z = gumbel_softmax_sample(q, reparam_config, generator=generator)

        # Steps 6-7: pairwise decoder (Eq. 3-4)
        eta, lam = model.decoder(h, z)

        # Step 8: reparameterized existence gate
        e = gumbel_sigmoid_sample(eta, reparam_config, generator=generator)

        # Step 9: reparameterized log-normal weight, masked (no self-exposure)
        loc = torch.log(lam + train_config.delta)
        scale = torch.sqrt(model.sigma2)
        w_sample = lognormal_rsample(loc, scale, generator=generator)
        W_tilde = e * w_sample * mask

        # Step 10: project onto the feasible set (Algorithm 1), against the
        # period's OBSERVED marginals/cap -- not anything resampled from W_tilde
        W_t = capacitated_projection(W_tilde, period.r, period.c, period.cap, projection_config)

        # Steps 11-14: reconstruction loss, synthetic branch only
        if period.W_true is not None:
            total_recon = total_recon + _zero_inflated_lognormal_nll(
                period.W_true, eta, lam, model.sigma2, mask, train_config
            )

        # Step 15: KL to the block-assignment prior
        total_kl = total_kl + _categorical_kl(q, prior_probs, train_config.delta, train_config.kl_free_bits)

        # IIC/IMSAT-style clustering regularizer: rewards confident, block-diverse q
        # beyond what kl_free_bits alone incentivizes (Sec 10 follow-up)
        total_mi_reg = total_mi_reg + _mutual_info_regularizer(q, train_config.delta)

        if W_prev is not None:
            smooth_terms.append((W_t - W_prev).pow(2).sum())
        W_prev = W_t

    # Steps 16-18: smoothness, weight regularization, total loss (Eq. 9's beta_t
    # anneals the KL term's weight -- see TrainStepConfig.kl_beta; defaults to 1.0,
    # the original unweighted term, so callers that never set it are unaffected)
    # total_kl is also meaned over periods here (on top of _categorical_kl's per-period
    # mean over nodes), so it's the mean per-(node, period) KL -- invariant to batch
    # length T, matching _categorical_kl's panel-size-invariance rationale above.
    total_kl = total_kl / len(batch)
    L_smooth = torch.stack(smooth_terms).sum() if smooth_terms else torch.zeros((), device=device, dtype=dtype)
    loss = (
        total_recon
        + train_config.kl_beta * total_kl
        + train_config.lambda_mi * total_mi_reg
        + train_config.lambda_smooth * L_smooth
    )

    # Step 19: backprop and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
