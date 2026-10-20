#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///

import torch

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
    posterior + pairwise decoder), Lambda (block-pair intensities, Eq. 3), w
    (fitness weights, Eq. 2), and sigma^2 (log-normal dispersion, Sec 10 -- a
    single global scalar here, not per-block, since that choice isn't settled
    yet and global is the simpler default). model.parameters() goes straight
    into the caller's optimizer -- train_step never touches an optimizer's
    internals directly.
    """

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.encoder = TemporalEncoder(config)
        self.posterior = BlockPosteriorMLP(config)
        self.decoder = PairwiseDecoder(config)
        self.Lambda = nn.Parameter(torch.rand(config.num_blocks, config.num_blocks) * 0.5 + 0.1)
        self.w = nn.Parameter(torch.zeros(config.input_dim))
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


def _categorical_kl(q: torch.Tensor, prior_probs: torch.Tensor, delta: float) -> torch.Tensor:
    """KL(q_psi(z_i) || p(z)) per node, summed over nodes. q: (N, K); prior_probs:
    (K,), broadcast across nodes -- same prior for every node (mean-field, i.i.d.
    across i, matching Eq. 1's generative assumption).
    """
    log_q = torch.log(q + delta)
    log_p = torch.log(prior_probs + delta)
    return (q * (log_q - log_p)).sum(dim=-1).sum()


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
            lambda_smooth/lambda_reg/recon_zero_tol/delta respectively.
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
    smooth_terms = []
    W_prev = None

    for period in batch:
        # Steps 2-3: temporal encoder (no GNN refinement -- optional, Sec 8c, not built)
        h = model.encoder(period.X, h)

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
        total_kl = total_kl + _categorical_kl(q, prior_probs, train_config.delta)

        if W_prev is not None:
            smooth_terms.append((W_t - W_prev).pow(2).sum())
        W_prev = W_t

    # Steps 16-18: smoothness, weight regularization, total loss
    L_smooth = torch.stack(smooth_terms).sum() if smooth_terms else torch.zeros((), device=device, dtype=dtype)
    L_reg = model.Lambda.pow(2).sum() + model.w.pow(2).sum()
    loss = total_recon + total_kl + train_config.lambda_smooth * L_smooth + train_config.lambda_reg * L_reg

    # Step 19: backprop and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
