#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Provide the reparameterized/relaxed samplers used by Algorithm 3 (Sec 4).

Three differentiable samplers that let gradients flow through otherwise-discrete or
stochastic steps: a Gumbel-softmax draw for the categorical block assignment z
(Eq. 7), a Gumbel-sigmoid (binary-concrete) draw for the existence gate e_ij
(Eq. 4), and a reparameterized log-normal draw for the edge weight (Eq. 3). Each
accepts an optional torch.Generator so sampling stays reproducible, threaded through
the same way the rest of the package does.
"""

import torch
import torch.nn.functional as F

try:
    from .config import ReparamConfig
except ImportError:
    from config import ReparamConfig


def _rand_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.rand_like(x)
    return torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def _randn_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def gumbel_softmax_sample(
    probs: torch.Tensor,
    config: ReparamConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a differentiable relaxed one-hot from Categorical(q_i) (Algorithm 3 step 5).

    Relaxes Categorical(q_i) into a differentiable sample (Jang et al., 2017; Maddison
    et al., 2017). Takes probabilities (q_i, already softmaxed per Eq. 7's step 4) --
    internally moves to log-space before adding Gumbel noise, since that's where the
    Gumbel-max trick is actually valid.

    Formula (per row, elementwise u ~ Uniform(0, 1)):
        g = -log(-log(u))                       Gumbel(0, 1) noise
        z = softmax( (log(q_i) + g) / tau )
    where tau = config.gumbel_softmax_tau and config.delta stabilizes each log(.).

    Args:
        probs: (..., num_blocks) probabilities, each row sums to 1.
        config: ReparamConfig (gumbel_softmax_tau, delta).
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Same shape as probs; a relaxed one-hot-like distribution, each row still
        sums to ~1 (it's a softmax output), differentiable w.r.t. probs.
    """
    u = _rand_like(probs, generator)
    gumbel_noise = -torch.log(-torch.log(u + config.delta) + config.delta)
    logits = torch.log(probs + config.delta) + gumbel_noise
    return F.softmax(logits / config.gumbel_softmax_tau, dim=-1)


def gumbel_sigmoid_sample(
    eta: torch.Tensor,
    config: ReparamConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a differentiable relaxed Bernoulli existence gate (Algorithm 3 step 8).

    The binary-concrete relaxation of Bernoulli(sigmoid(eta_ij)) (Eq. 4's existence
    gate). Takes a raw logit directly (eta, the decoder's output) -- unlike the
    softmax case, no log-transform is needed since eta already IS a logit.

    Formula (elementwise u ~ Uniform(0, 1)):
        L = log(u) - log(1 - u)                 Logistic(0, 1) noise
        e = sigmoid( (eta + L) / tau )
    where tau = config.gumbel_sigmoid_tau and config.delta stabilizes each log(.).

    Args:
        eta: (...,) zero-inflation logits.
        config: ReparamConfig (gumbel_sigmoid_tau, delta).
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Same shape as eta, values in (0, 1), differentiable w.r.t. eta.
    """
    u = _rand_like(eta, generator)
    logistic_noise = torch.log(u + config.delta) - torch.log(1 - u + config.delta)
    return torch.sigmoid((eta + logistic_noise) / config.gumbel_sigmoid_tau)


def lognormal_rsample(
    loc: torch.Tensor,
    scale: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a reparameterized log-normal edge weight (Algorithm 3 step 9).

    Implements W_tilde_ij,t <- e_ij,t * LogNormalSample(log lambda_ij,t, sigma^2),
    the "reparameterized weight" sample from Sec 4's last bullet. Reparameterized
    directly (X = exp(loc + scale * eps), eps ~ N(0, 1)) rather than via
    torch.distributions.LogNormal.rsample(), since the latter doesn't accept a
    generator and this package threads one through everywhere else for
    reproducibility.

    Args:
        loc: (...,) = log(lambda), the underlying Normal's mean.
        scale: (...,) or scalar = sqrt(sigma^2), the underlying Normal's std.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Same shape as loc, strictly positive, differentiable w.r.t. loc and scale.
    """
    eps = _randn_like(loc, generator)
    return torch.exp(loc + scale * eps)
