#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///

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
    """Algorithm 3 step 5: z_i <- GumbelSoftmaxSample(q_i), relaxing
    Categorical(q_i) into a differentiable sample (Jang et al., 2017; Maddison et
    al., 2017). Takes probabilities (q_i, already softmaxed per Eq. 7's step 4) --
    internally moves to log-space before adding Gumbel noise, since that's where
    the Gumbel-max trick is actually valid.

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
    """Algorithm 3 step 8: e_ij,t <- GumbelSigmoidSample(eta_ij,t), the binary-
    concrete relaxation of Bernoulli(sigmoid(eta_ij)) (Eq. 4's existence gate).
    Takes a raw logit directly (eta, the decoder's output) -- unlike the softmax
    case, no log-transform is needed since eta already IS a logit.

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
    """Algorithm 3 step 9: W_tilde_ij,t <- e_ij,t * LogNormalSample(log
    lambda_ij,t, sigma^2) -- the "reparameterized weight" sample from Sec 4's last
    bullet. Reparameterized directly (X = exp(loc + scale * eps), eps ~ N(0,1))
    rather than via torch.distributions.LogNormal.rsample(), since the latter
    doesn't accept a generator and this package threads one through everywhere
    else for reproducibility.

    Args:
        loc: (...,) = log(lambda), the underlying Normal's mean.
        scale: (...,) or scalar = sqrt(sigma^2), the underlying Normal's std.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Same shape as loc, strictly positive, differentiable w.r.t. loc and scale.
    """
    eps = _randn_like(loc, generator)
    return torch.exp(loc + scale * eps)
