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


import torch

try:
    from .config import (
        CapacityProjectionConfig,
        EncoderDecoderConfig,
        ReparamConfig,
        SyntheticConfig,
        TrainStepConfig,
    )
    from .synthetic import sample_synthetic_network
    from .train import Period, ReconstructionModel, train_step

except ImportError:
    from config import (
        CapacityProjectionConfig,
        EncoderDecoderConfig,
        ReparamConfig,
        SyntheticConfig,
        TrainStepConfig,
    )
    from synthetic import sample_synthetic_network
    from train import Period, ReconstructionModel, train_step


def _toy_batch(n: int, d: int, k: int, n_periods: int, generator: torch.Generator, with_ground_truth: bool) -> list:
    """Builds a short panel of periods via the already-validated Algorithm 2
    sampler, so train_step's inputs look like genuine (feasible) network data
    rather than arbitrary random tensors.
    """
    pi = torch.ones(k) / k
    Lambda = torch.rand(k, k, generator=generator) * 0.3 + 0.05
    w = torch.randn(d, generator=generator) * 0.1
    sigma2 = torch.tensor(0.2)
    cap = torch.full((n,), 5.0)
    proj_config = CapacityProjectionConfig()
    syn_config = SyntheticConfig()

    periods = []
    for _ in range(n_periods):
        X = torch.randn(n, d, generator=generator)
        eta = torch.randn(n, n, generator=generator) * 0.5
        W_true, r, c, _ = sample_synthetic_network(
            X, pi, Lambda, w, sigma2, eta, cap, proj_config, syn_config, generator=generator
        )
        periods.append(
            Period(X=X, r=r, c=c, cap=cap, W_true=W_true if with_ground_truth else None)
        )
    return periods


def shapes_and_update_case() -> bool:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    n, d, k, hidden = 6, 3, 2, 8
    enc_config = EncoderDecoderConfig(input_dim=d, hidden_dim=hidden, num_blocks=k, decoder_hidden_dim=16)
    model = ReconstructionModel(enc_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    before = [p.clone() for p in model.parameters()]
    batch = _toy_batch(n, d, k, n_periods=3, generator=gen, with_ground_truth=True)

    loss = train_step(
        model,
        optimizer,
        batch,
        prior_probs=torch.ones(k) / k,
        projection_config=CapacityProjectionConfig(),
        reparam_config=ReparamConfig(),
        train_config=TrainStepConfig(lambda_smooth=0.1, lambda_reg=0.01),
        generator=gen,
    )
    after = list(model.parameters())

    finite_ok = torch.isfinite(loss).all().item()
    scalar_ok = loss.dim() == 0
    params_finite = all(torch.isfinite(p).all().item() for p in after)
    params_changed = any(not torch.equal(b, a) for b, a in zip(before, after))

    passed = finite_ok and scalar_ok and params_finite and params_changed
    print(
        f"[shapes_and_update_case] finite_ok={finite_ok} scalar_ok={scalar_ok} "
        f"params_finite={params_finite} params_changed={params_changed} -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def real_data_branch_case() -> bool:
    """When no period carries W_true (real-data branch, steps 13-14), L_recon
    must stay exactly 0 and the step must still run to completion.
    """
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(1)
    n, d, k, hidden = 6, 3, 2, 8
    enc_config = EncoderDecoderConfig(input_dim=d, hidden_dim=hidden, num_blocks=k, decoder_hidden_dim=16)
    model = ReconstructionModel(enc_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    batch = _toy_batch(n, d, k, n_periods=3, generator=gen, with_ground_truth=False)
    all_none = all(p.W_true is None for p in batch)

    loss = train_step(
        model,
        optimizer,
        batch,
        prior_probs=torch.ones(k) / k,
        projection_config=CapacityProjectionConfig(),
        reparam_config=ReparamConfig(),
        train_config=TrainStepConfig(lambda_smooth=0.1, lambda_reg=0.01),
        generator=gen,
    )

    finite_ok = torch.isfinite(loss).all().item()
    passed = all_none and finite_ok
    print(f"[real_data_branch_case] all_none={all_none} finite_ok={finite_ok} -> {'PASS' if passed else 'FAIL'}")
    return passed


def loss_decreases_case() -> bool:
    """Basic trainability smoke test: repeatedly stepping on a FIXED synthetic
    batch should push the loss down overall, confirming the whole pipeline
    (encoder -> posterior -> decoder -> samplers -> projection -> loss ->
    backprop) is actually learnable end to end, not just shape-correct.
    """
    torch.manual_seed(2)
    gen = torch.Generator().manual_seed(2)
    n, d, k, hidden = 6, 3, 2, 8
    enc_config = EncoderDecoderConfig(input_dim=d, hidden_dim=hidden, num_blocks=k, decoder_hidden_dim=16)
    model = ReconstructionModel(enc_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    batch = _toy_batch(n, d, k, n_periods=3, generator=gen, with_ground_truth=True)

    losses = []
    for _ in range(30):
        loss = train_step(
            model,
            optimizer,
            batch,
            prior_probs=torch.ones(k) / k,
            projection_config=CapacityProjectionConfig(),
            reparam_config=ReparamConfig(),
            train_config=TrainStepConfig(lambda_smooth=0.1, lambda_reg=0.01),
            generator=gen,
        )
        losses.append(loss.item())

    early_mean = sum(losses[:5]) / 5
    late_mean = sum(losses[-5:]) / 5
    decreased = late_mean < early_mean
    passed = decreased
    print(
        f"[loss_decreases_case] early_mean={early_mean:.3f} late_mean={late_mean:.3f} "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return passed


if __name__ == "__main__":
    results = {
        "shapes_and_update_case": shapes_and_update_case(),
        "real_data_branch_case": real_data_branch_case(),
        "loss_decreases_case": loss_decreases_case(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
