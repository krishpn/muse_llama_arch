#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Report ReconstructionModel's trainable parameter count against panel size.

Report the trainable parameter count of ReconstructionModel and compare it to
the panel's N x T, for the Section 7 over/under-parameterization check.

The count is FIXED by the architecture (input_dim d, hidden_dim h, num_blocks K,
decoder_hidden_dim) and does NOT depend on N or T -- the same network serves the
synthetic (N=24, T=16) and real (N=16, T=133) panels. This script just confirms
the number and prints it against the data-size reference quantities.

Run on the remote box (needs torch):
    uv run python -m open_nrb.evaluation.param_count
    uv run python -m open_nrb.evaluation.param_count --input-dim 5 --hidden 16 --blocks 2 --decoder-hidden 16 --n 16 --t 133
"""
from __future__ import annotations

import argparse

try:
    from ..models.config import EncoderDecoderConfig
    from ..models.train import ReconstructionModel
except ImportError:  # pragma: no cover - direct-script fallback
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.config import EncoderDecoderConfig
    from models.train import ReconstructionModel


def main() -> None:
    """Build the model from CLI args, count trainable parameters, and print size ratios.

    Compares the architecture-fixed trainable parameter total P against three
    data-size reference quantities for the Section 7 over/under-parameterization
    check:
        N * T          panel cells             (ratio P / (N * T))
        2 * N * T      marginal observations   (ratio P / (2 * N * T))
        N * (N - 1) * T    latent bilateral entries    (ratio P / (N * (N - 1) * T))
    """
    p = argparse.ArgumentParser(description=__doc__)
    # Baseline real config (experiments/configs/baseline.yaml) + COVARIATE_ORDER (len 5).
    p.add_argument("--input-dim", type=int, default=5, help="d = len(COVARIATE_ORDER)")
    p.add_argument("--hidden", type=int, default=16, help="model.hidden_dim")
    p.add_argument("--blocks", type=int, default=2, help="model.num_blocks (K)")
    p.add_argument("--decoder-hidden", type=int, default=16, help="model.decoder_hidden_dim")
    p.add_argument("--n", type=int, default=16, help="N banks (real panel)")
    p.add_argument("--t", type=int, default=133, help="T periods (real panel)")
    args = p.parse_args()

    cfg = EncoderDecoderConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden,
        num_blocks=args.blocks,
        decoder_hidden_dim=args.decoder_hidden,
    )
    model = ReconstructionModel(cfg)

    total = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(f"config: d={args.input_dim} h={args.hidden} K={args.blocks} "
          f"decoder_h={args.decoder_hidden}")
    print("per-submodule:")
    for name, module in [
        ("encoder (GRUCell)", model.encoder),
        ("posterior MLP", model.posterior),
        ("pairwise decoder", model.decoder),
    ]:
        sub = sum(q.numel() for q in module.parameters())
        print(f"  {name:22s} {sub:7d}")
    print(f"  {'_raw_sigma2 (scalar)':22s} {model._raw_sigma2.numel():7d}")
    print(f"TOTAL PARAMETERS: {total}")

    nt = args.n * args.t
    marg = 2 * args.n * args.t
    latent = args.n * (args.n - 1) * args.t
    print()
    print(f"N x T                         = {nt}   (params/NT = {total / nt:.3f})")
    print(f"marginal observations (2 N T) = {marg}   (params/2NT = {total / marg:.3f})")
    print(f"latent bilateral entries      = {latent}   (N(N-1)T; params/latent = {total / latent:.4f})")
    print()
    print("Parameter count is independent of N and T (set only by d, h, K, decoder_h).")


if __name__ == "__main__":
    main()
