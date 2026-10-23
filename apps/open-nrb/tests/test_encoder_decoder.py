#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "numpy"
# ]
# ///


from pathlib import Path
import sys

# Ensure the 'src' directory is in sys.path so 'open_nrb' can be imported
# whether running from project root or inside the tests/ folder.
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import torch.nn.functional as F


from open_nrb.models.config import EncoderDecoderConfig
from open_nrb.models.encoder_decoder import BlockPosteriorMLP, PairwiseDecoder, TemporalEncoder


def temporal_encoder_case() -> bool:
    torch.manual_seed(0)
    n, d, hidden = 10, 3, 8
    config = EncoderDecoderConfig(input_dim=d, hidden_dim=hidden, num_blocks=2, decoder_hidden_dim=16)
    encoder = TemporalEncoder(config)

    X_t = torch.randn(n, d)
    r_t, c_t = torch.randn(n), torch.randn(n)
    h_prev = encoder.init_hidden(n)
    h_t = encoder(X_t, r_t, c_t, h_prev)

    shape_ok = h_t.shape == (n, hidden)
    finite_ok = torch.isfinite(h_t).all().item()
    passed = shape_ok and finite_ok
    print(f"[temporal_encoder_case] shape_ok={shape_ok} finite_ok={finite_ok} -> {'PASS' if passed else 'FAIL'}")
    return passed


def block_posterior_case() -> bool:
    torch.manual_seed(1)
    n, hidden, k = 10, 8, 3
    config = EncoderDecoderConfig(input_dim=4, hidden_dim=hidden, num_blocks=k, decoder_hidden_dim=16)
    posterior = BlockPosteriorMLP(config)

    h = torch.randn(n, hidden)
    logits = posterior(h)
    probs = F.softmax(logits, dim=-1)

    shape_ok = logits.shape == (n, k)
    sums_to_one = torch.allclose(probs.sum(dim=-1), torch.ones(n), atol=1e-5)
    passed = shape_ok and sums_to_one
    print(f"[block_posterior_case] shape_ok={shape_ok} sums_to_one={sums_to_one} -> {'PASS' if passed else 'FAIL'}")
    return passed


def pairwise_decoder_case() -> bool:
    torch.manual_seed(2)
    n, hidden, k = 10, 8, 3
    config = EncoderDecoderConfig(input_dim=4, hidden_dim=hidden, num_blocks=k, decoder_hidden_dim=16)
    decoder = PairwiseDecoder(config)

    h = torch.randn(n, hidden)
    z_soft = F.softmax(torch.randn(n, k), dim=-1)
    eta, lam = decoder(h, z_soft)

    shape_ok = eta.shape == (n, n) and lam.shape == (n, n)
    finite_ok = torch.isfinite(eta).all().item() and torch.isfinite(lam).all().item()
    nonneg_ok = (lam >= 0).all().item()
    passed = shape_ok and finite_ok and nonneg_ok
    print(
        f"[pairwise_decoder_case] shape_ok={shape_ok} finite_ok={finite_ok} "
        f"nonneg_ok={nonneg_ok} -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def gradient_flow_case() -> bool:
    """End-to-end chain: encoder -> block posterior -> decoder, backprop a scalar
    loss, confirm every module's parameters get finite gradients. This has to hold
    before Algorithm 3's train_step can be built on top of these three modules.
    """
    torch.manual_seed(3)
    n, d, hidden, k = 10, 4, 8, 3
    config = EncoderDecoderConfig(input_dim=d, hidden_dim=hidden, num_blocks=k, decoder_hidden_dim=16)
    encoder = TemporalEncoder(config)
    posterior = BlockPosteriorMLP(config)
    decoder = PairwiseDecoder(config)

    X_t = torch.randn(n, d)
    r_t, c_t = torch.randn(n), torch.randn(n)
    h_prev = encoder.init_hidden(n)
    h_t = encoder(X_t, r_t, c_t, h_prev)
    logits = posterior(h_t)
    z_soft = F.softmax(logits, dim=-1)
    eta, lam = decoder(h_t, z_soft)

    loss = eta.sum() + lam.sum()
    loss.backward()

    modules = {"encoder": encoder, "posterior": posterior, "decoder": decoder}
    all_ok = True
    for name, module in modules.items():
        grads = [p.grad for p in module.parameters()]
        has_grad = all(g is not None for g in grads)
        finite = has_grad and all(torch.isfinite(g).all().item() for g in grads)
        ok = has_grad and finite
        all_ok = all_ok and ok
        print(f"[gradient_flow_case] {name}: has_grad={has_grad} finite={finite} -> {'PASS' if ok else 'FAIL'}")

    print(f"[gradient_flow_case] overall -> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    results = {
        "temporal_encoder_case": temporal_encoder_case(),
        "block_posterior_case": block_posterior_case(),
        "pairwise_decoder_case": pairwise_decoder_case(),
        "gradient_flow_case": gradient_flow_case(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")