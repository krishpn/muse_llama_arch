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
from open_nrb.models.config import ReparamConfig
from open_nrb.models.reparam import gumbel_sigmoid_sample, gumbel_softmax_sample, lognormal_rsample



def gumbel_softmax_case() -> bool:
    torch.manual_seed(0)
    n, k = 20, 4
    logits = torch.randn(n, k, requires_grad=True)
    probs = F.softmax(logits, dim=-1)

    y = gumbel_softmax_sample(probs, ReparamConfig(gumbel_softmax_tau=0.5))
    shape_ok = y.shape == (n, k)
    sums_to_one = torch.allclose(y.sum(dim=-1), torch.ones(n), atol=1e-4)

    # Low temperature should push samples close to one-hot (near-argmax of probs)
    y_low_tau = gumbel_softmax_sample(probs.detach(), ReparamConfig(gumbel_softmax_tau=0.05))
    near_onehot = (y_low_tau.max(dim=-1).values.mean() > 0.9).item()

    passed = shape_ok and sums_to_one and near_onehot
    print(
        f"[gumbel_softmax_case] shape_ok={shape_ok} sums_to_one={sums_to_one} "
        f"near_onehot_at_low_tau={near_onehot} -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def gumbel_sigmoid_case() -> bool:
    torch.manual_seed(1)
    n = 50
    eta = torch.randn(n, n, requires_grad=True)

    y = gumbel_sigmoid_sample(eta, ReparamConfig(gumbel_sigmoid_tau=0.5))
    shape_ok = y.shape == (n, n)
    # Closed interval, not open: with n*n=2500 draws, some u land close enough to
    # 0/1 that sigmoid((eta+logistic_noise)/tau) saturates to exactly 0.0 or 1.0 in
    # float32 -- a harmless boundary artifact of the relaxation, not a bug (the
    # gradient through it stays finite, per gradient_flow_case).
    range_ok = bool(((y >= 0) & (y <= 1)).all())

    # Strongly positive eta should push samples toward 1, strongly negative toward 0
    y_pos = gumbel_sigmoid_sample(torch.full((n, n), 8.0), ReparamConfig(gumbel_sigmoid_tau=0.3))
    y_neg = gumbel_sigmoid_sample(torch.full((n, n), -8.0), ReparamConfig(gumbel_sigmoid_tau=0.3))
    direction_ok = (y_pos.mean() > 0.9).item() and (y_neg.mean() < 0.1).item()

    passed = shape_ok and range_ok and direction_ok
    print(
        f"[gumbel_sigmoid_case] shape_ok={shape_ok} range_ok={range_ok} "
        f"direction_ok={direction_ok} -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def lognormal_rsample_case() -> bool:
    torch.manual_seed(2)
    n = 20
    loc = torch.randn(n, n, requires_grad=True)
    scale = torch.tensor(0.3, requires_grad=True)

    x = lognormal_rsample(loc, scale)
    shape_ok = x.shape == (n, n)
    positive_ok = (x > 0).all().item()
    finite_ok = torch.isfinite(x).all().item()

    passed = shape_ok and positive_ok and finite_ok
    print(
        f"[lognormal_rsample_case] shape_ok={shape_ok} positive_ok={positive_ok} "
        f"finite_ok={finite_ok} -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def gradient_flow_case() -> bool:
    """Chains all three samplers the way Algorithm 3 does (steps 5, 8, 9) and
    backprops a scalar loss through the whole thing.
    """
    torch.manual_seed(3)
    n, k = 10, 3
    config = ReparamConfig()

    block_logits = torch.randn(n, k, requires_grad=True)
    probs = F.softmax(block_logits, dim=-1)
    z = gumbel_softmax_sample(probs, config)

    eta = torch.randn(n, n, requires_grad=True)
    e = gumbel_sigmoid_sample(eta, config)

    log_lambda = torch.randn(n, n, requires_grad=True)
    sigma = torch.tensor(0.3, requires_grad=True)
    w_sample = lognormal_rsample(log_lambda, sigma)

    W_tilde = e * w_sample
    loss = W_tilde.sum() + z.sum()
    loss.backward()

    grads = {
        "block_logits": block_logits.grad,
        "eta": eta.grad,
        "log_lambda": log_lambda.grad,
        "sigma": sigma.grad,
    }
    all_ok = True
    for name, g in grads.items():
        has_grad = g is not None
        finite = has_grad and torch.isfinite(g).all().item()
        ok = has_grad and finite
        all_ok = all_ok and ok
        print(f"[gradient_flow_case] {name}: has_grad={has_grad} finite={finite} -> {'PASS' if ok else 'FAIL'}")

    print(f"[gradient_flow_case] overall -> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    results = {
        "gumbel_softmax_case": gumbel_softmax_case(),
        "gumbel_sigmoid_case": gumbel_sigmoid_case(),
        "lognormal_rsample_case": lognormal_rsample_case(),
        "gradient_flow_case": gradient_flow_case(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")