#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///

import sys
from pathlib import Path

import torch

try:
    from open_nrb.models.config import SinkhornConfig
    from open_nrb.models.sinkhorn import sinkhorn_projection
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from config import SinkhornConfig
    from sinkhorn import sinkhorn_projection


def feasible_case() -> bool:
    torch.manual_seed(0)
    n = 8
    W_tilde = torch.rand(n, n) + 0.1
    r = W_tilde.sum(dim=1)
    c = W_tilde.sum(dim=0)

    W = sinkhorn_projection(W_tilde, r, c, SinkhornConfig())

    row_err = (W.sum(dim=1) - r).abs().max().item()
    col_err = (W.sum(dim=0) - c).abs().max().item()
    passed = row_err < 1e-3 and col_err < 1e-3
    print(f"[feasible_case] row_err={row_err:.2e} col_err={col_err:.2e} -> {'PASS' if passed else 'FAIL'}")
    return passed


def gradient_flow_case() -> bool:
    torch.manual_seed(2)
    n = 6
    W_tilde = (torch.rand(n, n) + 0.1).requires_grad_(True)
    r = W_tilde.detach().sum(dim=1)
    c = W_tilde.detach().sum(dim=0)

    W = sinkhorn_projection(W_tilde, r, c, SinkhornConfig())
    W.sum().backward()

    grad = W_tilde.grad
    has_grad = grad is not None
    finite = has_grad and torch.isfinite(grad).all().item()
    passed = bool(has_grad and finite)
    print(f"[gradient_flow_case] grad_exists={has_grad} grad_finite={finite} -> {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = {
        "feasible_case": feasible_case(),
        "gradient_flow_case": gradient_flow_case(),
    }
    print("\n=== SUMMARY (uncapped Sinkhorn/IPF -- Gandy-Veraart/RAS baseline only;")
    print("    the capped constraint layer is tested separately in")
    print("    test_capacity_projection.py) ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
