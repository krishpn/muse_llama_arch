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
    from open_nrb.models.config import CapacityProjectionConfig
    from open_nrb.models.capacity_projection import capacitated_projection
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from .config import CapacityProjectionConfig
    from .capacity_projection import capacitated_projection


def feasible_case() -> bool:
    torch.manual_seed(0)
    n = 8
    W_tilde = torch.rand(n, n) + 0.1
    r = W_tilde.sum(dim=1)
    c = W_tilde.sum(dim=0)
    cap = r.max() * 10 * torch.ones(n)  # generous, never binds

    W = capacitated_projection(W_tilde, r, c, cap, CapacityProjectionConfig())

    row_err = (W.sum(dim=1) - r).abs().max().item()
    col_err = (W.sum(dim=0) - c).abs().max().item()
    passed = row_err < 1e-3 and col_err < 1e-3
    print(f"[feasible_case] row_err={row_err:.2e} col_err={col_err:.2e} -> {'PASS' if passed else 'FAIL'}")
    return passed


def binding_cap_case() -> bool:
    """Unlike test_sinkhorn.py's old (retired) binding_cap_case, this checks a true
    PER-ENTRY cap: cap is set below several individual raw entries in rows 0/1, but
    comfortably above r_i/n, so hitting the row marginal by spreading mass across
    multiple entries is still feasible -- the thing the old row-scalar Sinkhorn clip
    could never actually do.
    """
    torch.manual_seed(1)
    n = 8
    W_tilde = torch.rand(n, n) + 0.1
    r = W_tilde.sum(dim=1)
    c = W_tilde.sum(dim=0)

    cap = r.max() * 10 * torch.ones(n)
    cap[0] = (r[0] / n) * 1.2  # feasible (cap*n > r), but below several raw entries
    cap[1] = (r[1] / n) * 1.5

    W = capacitated_projection(W_tilde, r, c, cap, CapacityProjectionConfig())

    max_entry_violation = (W - cap.unsqueeze(1)).max().item()
    cap_ok = max_entry_violation < 1e-4

    row_err = (W.sum(dim=1) - r).abs().max().item()
    col_err = (W.sum(dim=0) - c).abs().max().item()
    marginals_ok = row_err < 1e-3 and col_err < 1e-3

    n_pinned_row0 = (W[0] >= cap[0] - 1e-4).sum().item()
    n_pinned_row1 = (W[1] >= cap[1] - 1e-4).sum().item()

    passed = cap_ok and marginals_ok
    print(
        f"[binding_cap_case] max_entry_violation={max_entry_violation:.2e} "
        f"row_err={row_err:.2e} col_err={col_err:.2e} -> {'PASS' if passed else 'FAIL'}"
    )
    print(
        f"[binding_cap_case] entries pinned at cap: row0={n_pinned_row0}/{n} "
        f"row1={n_pinned_row1}/{n} (>1 confirms a true per-entry cap, not a row-sum "
        "cap -- cf. the retired Sinkhorn-clip finding)"
    )
    return passed


def infeasible_case() -> bool:
    """Informational: cap set so tight (cap_i * N < r_i) that no feasible W exists for
    row 0. Not expected to hit that row's marginal exactly -- only checks the cap
    itself is still respected (Dykstra converges to a least-violation compromise, not
    a silent cap breach).
    """
    torch.manual_seed(3)
    n = 8
    W_tilde = torch.rand(n, n) + 0.1
    r = W_tilde.sum(dim=1)
    c = W_tilde.sum(dim=0)

    cap = r.max() * 10 * torch.ones(n)
    cap[0] = (r[0] / n) * 0.3  # cap[0] * n < r[0]: provably infeasible for row 0

    W = capacitated_projection(W_tilde, r, c, cap, CapacityProjectionConfig())

    row0_shortfall = (W.sum(dim=1) - r).abs()[0].item()
    cap_violation = (W[0] - cap[0]).max().item()
    print(
        f"[infeasible_case] row0 shortfall={row0_shortfall:.4f} (cap*n="
        f"{cap[0].item() * n:.4f} < target r={r[0].item():.4f}, nonzero expected)"
    )
    cap_held = cap_violation < 1e-4
    print(
        f"[infeasible_case] max per-entry cap violation on row0={cap_violation:.2e} "
        f"-> {'PASS (cap held)' if cap_held else 'FAIL (cap breached)'}"
    )
    return cap_held


def gradient_flow_case() -> bool:
    torch.manual_seed(2)
    n = 6
    W_tilde = (torch.rand(n, n) + 0.1).requires_grad_(True)
    r = W_tilde.detach().sum(dim=1)
    c = W_tilde.detach().sum(dim=0)
    cap = r.max() * 10 * torch.ones(n)

    W = capacitated_projection(W_tilde, r, c, cap, CapacityProjectionConfig())
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
        "binding_cap_case": binding_cap_case(),
        "infeasible_case": infeasible_case(),
        "gradient_flow_case": gradient_flow_case(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
