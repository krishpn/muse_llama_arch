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


from open_nrb.models.config import CapacityProjectionConfig, SyntheticConfig
from open_nrb.models.synthetic import sample_synthetic_network



def _toy_inputs(n: int, k: int, d: int, eta_value: float, cap_multiplier: float):
    X = torch.rand(n, d) * 0.5
    pi = torch.ones(k) / k
    Lambda = torch.rand(k, k) * 0.5 + 0.1
    w = torch.zeros(d)  # phi_i = exp(0) = 1 for all i -- isolates Lambda/eta effects
    sigma2 = torch.tensor(0.1)
    eta = torch.full((n, n), eta_value)
    r_upper_bound = Lambda.max() * 1.0 * n  # loose upper bound on any row sum
    cap = (r_upper_bound / n) * cap_multiplier * torch.ones(n)
    return X, pi, Lambda, w, sigma2, eta, cap


def feasible_shapes_case() -> bool:
    torch.manual_seed(0)
    n, k, d = 10, 2, 3
    X, pi, Lambda, w, sigma2, eta, cap = _toy_inputs(n, k, d, eta_value=5.0, cap_multiplier=50.0)

    W_true, r, c, z = sample_synthetic_network(
        X, pi, Lambda, w, sigma2, eta, cap, CapacityProjectionConfig(), SyntheticConfig()
    )

    shape_ok = W_true.shape == (n, n) and r.shape == (n,) and c.shape == (n,) and z.shape == (n,)
    finite_ok = torch.isfinite(W_true).all().item()
    diag_ok = torch.allclose(torch.diag(W_true), torch.zeros(n), atol=1e-5)
    nonneg_ok = (W_true >= -1e-6).all().item()
    z_range_ok = bool((z >= 0).all() and (z < k).all())

    passed = shape_ok and finite_ok and diag_ok and nonneg_ok and z_range_ok
    print(
        f"[feasible_shapes_case] shape_ok={shape_ok} finite_ok={finite_ok} diag_ok={diag_ok} "
        f"nonneg_ok={nonneg_ok} z_range_ok={z_range_ok} -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def zero_inflation_rate_case() -> bool:
    """Statistical check on Eq. 4: with eta strongly negative, most entries should be
    exactly zero; with eta strongly positive, almost none should be. Uses a large N so
    the empirical rate concentrates near the true sigmoid(eta) probability.
    """
    torch.manual_seed(1)
    n, k, d = 60, 2, 3

    results = []
    for eta_value, expect_mostly_zero in [(-6.0, True), (6.0, False)]:
        X, pi, Lambda, w, sigma2, eta, cap = _toy_inputs(
            n, k, d, eta_value=eta_value, cap_multiplier=50.0
        )
        W_true, _, _, _ = sample_synthetic_network(
            X, pi, Lambda, w, sigma2, eta, cap, CapacityProjectionConfig(), SyntheticConfig()
        )
        off_diag = W_true[~torch.eye(n, dtype=torch.bool)]
        zero_rate = (off_diag.abs() < 1e-6).float().mean().item()
        ok = zero_rate > 0.9 if expect_mostly_zero else zero_rate < 0.1
        results.append(ok)
        print(
            f"[zero_inflation_rate_case] eta={eta_value} zero_rate={zero_rate:.3f} "
            f"(expect {'mostly zero' if expect_mostly_zero else 'mostly nonzero'}) "
            f"-> {'PASS' if ok else 'FAIL'}"
        )

    passed = all(results)
    return passed


def reproducibility_case() -> bool:
    n, k, d = 8, 2, 3
    X, pi, Lambda, w, sigma2, eta, cap = _toy_inputs(n, k, d, eta_value=2.0, cap_multiplier=50.0)

    gen1 = torch.Generator().manual_seed(42)
    W1, r1, c1, z1 = sample_synthetic_network(
        X, pi, Lambda, w, sigma2, eta, cap, CapacityProjectionConfig(), SyntheticConfig(),
        generator=gen1,
    )
    gen2 = torch.Generator().manual_seed(42)
    W2, r2, c2, z2 = sample_synthetic_network(
        X, pi, Lambda, w, sigma2, eta, cap, CapacityProjectionConfig(), SyntheticConfig(),
        generator=gen2,
    )

    passed = torch.equal(W1, W2) and torch.equal(z1, z2)
    print(f"[reproducibility_case] same_seed_identical_output={passed} -> {'PASS' if passed else 'FAIL'}")
    return passed


def cap_respected_case() -> bool:
    torch.manual_seed(2)
    n, k, d = 10, 2, 3
    X, pi, Lambda, w, sigma2, eta, cap = _toy_inputs(n, k, d, eta_value=5.0, cap_multiplier=1.3)

    W_true, r, c, _ = sample_synthetic_network(
        X, pi, Lambda, w, sigma2, eta, cap, CapacityProjectionConfig(), SyntheticConfig()
    )

    max_violation = (W_true - cap.unsqueeze(1)).max().item()
    passed = max_violation < 1e-4
    print(
        f"[cap_respected_case] max_cap_violation={max_violation:.2e} "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return passed


if __name__ == "__main__":
    results = {
        "feasible_shapes_case": feasible_shapes_case(),
        "zero_inflation_rate_case": zero_inflation_rate_case(),
        "reproducibility_case": reproducibility_case(),
        "cap_respected_case": cap_respected_case(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
