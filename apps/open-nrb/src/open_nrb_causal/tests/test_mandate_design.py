#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Tests for the fiscal calendar and DiD/event-study design construction (mandate_design.py)."""

import math
import sys
from pathlib import Path

import torch

try:
    from open_nrb_causal.models.mandate_design import (
        decumulate_fiscal_ytd,
        did_design,
        event_bins,
        fiscal_year,
        freeze_pre_mandate,
        mandate_timing,
        quarterly_from_fiscal_ytd,
        ytd_cumulation_diagnostic,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from mandate_design import (  # type: ignore[no-redef]
        decumulate_fiscal_ytd,
        did_design,
        event_bins,
        fiscal_year,
        freeze_pre_mandate,
        mandate_timing,
        quarterly_from_fiscal_ytd,
        ytd_cumulation_diagnostic,
    )

# Real panel's first 16 months: Asar 2072 .. Kartik 2073.
_MONTHS = ["Asar", "Shrawan", "Bhadra", "Ashwin", "Kartik", "Mangsir", "Poush", "Magh",
           "Falgun", "Chaitra", "Baisakh", "Jestha", "Asar", "Shrawan", "Bhadra", "Ashwin"]
_PERIODS = [(2072 if i < 10 else 2073, m) for i, m in enumerate(_MONTHS)]


def fiscal_year_rolls_over_at_shrawan_case() -> bool:
    """Assert fiscal_year rolls over to the new BS fiscal year at Shrawan.

    The Nepali fiscal year starts at Shrawan: (2073, Shrawan) is FY 2073, while
    Asar, Baisakh, and Chaitra of the surrounding BS years map to FY 2072.
    """
    passed = (
        fiscal_year((2073, "Asar")) == 2072
        and fiscal_year((2073, "Shrawan")) == 2073
        and fiscal_year((2073, "Baisakh")) == 2072
        and fiscal_year((2072, "Chaitra")) == 2072
    )
    print(f"[fiscal_year_rolls_over_at_shrawan_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def event_bins_match_real_mandate_timing_case() -> bool:
    """Assert event_bins and timing.post match the real mandate timing.

    For the real 16-month panel the event bins equal
    [-2] + [-1]*5 + ["transition"]*7 + [0]*3, and the post indicator is 1 only
    for the final 3 (effective) months.
    """
    timing = mandate_timing(_PERIODS, (2072, "Mangsir"), (2073, "Asar"))
    bins = event_bins(_PERIODS, timing)
    expected = [-2] + [-1] * 5 + ["transition"] * 7 + [0] * 3
    post = timing.post(len(_PERIODS))
    passed = bins == expected and post.tolist() == [0.0] * 13 + [1.0] * 3
    print(f"[event_bins_match_real_mandate_timing_case] bins={bins} -> {'PASS' if passed else 'FAIL'}")
    return passed


def mandate_timing_rejects_bad_config_case() -> bool:
    """Assert mandate_timing raises ValueError for each invalid configuration.

    All three bad configs (effective period not in the panel, pre-end after the
    effective period, no post cells) raise ValueError (raised == 3).
    """
    raised = 0
    for pre_end, eff in [((2072, "Mangsir"), (2090, "Asar")),  # not in panel
                         ((2073, "Asar"), (2072, "Mangsir")),  # pre after effective
                         ((2072, "Mangsir"), (2073, "Ashwin"))]:  # no post cells
        try:
            mandate_timing(_PERIODS, pre_end, eff)
        except ValueError:
            raised += 1
    passed = raised == 3
    print(f"[mandate_timing_rejects_bad_config_case] raised={raised}/3 -> {'PASS' if passed else 'FAIL'}")
    return passed


def freeze_pre_mandate_averages_pre_window_only_case() -> bool:
    """Assert freeze_pre_mandate averages only the pre-mandate window, skipping NaNs.

    The per-bank output is the mean over pre-window cells ignoring NaN
    ([1, 3] -> 2, [nan, 4] -> 4); an all-NaN pre window raises ValueError.
    """
    timing = mandate_timing(_PERIODS, (2072, "Shrawan"), (2073, "Asar"))
    x = torch.tensor([[1.0, 3.0] + [100.0] * 14, [float("nan"), 4.0] + [100.0] * 14])
    out = freeze_pre_mandate(x, timing)
    passed = torch.allclose(out, torch.tensor([2.0, 4.0]))
    try:
        freeze_pre_mandate(torch.full((1, 16), float("nan")), timing)
        passed = False
    except ValueError:
        pass
    print(f"[freeze_pre_mandate_averages_pre_window_only_case] out={out.tolist()} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def decumulate_and_diagnose_ytd_case() -> bool:
    """Assert decumulate_fiscal_ytd recovers flows and the YTD diagnostic reads 1/12 vs 1.

    Monthly flow of 10/month reported as fiscal-YTD cumulative (resetting at
    Shrawan): de-cumulation recovers 10 everywhere except the panel's first
    month (no previous column, NaN), and ytd_cumulation_diagnostic reads ~1/12
    for the cumulative series and ~1 for the flow series.
    """
    fy_month = [12] + list(range(1, 13)) + [1, 2, 3]  # Asar=12th month of FY, Shrawan=1st
    cumulative = torch.tensor([[10.0 * m for m in fy_month]])
    flow = decumulate_fiscal_ytd(cumulative, _PERIODS)
    ok_flow = math.isnan(flow[0, 0].item()) and torch.allclose(flow[0, 1:], torch.full((15,), 10.0))
    ratio_cum = ytd_cumulation_diagnostic(cumulative, _PERIODS)
    ratio_flow = ytd_cumulation_diagnostic(torch.full((1, 16), 10.0), _PERIODS)
    passed = ok_flow and abs(ratio_cum - 1 / 12) < 1e-6 and abs(ratio_flow - 1.0) < 1e-6
    print(f"[decumulate_and_diagnose_ytd_case] ratio_cum={ratio_cum:.4f} ratio_flow={ratio_flow:.4f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def quarterly_from_fiscal_ytd_case() -> bool:
    """Assert quarterly_from_fiscal_ytd differences YTD into 30-per-quarter flows.

    Same 10/month YTD series: every quarter end with a previous quarter end in
    the panel reads 30, Ashwin reads its own YTD (30), the first Asar (no
    Chaitra 2071) is NaN, and non-quarter months are NaN. Asar 2073 - Chaitra
    2072 crosses the BS year boundary.
    """
    fy_month = [12] + list(range(1, 13)) + [1, 2, 3]
    cumulative = torch.tensor([[10.0 * m for m in fy_month]])
    q = quarterly_from_fiscal_ytd(cumulative, _PERIODS)[0]
    quarter_idx = [3, 6, 9, 12, 15]  # Ashwin, Poush, Chaitra 2072, Asar 2073, Ashwin 2073
    others = [i for i in range(len(_PERIODS)) if i not in quarter_idx]
    passed = (
        torch.allclose(q[quarter_idx], torch.full((5,), 30.0))
        and all(math.isnan(q[i].item()) for i in others)
    )
    print(f"[quarterly_from_fiscal_ytd_case] q={q.tolist()} -> {'PASS' if passed else 'FAIL'}")
    return passed


def did_design_uses_own_network_weights_case() -> bool:
    """Assert did_design uses each network's own weights and zeroes transition cells.

    Guards the RAS-robustness bug fixed 2026-11-03: two different networks yield
    different exposures AND different HT weights, transition cells are
    zero-weighted in the pooled DiD but kept (> 0) for the event study, dose_x_post
    is zero up to and including the effective period, and the controls are
    exactly {exposure_level, liquidity_ratio_pre_x_post}.
    """
    torch.manual_seed(0)
    n, t = 6, len(_PERIODS)
    timing = mandate_timing(_PERIODS, (2072, "Mangsir"), (2073, "Asar"))
    dose_pre = torch.randn(n)
    controls_pre = {"liquidity_ratio": torch.rand(n)}
    a = did_design(torch.rand(t, n, n), dose_pre, controls_pre, timing)
    b = did_design(torch.rand(t, n, n), dose_pre, controls_pre, timing)
    transition = timing.transition(t).bool()
    passed = (
        not torch.allclose(a.exposure, b.exposure)
        and not torch.allclose(a.event_weights, b.event_weights)
        and (a.weights[:, transition] == 0).all().item()
        and (a.event_weights[:, transition] > 0).all().item()
        and (a.dose_x_post[:, : timing.effective_idx + 1] == 0).all().item()
        and set(a.controls) == {"exposure_level", "liquidity_ratio_pre_x_post"}
    )
    print(f"[did_design_uses_own_network_weights_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


CASES = [
    fiscal_year_rolls_over_at_shrawan_case,
    event_bins_match_real_mandate_timing_case,
    mandate_timing_rejects_bad_config_case,
    freeze_pre_mandate_averages_pre_window_only_case,
    decumulate_and_diagnose_ytd_case,
    quarterly_from_fiscal_ytd_case,
    did_design_uses_own_network_weights_case,
]


def test_fiscal_year_rolls_over_at_shrawan():
    """Assert fiscal_year_rolls_over_at_shrawan passes."""
    assert fiscal_year_rolls_over_at_shrawan_case()


def test_event_bins_match_real_mandate_timing():
    """Assert event_bins_match_real_mandate_timing passes."""
    assert event_bins_match_real_mandate_timing_case()


def test_mandate_timing_rejects_bad_config():
    """Assert mandate_timing_rejects_bad_config passes."""
    assert mandate_timing_rejects_bad_config_case()


def test_freeze_pre_mandate_averages_pre_window_only():
    """Assert freeze_pre_mandate_averages_pre_window_only passes."""
    assert freeze_pre_mandate_averages_pre_window_only_case()


def test_decumulate_and_diagnose_ytd():
    """Assert decumulate_and_diagnose_ytd passes."""
    assert decumulate_and_diagnose_ytd_case()


def test_quarterly_from_fiscal_ytd():
    """Assert quarterly_from_fiscal_ytd passes."""
    assert quarterly_from_fiscal_ytd_case()


def test_did_design_uses_own_network_weights():
    """Assert did_design_uses_own_network_weights passes."""
    assert did_design_uses_own_network_weights_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
