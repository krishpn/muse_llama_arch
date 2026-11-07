#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Run Sec 0a/1 robustness checks against the RAS/Gandy-Veraart baseline carryover."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from .estimator import DiDResult, fit_did
    from .mandate_design import DesignInputs
except ImportError:
    from estimator import DiDResult, fit_did  # type: ignore[no-redef]
    from mandate_design import DesignInputs  # type: ignore[no-redef]


@dataclass
class BaselineComparison:
    """Pair the main-network and RAS-baseline DiD fits with their coefficient differences."""

    main: DiDResult
    baseline: DiDResult
    beta_dose_diff: float
    beta_exposure_diff: float


def compare_against_ras_baseline(
    outcome: torch.Tensor,
    design_main: DesignInputs,
    design_baseline: DesignInputs,
    n_boot: int = 0,
    seed: int = 0,
) -> BaselineComparison:
    """Re-run the DiD under the RAS baseline network and compare it to W_hat.

    Re-runs fit_did with the design built from the classical uncapped
    Sinkhorn/RAS baseline (W_baseline, already in the artifact -- no retrain
    needed) instead of the model's W_hat. Each design carries its own
    exposure AND its own propensity/HT weights (mandate_design.did_design
    called once per network) -- reusing W_hat's weights for the RAS run would
    weight it by the wrong exposure's GPS. A result that only holds under
    W_hat and vanishes under W_baseline is a reconstruction-method artifact,
    not evidence of a real spillover (Sec 0a's robustness-check framing).
    """
    def _fit(d: DesignInputs) -> DiDResult:
        return fit_did(outcome, d.dose_x_post, d.exposure_x_post, d.controls, d.weights,
                       n_boot=n_boot, seed=seed)

    main = _fit(design_main)
    baseline = _fit(design_baseline)

    return BaselineComparison(
        main=main,
        baseline=baseline,
        beta_dose_diff=main.beta_dose - baseline.beta_dose,
        beta_exposure_diff=main.beta_exposure - baseline.beta_exposure,
    )
