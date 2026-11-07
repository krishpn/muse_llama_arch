#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Align the covariate/dose/outcome panel to the saved artifact's index set.

Aligns to the exact (bank, period) index set already fixed by open_nrb's saved
artifact (artifact_io.Paper2Artifact).

Imports open_nrb.models.data_loader read-only -- never edits it, never
retrains. load_panel/filter_commercial_banks/filter_from_period are the same
calls open_nrb.models.run_real_experiment.main() makes before its own
complete-case filter; Paper 2 reuses them so the raw panel is built exactly
the same way, then reindexes down to the artifact's own bfi_names/periods
(which already reflect open_nrb's complete-case filter) instead of
re-deriving a possibly-different complete-case set itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    from open_nrb.models.data_loader import (
        COVARIATE_ORDER,
        MIN_PERIOD,
        Panel,
        filter_commercial_banks,
        filter_from_period,
        load_panel,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from models.data_loader import (  # type: ignore[no-redef]
        COVARIATE_ORDER,
        MIN_PERIOD,
        Panel,
        filter_commercial_banks,
        filter_from_period,
        load_panel,
    )

try:
    from .artifact_io import Paper2Artifact
except ImportError:
    from artifact_io import Paper2Artifact  # type: ignore[no-redef]


def load_commercial_panel(csv_path: Path, start: tuple[int, str] = MIN_PERIOD) -> Panel:
    """Load the raw commercial-bank panel the way open_nrb's real run does.

    Same raw-load + scoping open_nrb.models.run_real_experiment.main() does,
    minus its own complete-case filter (that's reproduced by aligning against
    the artifact below instead). start: first period kept (Paper 1's MIN_PERIOD;
    total_effect passes its earlier TOTAL_EFFECT_START).
    """
    panel = load_panel(csv_path)
    panel = filter_commercial_banks(panel)
    panel = filter_from_period(panel, start)
    return panel


def reindex_panel(panel: Panel, bfi_names: list[str], periods: list) -> Panel:
    """Restrict and reorder panel to the given bfi_names x periods (all present in panel)."""
    name_pos = {name: i for i, name in enumerate(panel.bfi_names)}
    period_pos = {period: t for t, period in enumerate(panel.periods)}
    bank_idx = torch.tensor([name_pos[n] for n in bfi_names], dtype=torch.long)
    period_idx = torch.tensor([period_pos[p] for p in periods], dtype=torch.long)
    return Panel(
        bfi_names=list(bfi_names),
        periods=list(periods),
        X=panel.X[bank_idx][:, period_idx, :],
        r=panel.r[bank_idx][:, period_idx],
        c=panel.c[bank_idx][:, period_idx],
        K=panel.K[bank_idx][:, period_idx],
    )


def align_to_artifact(panel: Panel, artifact: Paper2Artifact) -> Panel:
    """Reindex panel down to exactly artifact.bfi_names x artifact.periods.

    Raises if the artifact references a bank or period the raw panel doesn't
    have -- that would mean the CSV passed here isn't the one open_nrb's
    saved run actually used, which Paper 2 must not silently paper over.
    """
    missing_names = [n for n in artifact.bfi_names if n not in panel.bfi_names]
    missing_periods = [p for p in artifact.periods if p not in panel.periods]
    if missing_names or missing_periods:
        raise ValueError(
            "panel is not a superset of the artifact's index set -- "
            f"missing banks: {missing_names}, missing periods: {missing_periods}. "
            "Is --csv the same file open_nrb's saved run used?"
        )
    return reindex_panel(panel, artifact.bfi_names, artifact.periods)


def check_marginals(panel: Panel, artifact: Paper2Artifact, rtol: float = 1e-4) -> float | None:
    """Check the aligned panel's marginals against the artifact's and return kappa.

    Value check behind align_to_artifact's index check: the aligned panel's r, c and K must be
    the marginals and cap open_nrb's saved run projected W_hat onto (same CSV, same extraction).
    Raises on a mismatch. Returns the implied kappa (cap / K, constant across cells), or None if
    the artifact predates the r/c/cap keys.
    """
    if artifact.r is None or artifact.c is None or artifact.cap is None:
        return None
    for name, ours, theirs in [("r", panel.r, artifact.r), ("c", panel.c, artifact.c)]:
        if ours.shape != theirs.shape or not torch.allclose(ours, theirs.to(ours.dtype), rtol=rtol, atol=1e-6):
            diff = (ours - theirs.to(ours.dtype)).abs().max().item() if ours.shape == theirs.shape else float("nan")
            raise ValueError(f"panel.{name} does not match the artifact's {name} (max abs diff {diff:.4g}) -- "
                             "the CSV is not the one open_nrb's saved run used")
    pos = panel.K > 0
    kappa = artifact.cap.to(panel.K.dtype)[pos] / panel.K[pos]
    if kappa.numel() == 0 or (kappa.max() - kappa.min()).abs().item() > rtol * kappa.mean().abs().item():
        raise ValueError("artifact cap / panel K is not a single constant -- K differs from the saved run's")
    return kappa.mean().item()


def capital_ratio(panel: Panel) -> torch.Tensor:
    """Return (N, T) capital ratio = CAPITAL FUND / size.

    panel.X[..., COVARIATE_ORDER.index('capital_ratio')], already computed by
    data_loader._compute_fields.
    """
    return panel.X[:, :, COVARIATE_ORDER.index("capital_ratio")]
