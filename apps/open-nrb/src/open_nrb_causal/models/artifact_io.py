#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Load open_nrb's already-saved real-data artifact, read-only.

results/<config>/real_run_seed{seed}.pt, written by
open_nrb.models.run_real_experiment.main(). Paper 2 never retrains and never
imports open_nrb.models.train -- it only consumes the point-estimate W_hat
this artifact already contains.

The saved dict has more keys than Paper2Artifact declares (q, losses,
components_trend, model_stability, baseline_stability, baseline_diff) --
load_artifact() below pulls fields out explicitly rather than
Paper2Artifact(**data), which would raise TypeError against unknown keys.
r, c, cap and the NaN-filled period lists are optional (older artifacts lack
r/c); panel_alignment.check_marginals uses them when present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Paper2Artifact:
    """Hold the point estimates and index set from one saved open_nrb real-data run."""

    bfi_names: list[str]
    periods: list[tuple[int, str]]
    W_hat: torch.Tensor  # (T, N, N), posterior-mean point estimate
    W_baseline: torch.Tensor  # (T, N, N), uncapped Sinkhorn/RAS baseline -- robustness.py only
    seed: int
    r: torch.Tensor | None = None  # (N, T) row marginals W_hat was projected onto, NPR mn
    c: torch.Tensor | None = None  # (N, T) column marginals, NPR mn
    cap: torch.Tensor | None = None  # (N, T) kappa * K, NPR mn
    non_finite_periods: list[int] = field(default_factory=list)  # W_hat NaN-filled (model)
    degenerate_periods: list[int] = field(default_factory=list)  # W_hat NaN-filled, W_baseline zero


_REQUIRED_KEYS = ("bfi_names", "periods", "W_hat", "W_baseline", "seed")


def load_artifact(path: Path) -> Paper2Artifact:
    """Load and validate a saved artifact at `path` into a Paper2Artifact.

    Raises KeyError if a required key is missing and ValueError if W_hat's shape
    disagrees with bfi_names/periods; optional r/c/cap and the NaN-filled period
    lists are pulled out when present.
    """
    data = torch.load(path, map_location="cpu")
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"{path}: artifact missing expected key(s) {missing} -- was this saved by "
            "open_nrb.models.run_real_experiment.main(), and is it the current save-dict shape?"
        )
    n_names = len(data["bfi_names"])
    n_periods = len(data["periods"])
    t, n_rows, n_cols = data["W_hat"].shape
    if t != n_periods or n_rows != n_names or n_cols != n_names:
        raise ValueError(
            f"{path}: shape mismatch -- W_hat is {tuple(data['W_hat'].shape)} but "
            f"bfi_names has {n_names} entries and periods has {n_periods} entries"
        )
    return Paper2Artifact(
        bfi_names=list(data["bfi_names"]),
        periods=list(data["periods"]),
        W_hat=data["W_hat"],
        W_baseline=data["W_baseline"],
        seed=int(data["seed"]),
        r=data.get("r"),
        c=data.get("c"),
        cap=data.get("cap"),
        non_finite_periods=[int(t) for t in data.get("non_finite_periods", [])],
        degenerate_periods=[int(t) for t in data.get("degenerate_periods", [])],
    )
