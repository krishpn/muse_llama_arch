#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""Load the YAML experiment config for run_causal_experiment.py.

experiments/configs/*.yaml, same no-default-fields convention as
open_nrb.models.experiment_config: a config file is expected to be complete,
loaded via explicit field-by-field dataclass construction (never Config(**raw)).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "experiments" / "configs"


@dataclass
class CausalExperimentConfig:
    """Hold one causal experiment's fully specified configuration (no defaults)."""

    artifact_path: Path
    csv_path: Path
    pnl_csv_path: Path
    capital_floor: float
    paid_up_capital_requirement: float | None  # NPR mn, alternative dose; None = not yet supplied
    pre_window_end: tuple[int, str]  # (year_bs, month_bs), last pre-mandate period (inclusive)
    effective_period: tuple[int, str]  # mandate effective; post = strictly after
    post_window_end: tuple[int, str] | None  # last period kept (inclusive); None = panel end
    pnl_ytd_cumulative: bool  # de-cumulate fiscal-YTD P&L before scaling by size
    blank_as_zero: tuple[str, ...]  # outcome fields whose blanks read as 0 where other P&L was reported (item 11b)
    outcome_construction: str  # "monthly" (de-cumulated if YTD) or "quarterly" (quarter-end YTD only)
    weighting: str  # "unweighted" (main estimator) or "ht" (stabilised HT/GPS weights)
    n_bootstrap: int  # wild-cluster bootstrap draws; 0 = CR1 SEs only
    bootstrap_seed: int
    rdd_bandwidth: float
    rdd_window: tuple[tuple[int, str], tuple[int, str]]  # pooled periods, inclusive


OUTCOME_CONSTRUCTIONS = ("monthly", "quarterly")
WEIGHTINGS = ("unweighted", "ht")


def load_causal_experiment_config(path: Path) -> CausalExperimentConfig:
    """Load and validate the YAML config at `path` into a CausalExperimentConfig.

    Constructs the dataclass field by field (never Config(**raw)) and raises
    ValueError on an unknown outcome_construction/weighting, or a quarterly
    construction without pnl_ytd_cumulative.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = CausalExperimentConfig(
        artifact_path=Path(raw["artifact_path"]),
        csv_path=Path(raw["csv_path"]),
        pnl_csv_path=Path(raw["pnl_csv_path"]),
        capital_floor=raw["capital_floor"],
        paid_up_capital_requirement=(None if raw["paid_up_capital_requirement"] is None
                                     else float(raw["paid_up_capital_requirement"])),
        pre_window_end=tuple(raw["pre_window_end"]),
        effective_period=tuple(raw["effective_period"]),
        post_window_end=None if raw["post_window_end"] is None else tuple(raw["post_window_end"]),
        pnl_ytd_cumulative=bool(raw["pnl_ytd_cumulative"]),
        blank_as_zero=tuple(raw["blank_as_zero"] or ()),
        outcome_construction=raw["outcome_construction"],
        weighting=raw["weighting"],
        n_bootstrap=int(raw["n_bootstrap"]),
        bootstrap_seed=int(raw["bootstrap_seed"]),
        rdd_bandwidth=raw["rdd_bandwidth"],
        rdd_window=(tuple(raw["rdd_window"][0]), tuple(raw["rdd_window"][1])),
    )
    if cfg.outcome_construction not in OUTCOME_CONSTRUCTIONS:
        raise ValueError(f"{path}: outcome_construction must be one of {OUTCOME_CONSTRUCTIONS}")
    if cfg.weighting not in WEIGHTINGS:
        raise ValueError(f"{path}: weighting must be one of {WEIGHTINGS}")
    if cfg.outcome_construction == "quarterly" and not cfg.pnl_ytd_cumulative:
        raise ValueError(f"{path}: quarterly outcomes difference fiscal-YTD figures; "
                         "needs pnl_ytd_cumulative: true")
    return cfg
