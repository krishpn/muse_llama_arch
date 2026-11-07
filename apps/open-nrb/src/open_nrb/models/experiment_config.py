#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""Load YAML experiment configs for the real and synthetic experiment runners.

Backs run_real_experiment.py / run_synthetic_experiment.py (experiments/configs/
*.yaml) -- lets a named variant (baseline, an ablation, a sweep) be a file diff
instead of a code edit. Dataclass fields have no defaults: a config file is expected
to be complete, same "caller must set it explicitly" convention as
TrainStepConfig.lambda_smooth in config.py.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """Hold the training-loop hyperparameters for one experiment run."""

    epochs: int
    eval_every: int
    lr: float
    kl_beta_max: float
    kl_warmup_epochs: int
    kl_free_bits: float
    lambda_mi: float
    lambda_smooth: float


@dataclass
class ModelConfig:
    """Hold the encoder/decoder architecture dimensions."""

    hidden_dim: int
    decoder_hidden_dim: int
    num_blocks: int


@dataclass
class RealExperimentConfig:
    """Bundle the full configuration for a real-data experiment run."""

    training: TrainingConfig
    model: ModelConfig
    kappa: float
    seed: int


@dataclass
class SyntheticScenarioConfig:
    """Define the ground-truth SBM scenario sampled for a synthetic experiment."""

    n_banks: int
    n_periods: int
    covariate_dim: int
    pi: list[float]
    lambda_matrix: list[list[float]]
    eta_block: list[list[float]]
    w: list[float]
    sigma2: float
    cap: float


@dataclass
class SyntheticExperimentConfig:
    """Bundle the full configuration for a synthetic experiment run."""

    training: TrainingConfig
    model: ModelConfig
    scenario: SyntheticScenarioConfig
    seeds: list[int]


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_real_experiment_config(path: Path) -> RealExperimentConfig:
    """Load and parse a real-experiment YAML config into a RealExperimentConfig."""
    raw = _load_yaml(path)
    return RealExperimentConfig(
        training=TrainingConfig(**raw["training"]),
        model=ModelConfig(**raw["model"]),
        kappa=raw["kappa"],
        seed=raw["seed"],
    )


def load_synthetic_experiment_config(path: Path) -> SyntheticExperimentConfig:
    """Load and parse a synthetic-experiment YAML config into a SyntheticExperimentConfig."""
    raw = _load_yaml(path)
    return SyntheticExperimentConfig(
        training=TrainingConfig(**raw["training"]),
        model=ModelConfig(**raw["model"]),
        scenario=SyntheticScenarioConfig(**raw["scenario"]),
        seeds=raw["seeds"],
    )
