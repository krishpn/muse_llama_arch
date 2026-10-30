#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""YAML config loading for run_real_experiment.py / run_synthetic_experiment.py (experiments/
configs/*.yaml) -- lets a named variant (baseline, an ablation, a sweep) be a file diff instead
of a code edit. Dataclass fields have no defaults: a config file is expected to be complete,
same "caller must set it explicitly" convention as TrainStepConfig.lambda_smooth/lambda_reg in
config.py.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    epochs: int
    eval_every: int
    lr: float
    kl_beta_max: float
    kl_warmup_epochs: int
    lambda_smooth: float
    lambda_reg: float


@dataclass
class ModelConfig:
    hidden_dim: int
    decoder_hidden_dim: int
    num_blocks: int


@dataclass
class RealExperimentConfig:
    training: TrainingConfig
    model: ModelConfig
    kappa: float
    seed: int


@dataclass
class SyntheticScenarioConfig:
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
    training: TrainingConfig
    model: ModelConfig
    scenario: SyntheticScenarioConfig
    seeds: list[int]


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_real_experiment_config(path: Path) -> RealExperimentConfig:
    raw = _load_yaml(path)
    return RealExperimentConfig(
        training=TrainingConfig(**raw["training"]),
        model=ModelConfig(**raw["model"]),
        kappa=raw["kappa"],
        seed=raw["seed"],
    )


def load_synthetic_experiment_config(path: Path) -> SyntheticExperimentConfig:
    raw = _load_yaml(path)
    return SyntheticExperimentConfig(
        training=TrainingConfig(**raw["training"]),
        model=ModelConfig(**raw["model"]),
        scenario=SyntheticScenarioConfig(**raw["scenario"]),
        seeds=raw["seeds"],
    )
