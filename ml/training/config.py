"""Experiment configuration (YAML ↔ dataclass)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    name: str = "unet"
    seed: int = 42
    # data
    split_scheme: str = "official"  # official | geo_holdout_v1
    raw_root: str = "data/raw/sen1floods11"
    processed_root: str = "data/processed/sen1floods11"
    eval_splits: list[str] = field(default_factory=lambda: ["test"])
    preload: bool = True
    # model
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.1
    # optimisation
    epochs: int = 80
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    pos_weight: float | str | None = "auto_sqrt"  # number, None, or auto_sqrt
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    crop: int | None = 256
    noise_std: float = 0.05
    amp: bool = False
    grad_clip: float | None = 1.0
    # early stopping on val IoU
    patience: int = 15
    min_delta: float = 1e-3
    threshold: float = 0.5
    # runtime
    device: str = "auto"
    num_workers: int = 0
    output_root: str = "data/checkpoints"
    mlflow: bool = True
    mlflow_uri: str = "sqlite:///data/mlflow/mlflow.db"
    mlflow_experiment: str = "sen1floods11-segmentation"
    max_train_batches: int | None = None  # for smoke tests
    latency_size: int = 512

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
