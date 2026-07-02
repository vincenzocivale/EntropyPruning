"""Checkpoint utilities for WSI-level ABMIL classifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.models.wsi.abmil import ABMILClassifier


_SCHEMA_VERSION = 1
_MODEL_TYPE = "ABMILClassifier"


def load_trusted_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a trusted EAF ABMIL checkpoint with full metadata."""

    return torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )


@dataclass(frozen=True)
class ABMILClassifierConfig:
    """Serializable config for ``ABMILClassifier``."""

    feature_dim: int
    hidden_dim: int = 256
    n_classes: int = 2
    dropout: float = 0.1
    gated: bool = True

    def build(self) -> ABMILClassifier:
        return ABMILClassifier(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            n_classes=self.n_classes,
            dropout=self.dropout,
            gated=self.gated,
        )


@dataclass(frozen=True)
class ABMILClassifierCheckpoint:
    """Loaded ABMIL classifier checkpoint."""

    model: ABMILClassifier
    config: ABMILClassifierConfig
    epoch: int | None
    metrics: dict[str, float]
    metadata: dict[str, Any]


def save_abmil_classifier_checkpoint(
    path: str | Path,
    *,
    model: ABMILClassifier,
    config: ABMILClassifierConfig,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save an ABMIL classifier checkpoint."""

    if not isinstance(model, ABMILClassifier):
        raise TypeError(
            "model must be an ABMILClassifier; "
            f"got {type(model).__name__}."
        )

    if not isinstance(config, ABMILClassifierConfig):
        raise TypeError(
            "config must be an ABMILClassifierConfig; "
            f"got {type(config).__name__}."
        )

    if epoch is not None and epoch < 0:
        raise ValueError("epoch must be non-negative when provided.")

    metrics = {} if metrics is None else dict(metrics)
    metadata = {} if metadata is None else dict(metadata)

    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError("metric names must be strings.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"metric '{key}' must be numeric.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "model_type": _MODEL_TYPE,
        "model_config": asdict(config),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "metadata": metadata,
    }

    torch.save(payload, output_path)


def load_abmil_classifier_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ABMILClassifierCheckpoint:
    """Load an ABMIL classifier checkpoint."""

    payload = load_trusted_training_checkpoint(path, map_location=map_location)

    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary.")

    schema_version = payload.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version: {schema_version!r}; "
            f"expected {_SCHEMA_VERSION}."
        )

    model_type = payload.get("model_type")
    if model_type != _MODEL_TYPE:
        raise ValueError(
            f"unsupported model_type: {model_type!r}; "
            f"expected '{_MODEL_TYPE}'."
        )

    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing a valid model_config.")

    config = ABMILClassifierConfig(**raw_config)
    model = config.build()
    model.load_state_dict(payload["state_dict"])

    epoch = payload.get("epoch")
    if epoch is not None:
        epoch = int(epoch)

    raw_metrics = payload.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        raise ValueError("checkpoint metrics must be a dictionary.")

    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary.")

    return ABMILClassifierCheckpoint(
        model=model,
        config=config,
        epoch=epoch,
        metrics={str(key): float(value) for key, value in raw_metrics.items()},
        metadata=raw_metadata,
    )
