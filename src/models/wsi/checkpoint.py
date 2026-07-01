"""Checkpoint utilities for WSI-level tile importance forecasters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.models.wsi.tile_attention_forecaster import (
    WSI_TILE_IMPORTANCE_LOSS_TYPES,
    WSITileAttentionForecaster,
    WSITileImportanceForecaster,
)


_SCHEMA_VERSION = 1
_ATTENTION_MODEL_TYPE = "WSITileAttentionForecaster"
_IMPORTANCE_MODEL_TYPE = "WSITileImportanceForecaster"
_LOADABLE_MODEL_TYPES = (_ATTENTION_MODEL_TYPE, _IMPORTANCE_MODEL_TYPE)


@dataclass(frozen=True)
class WSITileAttentionForecasterConfig:
    """Serializable config for ``WSITileAttentionForecaster``."""

    feature_dim: int
    hidden_dim: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    def build(self) -> WSITileAttentionForecaster:
        return WSITileAttentionForecaster(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )


@dataclass(frozen=True)
class WSITileAttentionForecasterCheckpoint:
    """Loaded WSI tile attention forecaster checkpoint."""

    model: WSITileAttentionForecaster
    config: WSITileAttentionForecasterConfig
    epoch: int | None
    metrics: dict[str, float]
    metadata: dict[str, Any]


def save_wsi_tile_attention_forecaster_checkpoint(
    path: str | Path,
    *,
    model: WSITileAttentionForecaster,
    config: WSITileAttentionForecasterConfig,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a WSI tile attention forecaster checkpoint."""

    if not isinstance(model, WSITileAttentionForecaster):
        raise TypeError(
            "model must be a WSITileAttentionForecaster; "
            f"got {type(model).__name__}."
        )

    if not isinstance(config, WSITileAttentionForecasterConfig):
        raise TypeError(
            "config must be a WSITileAttentionForecasterConfig; "
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
        "model_type": "WSITileAttentionForecaster",
        "model_config": asdict(config),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "metadata": metadata,
    }

    torch.save(payload, output_path)


def load_wsi_tile_attention_forecaster_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> WSITileAttentionForecasterCheckpoint:
    """Load a WSI tile attention forecaster checkpoint."""

    payload = torch.load(Path(path), map_location=map_location)

    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary.")

    schema_version = payload.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version: {schema_version!r}; "
            f"expected {_SCHEMA_VERSION}."
        )

    model_type = payload.get("model_type")
    if model_type != "WSITileAttentionForecaster":
        raise ValueError(
            f"unsupported model_type: {model_type!r}; "
            "expected 'WSITileAttentionForecaster'."
        )

    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing a valid model_config.")

    config = WSITileAttentionForecasterConfig(**raw_config)
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

    return WSITileAttentionForecasterCheckpoint(
        model=model,
        config=config,
        epoch=epoch,
        metrics={str(key): float(value) for key, value in raw_metrics.items()},
        metadata=raw_metadata,
    )


# Backward-compatible aliases: the config/checkpoint shape did not change when
# the model gained non-attention importance targets, only the semantic name
# did (see WSITileImportanceForecaster in tile_attention_forecaster.py).
WSITileImportanceForecasterConfig = WSITileAttentionForecasterConfig
WSITileImportanceForecasterCheckpoint = WSITileAttentionForecasterCheckpoint


def save_wsi_tile_importance_forecaster_checkpoint(
    path: str | Path,
    *,
    model: WSITileImportanceForecaster,
    config: WSITileImportanceForecasterConfig,
    loss: str,
    target_type: str = "tile_importance",
    target_source: str | None = None,
    input_feature_store: str | None = None,
    target_feature_store: str | None = None,
    alignment_mode: str | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a WSI tile importance forecaster checkpoint.

    This writes the same on-disk schema as
    ``save_wsi_tile_attention_forecaster_checkpoint`` (so both are loadable by
    ``load_wsi_tile_importance_forecaster_checkpoint``), but stamps
    ``model_type="WSITileImportanceForecaster"`` and auto-populates
    ``metadata`` with the fields needed for downstream evaluation/pruning:
    ``model_type``, ``input_feature_dim``, ``hidden_dim``, ``n_heads``,
    ``n_layers``, ``loss``, ``target_type``, ``target_source``,
    ``input_feature_store``, ``target_feature_store``, ``alignment_mode``.
    Caller-provided ``metadata`` is layered on top (e.g. split slide ids,
    seed, top_k).
    """

    if not isinstance(model, WSITileImportanceForecaster):
        raise TypeError(
            "model must be a WSITileImportanceForecaster; "
            f"got {type(model).__name__}."
        )

    if not isinstance(config, WSITileImportanceForecasterConfig):
        raise TypeError(
            "config must be a WSITileImportanceForecasterConfig; "
            f"got {type(config).__name__}."
        )

    if loss not in WSI_TILE_IMPORTANCE_LOSS_TYPES:
        raise ValueError(
            f"loss must be one of {WSI_TILE_IMPORTANCE_LOSS_TYPES}; got {loss!r}."
        )

    if epoch is not None and epoch < 0:
        raise ValueError("epoch must be non-negative when provided.")

    metrics = {} if metrics is None else dict(metrics)
    caller_metadata = {} if metadata is None else dict(metadata)

    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError("metric names must be strings.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"metric '{key}' must be numeric.")

    required_metadata: dict[str, Any] = {
        "model_type": _IMPORTANCE_MODEL_TYPE,
        "input_feature_dim": config.feature_dim,
        "hidden_dim": config.hidden_dim,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "loss": loss,
        "target_type": target_type,
        "target_source": target_source,
        "input_feature_store": input_feature_store,
        "target_feature_store": target_feature_store,
        "alignment_mode": alignment_mode,
    }
    full_metadata = {**required_metadata, **caller_metadata}

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "model_type": _IMPORTANCE_MODEL_TYPE,
        "model_config": asdict(config),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "metadata": full_metadata,
    }

    torch.save(payload, output_path)


def load_wsi_tile_importance_forecaster_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> WSITileImportanceForecasterCheckpoint:
    """Load a WSI tile importance forecaster checkpoint.

    Accepts checkpoints written by either
    ``save_wsi_tile_importance_forecaster_checkpoint`` or the legacy
    ``save_wsi_tile_attention_forecaster_checkpoint`` (``model_type`` in
    ``{"WSITileImportanceForecaster", "WSITileAttentionForecaster"}``), since
    both describe the same architecture.
    """

    payload = torch.load(Path(path), map_location=map_location)

    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary.")

    schema_version = payload.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version: {schema_version!r}; "
            f"expected {_SCHEMA_VERSION}."
        )

    model_type = payload.get("model_type")
    if model_type not in _LOADABLE_MODEL_TYPES:
        raise ValueError(
            f"unsupported model_type: {model_type!r}; "
            f"expected one of {_LOADABLE_MODEL_TYPES}."
        )

    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing a valid model_config.")

    config = WSITileImportanceForecasterConfig(**raw_config)
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

    return WSITileImportanceForecasterCheckpoint(
        model=model,
        config=config,
        epoch=epoch,
        metrics={str(key): float(value) for key, value in raw_metrics.items()},
        metadata=raw_metadata,
    )
