"""Utilities for importing TRIDENT-style WSI feature outputs.

This module intentionally does not import TRIDENT. It consumes feature-level
artifacts produced by TRIDENT-like pipelines: one HDF5 feature file per slide
and, optionally, one HDF5 coordinate file per slide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.wsi.bag import WSIBag


try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


_DEFAULT_FEATURE_DATASET_NAMES = ("features", "feats", "embeddings", "tile_features")
_DEFAULT_COORD_DATASET_NAMES = ("coords", "coordinates")


@dataclass(frozen=True)
class TridentSlideRecord:
    """Manifest record for one TRIDENT-style slide."""

    slide_id: str
    features_path: Path
    coords_path: Path | None = None
    label: int | float | None = None
    metadata: dict[str, Any] | None = None


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required to import TRIDENT HDF5 features.")


def _read_h5_dataset(
    path: Path,
    *,
    dataset_name: str | None,
    fallback_names: tuple[str, ...],
    expected_ndim: int,
) -> torch.Tensor:
    _require_h5py()

    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        selected_name = _select_dataset_name(
            handle,
            dataset_name=dataset_name,
            fallback_names=fallback_names,
            expected_ndim=expected_ndim,
            path=path,
        )
        array = handle[selected_name][...]

    return torch.as_tensor(array)


def _select_dataset_name(
    handle: Any,
    *,
    dataset_name: str | None,
    fallback_names: tuple[str, ...],
    expected_ndim: int,
    path: Path,
) -> str:
    if dataset_name is not None:
        if dataset_name not in handle:
            raise KeyError(f"dataset '{dataset_name}' not found in {path}")
        dataset = handle[dataset_name]
        if getattr(dataset, "ndim", None) != expected_ndim:
            raise ValueError(
                f"dataset '{dataset_name}' in {path} must be {expected_ndim}D; "
                f"got {getattr(dataset, 'shape', None)}."
            )
        return dataset_name

    for name in fallback_names:
        if name in handle and getattr(handle[name], "ndim", None) == expected_ndim:
            return name

    candidates: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if getattr(obj, "ndim", None) == expected_ndim:
            candidates.append(name)

    handle.visititems(visitor)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            f"no {expected_ndim}D dataset found in {path}; "
            f"tried {fallback_names}."
        )

    raise ValueError(
        f"multiple {expected_ndim}D datasets found in {path}: {candidates}; "
        "pass an explicit dataset name."
    )


def read_trident_features(
    path: str | Path,
    *,
    dataset_name: str | None = None,
) -> torch.Tensor:
    """Read tile features from a TRIDENT-style HDF5 feature file."""

    feature_path = Path(path)
    try:
        features = _read_h5_dataset(
            feature_path,
            dataset_name=dataset_name,
            fallback_names=_DEFAULT_FEATURE_DATASET_NAMES,
            expected_ndim=2,
        )
    except ValueError as exc:
        if "no 2D dataset found" not in str(exc) and "must be 2D" not in str(exc):
            raise
        features = _read_h5_dataset(
            feature_path,
            dataset_name=dataset_name,
            fallback_names=_DEFAULT_FEATURE_DATASET_NAMES,
            expected_ndim=3,
        )
        if features.shape[1] != 1:
            raise ValueError(
                f"features must be 2D or [n_tiles, 1, feature_dim]; got {tuple(features.shape)}."
            ) from exc
        features = features[:, 0, :]

    if not torch.is_floating_point(features):
        features = features.to(torch.float32)

    if features.ndim != 2:
        raise ValueError(f"features must be 2D; got {tuple(features.shape)}.")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one tile.")
    if features.shape[1] == 0:
        raise ValueError("feature dimension must be positive.")
    if not torch.isfinite(features).all():
        raise ValueError("features contain NaN or Inf.")

    return features


def read_trident_coords(
    path: str | Path,
    *,
    dataset_name: str | None = None,
) -> torch.Tensor:
    """Read tile coordinates from a TRIDENT-style HDF5 coordinate file."""

    coords = _read_h5_dataset(
        Path(path),
        dataset_name=dataset_name,
        fallback_names=_DEFAULT_COORD_DATASET_NAMES,
        expected_ndim=2,
    )

    if coords.ndim != 2:
        raise ValueError(f"coords must be 2D; got {tuple(coords.shape)}.")
    if coords.shape[0] == 0:
        raise ValueError("coords must contain at least one tile.")
    if coords.shape[1] not in {2, 4}:
        raise ValueError(
            "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
            f"got {tuple(coords.shape)}."
        )

    return coords.to(torch.long)


def load_trident_slide_record(
    record: TridentSlideRecord,
    *,
    feature_dataset_name: str | None = None,
    coords_dataset_name: str | None = None,
    expected_feature_dim: int | None = None,
) -> WSIBag:
    """Load one manifest record as a ``WSIBag``."""

    features = read_trident_features(
        record.features_path,
        dataset_name=feature_dataset_name,
    )

    if expected_feature_dim is not None and features.shape[1] != expected_feature_dim:
        raise ValueError(
            f"slide {record.slide_id} has feature_dim={features.shape[1]}, "
            f"expected {expected_feature_dim}."
        )

    coords = None
    if record.coords_path is not None:
        coords = read_trident_coords(
            record.coords_path,
            dataset_name=coords_dataset_name,
        )
        if coords.shape[0] != features.shape[0]:
            raise ValueError(
                f"slide {record.slide_id} has {features.shape[0]} features but "
                f"{coords.shape[0]} coordinates."
            )

    metadata = dict(record.metadata) if record.metadata is not None else {}
    metadata.update(
        {
            "source": "trident",
            "trident_features_path": str(record.features_path),
        }
    )
    if record.coords_path is not None:
        metadata["trident_coords_path"] = str(record.coords_path)

    return WSIBag(
        slide_id=record.slide_id,
        tile_features=features,
        coords=coords,
        label=record.label,
        attention=None,
        metadata=metadata,
    )
