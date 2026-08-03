"""Generic feature-file importer for WSI bags.

Supported feature and coordinate file formats:

- .pt / .pth
- .npy
- .npz

The module intentionally keeps the contract simple: one feature file per slide,
optionally one coordinate file per slide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.wsi.bag import WSIBag


_FEATURE_FALLBACK_KEYS = ("tile_features", "features", "feats", "embeddings", "x")
_COORD_FALLBACK_KEYS = ("coords", "coordinates", "xy")


@dataclass(frozen=True)
class GenericFeatureSlideRecord:
    """Manifest record for one generic WSI feature slide."""

    slide_id: str
    features_path: Path
    coords_path: Path | None = None
    label: int | float | None = None
    metadata: dict[str, Any] | None = None


def _to_tensor(value: Any, *, path: Path) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()

    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)

    raise TypeError(
        f"unsupported array object in {path}: {type(value).__name__}; "
        "expected torch.Tensor or numpy.ndarray."
    )


def _select_from_mapping(
    mapping: dict[str, Any],
    *,
    path: Path,
    key: str | None,
    fallback_keys: tuple[str, ...],
    expected_ndim: int,
) -> torch.Tensor:
    if key is not None:
        if key not in mapping:
            raise KeyError(f"key '{key}' not found in {path}.")
        tensor = _to_tensor(mapping[key], path=path)
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"key '{key}' in {path} must be {expected_ndim}D; "
                f"got {tuple(tensor.shape)}."
            )
        return tensor

    for fallback_key in fallback_keys:
        if fallback_key in mapping:
            tensor = _to_tensor(mapping[fallback_key], path=path)
            if tensor.ndim == expected_ndim:
                return tensor

    candidates: list[tuple[str, torch.Tensor]] = []
    for candidate_key, value in mapping.items():
        try:
            tensor = _to_tensor(value, path=path)
        except TypeError:
            continue
        if tensor.ndim == expected_ndim:
            candidates.append((candidate_key, tensor))

    if len(candidates) == 1:
        return candidates[0][1]

    if not candidates:
        raise ValueError(
            f"no {expected_ndim}D tensor/array found in {path}; "
            f"tried keys {fallback_keys}."
        )

    candidate_names = [name for name, _ in candidates]
    raise ValueError(
        f"multiple {expected_ndim}D tensor/array candidates found in {path}: "
        f"{candidate_names}; pass an explicit key."
    )


def _load_raw(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")

    if suffix == ".npy":
        return np.load(path, allow_pickle=False)

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    raise ValueError(
        f"unsupported file suffix for {path}; expected .pt, .pth, .npy, or .npz."
    )


def read_generic_feature_tensor(
    path: str | Path,
    *,
    key: str | None = None,
) -> torch.Tensor:
    """Read tile features from .pt/.pth/.npy/.npz."""

    feature_path = Path(path)
    raw = _load_raw(feature_path)

    if isinstance(raw, dict):
        tensor = _select_from_mapping(
            raw,
            path=feature_path,
            key=key,
            fallback_keys=_FEATURE_FALLBACK_KEYS,
            expected_ndim=2,
        )
    else:
        tensor = _to_tensor(raw, path=feature_path)

    if tensor.ndim != 2:
        raise ValueError(f"features must be 2D; got {tuple(tensor.shape)}.")
    if tensor.shape[0] == 0:
        raise ValueError("features must contain at least one tile.")
    if tensor.shape[1] == 0:
        raise ValueError("feature dimension must be positive.")

    if not torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float32)
    else:
        tensor = tensor.to(torch.float32)

    if not torch.isfinite(tensor).all():
        raise ValueError(f"features contain NaN or Inf: {feature_path}")

    return tensor


def read_generic_coords_tensor(
    path: str | Path,
    *,
    key: str | None = None,
) -> torch.Tensor:
    """Read coordinates from .pt/.pth/.npy/.npz."""

    coords_path = Path(path)
    raw = _load_raw(coords_path)

    if isinstance(raw, dict):
        tensor = _select_from_mapping(
            raw,
            path=coords_path,
            key=key,
            fallback_keys=_COORD_FALLBACK_KEYS,
            expected_ndim=2,
        )
    else:
        tensor = _to_tensor(raw, path=coords_path)

    if tensor.ndim != 2:
        raise ValueError(f"coords must be 2D; got {tuple(tensor.shape)}.")
    if tensor.shape[0] == 0:
        raise ValueError("coords must contain at least one tile.")
    if tensor.shape[1] not in {2, 4}:
        raise ValueError(
            "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
            f"got {tuple(tensor.shape)}."
        )

    if not torch.isfinite(tensor.to(torch.float32)).all():
        raise ValueError(f"coords contain NaN or Inf: {coords_path}")

    return tensor.to(torch.long)


def load_generic_feature_slide_record(
    record: GenericFeatureSlideRecord,
    *,
    feature_key: str | None = None,
    coords_key: str | None = None,
    expected_feature_dim: int | None = None,
) -> WSIBag:
    """Load one generic feature manifest record as a ``WSIBag``."""

    features = read_generic_feature_tensor(record.features_path, key=feature_key)

    if expected_feature_dim is not None and features.shape[1] != expected_feature_dim:
        raise ValueError(
            f"slide {record.slide_id} has feature_dim={features.shape[1]}, "
            f"expected {expected_feature_dim}."
        )

    coords = None
    if record.coords_path is not None:
        coords = read_generic_coords_tensor(record.coords_path, key=coords_key)
        if coords.shape[0] != features.shape[0]:
            raise ValueError(
                f"slide {record.slide_id} has {features.shape[0]} features but "
                f"{coords.shape[0]} coordinates."
            )

    metadata = dict(record.metadata) if record.metadata is not None else {}
    metadata.update(
        {
            "source": "generic_feature_file",
            "features_path": str(record.features_path),
        }
    )
    if record.coords_path is not None:
        metadata["coords_path"] = str(record.coords_path)

    return WSIBag(
        slide_id=record.slide_id,
        tile_features=features,
        coords=coords,
        label=record.label,
        attention=None,
        metadata=metadata,
    )
