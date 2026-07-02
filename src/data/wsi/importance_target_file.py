"""Read precomputed tile-importance target artifacts for import into EAF.

Supports:

- HDF5 (``.h5``/``.hdf5``): a named dataset, or auto-detected from a small
  set of common key names (``attention``, ``importance``, ``tile_importance``,
  ``score``, ``scores``).
- ``.pt``/``.pth``/``.npy``/``.npz``: same key-detection convention.

This module intentionally does not import TRIDENT: it consumes whatever
target artifact a pipeline (TRIDENT-based or otherwise) produced, one file
per slide, mirroring ``src.data.wsi.trident`` and
``src.data.wsi.generic_features`` for tile features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


_DEFAULT_TARGET_KEYS = ("attention", "importance", "tile_importance", "score", "scores")
_DEFAULT_COORD_KEYS = ("coords", "coordinates")

_H5_SUFFIXES = {".h5", ".hdf5"}
_TORCH_SUFFIXES = {".pt", ".pth"}


def _load_trusted_torch_artifact(path: Path) -> Any:
    """Load a trusted local importer artifact that may contain metadata objects."""

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required to import HDF5 importance targets.")


def _select_h5_dataset_name(
    handle: Any,
    *,
    key: str | None,
    fallback_keys: tuple[str, ...],
    expected_ndim: int,
    path: Path,
) -> str:
    if key is not None:
        if key not in handle:
            raise KeyError(f"key '{key}' not found in {path}")
        dataset = handle[key]
        if getattr(dataset, "ndim", None) != expected_ndim:
            raise ValueError(
                f"key '{key}' in {path} must be {expected_ndim}D; "
                f"got {getattr(dataset, 'shape', None)}."
            )
        return key

    for name in fallback_keys:
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
            f"no {expected_ndim}D dataset found in {path}; tried {fallback_keys}."
        )
    raise ValueError(
        f"multiple {expected_ndim}D dataset candidates found in {path}: "
        f"{candidates}; pass an explicit key."
    )


def _read_h5_array(
    path: Path,
    *,
    key: str | None,
    fallback_keys: tuple[str, ...],
    expected_ndim: int,
) -> torch.Tensor:
    _require_h5py()
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        name = _select_h5_dataset_name(
            handle,
            key=key,
            fallback_keys=fallback_keys,
            expected_ndim=expected_ndim,
            path=path,
        )
        array = handle[name][...]

    return torch.as_tensor(array)


def _to_tensor(value: Any, *, path: Path) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    raise TypeError(
        f"unsupported array object in {path}: {type(value).__name__}; "
        "expected torch.Tensor or numpy.ndarray."
    )


def _load_raw(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix in _TORCH_SUFFIXES:
        return _load_trusted_torch_artifact(path)
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    raise ValueError(
        f"unsupported file suffix for {path}; expected .h5, .hdf5, .pt, .pth, "
        ".npy, or .npz."
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

    names = [name for name, _ in candidates]
    raise ValueError(
        f"multiple {expected_ndim}D tensor/array candidates found in {path}: "
        f"{names}; pass an explicit key."
    )


def _read_generic_array(
    path: Path,
    *,
    key: str | None,
    fallback_keys: tuple[str, ...],
    expected_ndim: int,
) -> torch.Tensor:
    raw = _load_raw(path)

    if isinstance(raw, dict):
        return _select_from_mapping(
            raw,
            path=path,
            key=key,
            fallback_keys=fallback_keys,
            expected_ndim=expected_ndim,
        )

    tensor = _to_tensor(raw, path=path)
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"array in {path} must be {expected_ndim}D; got {tuple(tensor.shape)}."
        )
    return tensor


def read_wsi_importance_target_tensor(
    path: str | Path,
    *,
    key: str | None = None,
) -> torch.Tensor:
    """Read a 1D tile-importance target vector from an importer artifact."""

    target_path = Path(path)
    suffix = target_path.suffix.lower()

    if suffix in _H5_SUFFIXES:
        tensor = _read_h5_array(
            target_path, key=key, fallback_keys=_DEFAULT_TARGET_KEYS, expected_ndim=1
        )
    else:
        tensor = _read_generic_array(
            target_path, key=key, fallback_keys=_DEFAULT_TARGET_KEYS, expected_ndim=1
        )

    tensor = tensor.to(torch.float32)

    if tensor.shape[0] == 0:
        raise ValueError(f"importance target must contain at least one tile: {target_path}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"importance target contains NaN or Inf: {target_path}")
    if (tensor < 0).any():
        raise ValueError(f"importance target must be non-negative: {target_path}")

    return tensor


def read_wsi_importance_coords_tensor(
    path: str | Path,
    *,
    key: str | None = None,
) -> torch.Tensor:
    """Read tile coordinates from an importer artifact (h5/pt/pth/npy/npz)."""

    coords_path = Path(path)
    suffix = coords_path.suffix.lower()

    if suffix in _H5_SUFFIXES:
        tensor = _read_h5_array(
            coords_path, key=key, fallback_keys=_DEFAULT_COORD_KEYS, expected_ndim=2
        )
    else:
        tensor = _read_generic_array(
            coords_path, key=key, fallback_keys=_DEFAULT_COORD_KEYS, expected_ndim=2
        )

    if tensor.ndim != 2:
        raise ValueError(f"coords must be 2D; got {tuple(tensor.shape)}.")
    if tensor.shape[0] == 0:
        raise ValueError("coords must contain at least one tile.")
    if tensor.shape[1] not in (2, 4):
        raise ValueError(
            "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
            f"got {tuple(tensor.shape)}."
        )
    if not torch.isfinite(tensor.to(torch.float32)).all():
        raise ValueError(f"coords contain NaN or Inf: {coords_path}")

    return tensor.to(torch.long)
