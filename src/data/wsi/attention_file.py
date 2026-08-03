"""Readers and reducers for precomputed WSI attention artifacts.

The public WSI pipeline historically exported attention/importance values in
HDF5, NumPy, or Torch files.  This module keeps those artifacts readable while
removing any dependency on the deleted WSI forecaster/training code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


_DEFAULT_ATTENTION_KEYS = (
    "attention",
    "attn",
    "importance",
    "tile_importance",
    "score",
    "scores",
)
_DEFAULT_COORD_KEYS = ("coords", "coordinates")
_H5_SUFFIXES = {".h5", ".hdf5"}
_TORCH_SUFFIXES = {".pt", ".pth"}


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 WSI artifacts.")


def _to_tensor(value: Any, *, path: Path) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if np.isscalar(value):
        return torch.as_tensor(value)
    raise TypeError(
        f"unsupported object in {path}: {type(value).__name__}; expected a tensor/array."
    )


def _load_mapping_or_array(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in _TORCH_SUFFIXES:
        # Existing EAF artifacts may contain ordinary dictionaries and metadata.
        # Only load trusted local files produced by the preprocessing pipeline.
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    raise ValueError(
        f"unsupported file suffix for {path}; expected .h5, .hdf5, .pt, .pth, .npy, or .npz."
    )


def _select_mapping_value(
    mapping: Mapping[str, Any],
    *,
    path: Path,
    key: str | None,
    fallback_keys: tuple[str, ...],
) -> torch.Tensor:
    if key is not None:
        if key not in mapping:
            raise KeyError(f"key '{key}' not found in {path}.")
        return _to_tensor(mapping[key], path=path)

    for candidate in fallback_keys:
        if candidate in mapping:
            try:
                return _to_tensor(mapping[candidate], path=path)
            except TypeError:
                pass

    tensor_candidates: list[tuple[str, torch.Tensor]] = []
    for candidate, value in mapping.items():
        try:
            tensor_candidates.append((candidate, _to_tensor(value, path=path)))
        except TypeError:
            continue

    if len(tensor_candidates) == 1:
        return tensor_candidates[0][1]
    if not tensor_candidates:
        raise ValueError(f"no tensor/array found in {path}; tried keys {fallback_keys}.")
    raise ValueError(
        f"multiple tensor/array candidates found in {path}: "
        f"{[name for name, _ in tensor_candidates]}; pass an explicit key."
    )


def _read_h5_tensor(
    path: Path,
    *,
    key: str | None,
    fallback_keys: tuple[str, ...],
) -> torch.Tensor:
    _require_h5py()
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        if key is not None:
            if key not in handle:
                raise KeyError(f"key '{key}' not found in {path}.")
            return torch.as_tensor(handle[key][...])

        for candidate in fallback_keys:
            if candidate in handle and hasattr(handle[candidate], "shape"):
                return torch.as_tensor(handle[candidate][...])

        datasets: list[str] = []

        def visitor(name: str, obj: Any) -> None:
            if hasattr(obj, "shape"):
                datasets.append(name)

        handle.visititems(visitor)
        if len(datasets) == 1:
            return torch.as_tensor(handle[datasets[0]][...])
        if not datasets:
            raise ValueError(f"no dataset found in {path}.")
        raise ValueError(
            f"multiple dataset candidates found in {path}: {datasets}; pass an explicit key."
        )


def read_artifact_tensor(
    path: str | Path,
    *,
    key: str | None = None,
    fallback_keys: tuple[str, ...] = _DEFAULT_ATTENTION_KEYS,
) -> torch.Tensor:
    """Read one tensor from a supported attention/coordinate artifact."""

    artifact_path = Path(path)
    if artifact_path.suffix.lower() in _H5_SUFFIXES:
        tensor = _read_h5_tensor(
            artifact_path,
            key=key,
            fallback_keys=fallback_keys,
        )
    else:
        raw = _load_mapping_or_array(artifact_path)
        if isinstance(raw, Mapping):
            tensor = _select_mapping_value(
                raw,
                path=artifact_path,
                key=key,
                fallback_keys=fallback_keys,
            )
        else:
            tensor = _to_tensor(raw, path=artifact_path)

    if tensor.numel() == 0:
        raise ValueError(f"artifact is empty: {artifact_path}")
    if not torch.isfinite(tensor.to(torch.float32)).all():
        raise ValueError(f"artifact contains NaN or Inf: {artifact_path}")
    return tensor


def read_attention_tensor(path: str | Path, *, key: str | None = None) -> torch.Tensor:
    """Read an attention tensor without assuming a particular number of axes."""

    return read_artifact_tensor(path, key=key, fallback_keys=_DEFAULT_ATTENTION_KEYS).to(
        torch.float32
    )


def read_attention_coords(path: str | Path, *, key: str | None = None) -> torch.Tensor:
    """Read tile coordinates from a supported artifact."""

    coords = read_artifact_tensor(path, key=key, fallback_keys=_DEFAULT_COORD_KEYS)
    if coords.ndim != 2 or coords.shape[1] not in (2, 4):
        raise ValueError(
            "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
            f"got {tuple(coords.shape)}."
        )
    return coords.to(torch.long)


def reduce_attention_tensor(
    attention: torch.Tensor,
    *,
    n_tiles: int,
    tile_axis: int | None = None,
    tile_slice_start: int | None = None,
    reduction: str = "mean",
    selections: Mapping[int, int] | None = None,
) -> torch.Tensor:
    """Reduce a native attention tensor to one score per tile.

    Non-tile axes can represent layers, heads, queries, or other model-specific
    dimensions. ``selections`` fixes selected axes before all remaining
    non-tile axes are reduced. ``tile_slice_start`` explicitly removes CLS or
    other special tokens from a longer token axis. The function deliberately
    refuses ambiguous axes instead of silently choosing the wrong semantics.
    """

    tensor = torch.as_tensor(attention).detach().cpu().to(torch.float32)
    if tensor.ndim == 0:
        raise ValueError("attention tensor must have at least one dimension.")
    if n_tiles <= 0:
        raise ValueError("n_tiles must be positive.")

    if tile_axis is None:
        if tile_slice_start is not None:
            raise ValueError("tile_slice_start requires an explicit tile_axis.")
        candidates = [axis for axis, size in enumerate(tensor.shape) if size == n_tiles]
        if len(candidates) != 1:
            raise ValueError(
                "could not infer a unique tile axis from attention shape "
                f"{tuple(tensor.shape)} and n_tiles={n_tiles}; candidates={candidates}. "
                "Pass tile_axis explicitly."
            )
        resolved_tile_axis = candidates[0]
    else:
        resolved_tile_axis = tile_axis % tensor.ndim
        axis_size = tensor.shape[resolved_tile_axis]
        if tile_slice_start is None and axis_size != n_tiles:
            raise ValueError(
                f"tile_axis={tile_axis} has length {axis_size}, expected n_tiles={n_tiles}. "
                "For token axes containing CLS/special tokens, pass tile_slice_start explicitly."
            )
        if tile_slice_start is not None:
            start = tile_slice_start if tile_slice_start >= 0 else axis_size + tile_slice_start
            if start < 0 or start + n_tiles > axis_size:
                raise ValueError(
                    f"tile slice [{start}:{start + n_tiles}] is outside axis length {axis_size}."
                )
            index = [slice(None)] * tensor.ndim
            index[resolved_tile_axis] = slice(start, start + n_tiles)
            tensor = tensor[tuple(index)]

    normalized_selections: dict[int, int] = {}
    for axis, index in (selections or {}).items():
        normalized_axis = axis % tensor.ndim
        if normalized_axis == resolved_tile_axis:
            raise ValueError("the tile axis cannot be selected to a single index.")
        if normalized_axis in normalized_selections:
            raise ValueError(f"axis {axis} was selected more than once.")
        normalized_selections[normalized_axis] = index

    for axis in sorted(normalized_selections, reverse=True):
        size = tensor.shape[axis]
        index = normalized_selections[axis]
        if not -size <= index < size:
            raise IndexError(f"selection {axis}={index} is out of range for size {size}.")
        tensor = tensor.select(axis, index)
        if axis < resolved_tile_axis:
            resolved_tile_axis -= 1

    tensor = tensor.movedim(resolved_tile_axis, 0)
    if tensor.ndim == 1:
        values = tensor
    else:
        reduce_dims = tuple(range(1, tensor.ndim))
        if reduction == "mean":
            values = tensor.mean(dim=reduce_dims)
        elif reduction == "sum":
            values = tensor.sum(dim=reduce_dims)
        elif reduction == "max":
            values = tensor.amax(dim=reduce_dims)
        elif reduction == "l2":
            values = tensor.square().sum(dim=reduce_dims).sqrt()
        else:
            raise ValueError("reduction must be one of: mean, sum, max, l2.")

    if values.shape != (n_tiles,):
        raise RuntimeError(
            f"attention reduction produced shape {tuple(values.shape)}, expected {(n_tiles,)}."
        )
    if not torch.isfinite(values).all():
        raise ValueError("reduced attention contains NaN or Inf.")
    return values.contiguous()
