from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .utils import atomic_output_path


@dataclass(frozen=True)
class TileFeatureRecord:
    slide_id: str
    coords: np.ndarray
    embeddings: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WSIOutputRecord:
    slide_id: str
    slide_embedding: np.ndarray
    coords: np.ndarray | None = None
    attention: Mapping[str, np.ndarray] = field(default_factory=dict)
    auxiliary: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _storage_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported storage dtype: {name}")


def _write_attrs(group: h5py.Group, metadata: Mapping[str, Any]) -> None:
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, bytes, int, float, bool, np.number)):
            group.attrs[key] = value
        else:
            group.attrs[key] = str(value)


def write_tile_feature_record(
    path: Path,
    record: TileFeatureRecord,
    *,
    storage_dtype: str = "float16",
    compression: str | None = "lzf",
) -> None:
    coords = np.asarray(record.coords)
    if coords.ndim != 2 or coords.shape[1] not in (2, 4):
        raise ValueError(f"coords must have shape [N,2] or [N,4], got {coords.shape}")
    n_tiles = coords.shape[0]
    dtype = _storage_dtype(storage_dtype)
    for name, values in record.embeddings.items():
        arr = np.asarray(values)
        if arr.ndim != 2 or arr.shape[0] != n_tiles:
            raise ValueError(f"Embedding {name!r} must have shape [N,D], got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Embedding {name!r} contains NaN/Inf")

    with atomic_output_path(Path(path)) as tmp:
        with h5py.File(tmp, "w") as handle:
            handle.attrs["schema"] = "eaf.wsi.tile_features.v2"
            handle.attrs["complete"] = False
            handle.attrs["slide_id"] = record.slide_id
            handle.attrs["n_tiles"] = n_tiles
            handle.attrs["storage_dtype"] = storage_dtype
            _write_attrs(handle, record.metadata)
            handle.create_dataset("coords", data=coords.astype(np.int32, copy=False), compression=compression)
            group = handle.create_group("embeddings")
            for name, values in record.embeddings.items():
                arr = np.asarray(values, dtype=dtype)
                group.create_dataset(
                    name,
                    data=arr,
                    chunks=(min(2048, n_tiles), arr.shape[1]),
                    compression=compression,
                    shuffle=True,
                )
            handle.attrs.modify("complete", True)
            handle.flush()


def read_tile_feature_record(path: Path, *, squeeze_singleton: bool = True) -> TileFeatureRecord:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        schema = str(handle.attrs.get("schema", "legacy"))
        slide_id = str(handle.attrs.get("slide_id", path.stem))
        metadata = {key: handle.attrs[key] for key in handle.attrs.keys()}
        if schema == "eaf.wsi.tile_features.v2":
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete feature file: {path}")
            coords = np.asarray(handle["coords"][:])
            embeddings = {name: np.asarray(handle["embeddings"][name][:]) for name in handle["embeddings"]}
        else:
            if "features" not in handle:
                raise KeyError(f"Legacy feature file has no 'features': {path}")
            coords = np.asarray(handle["coords"][:]) if "coords" in handle else None
            if coords is None:
                raise KeyError(f"Legacy feature file has no 'coords': {path}")
            values = np.asarray(handle["features"][:])
            if squeeze_singleton and values.ndim == 3 and values.shape[1] == 1:
                values = values[:, 0, :]
            embeddings = {"final": values}
    return TileFeatureRecord(slide_id=slide_id, coords=coords, embeddings=embeddings, metadata=metadata)


def write_wsi_output_record(
    path: Path,
    record: WSIOutputRecord,
    *,
    storage_dtype: str = "float16",
    compression: str | None = "lzf",
) -> None:
    dtype = _storage_dtype(storage_dtype)
    slide_embedding = np.asarray(record.slide_embedding)
    if slide_embedding.ndim not in (1, 2):
        raise ValueError(f"slide_embedding must be 1D or 2D, got {slide_embedding.shape}")
    with atomic_output_path(Path(path)) as tmp:
        with h5py.File(tmp, "w") as handle:
            handle.attrs["schema"] = "eaf.wsi.fm_output.v1"
            handle.attrs["complete"] = False
            handle.attrs["slide_id"] = record.slide_id
            handle.attrs["storage_dtype"] = storage_dtype
            _write_attrs(handle, record.metadata)
            handle.create_dataset("slide_embedding", data=slide_embedding.astype(dtype, copy=False))
            if record.coords is not None:
                handle.create_dataset("coords", data=np.asarray(record.coords, dtype=np.int32), compression=compression)
            attn_group = handle.create_group("attention")
            for name, values in record.attention.items():
                arr = np.asarray(values)
                if not np.isfinite(arr).all():
                    raise ValueError(f"Attention {name!r} contains NaN/Inf")
                chunks = True if arr.ndim > 0 else None
                attn_group.create_dataset(
                    name,
                    data=arr.astype(dtype, copy=False),
                    compression=compression if arr.ndim > 0 else None,
                    chunks=chunks,
                    shuffle=arr.ndim > 0,
                )
            auxiliary_group = handle.create_group("auxiliary")
            for name, values in record.auxiliary.items():
                arr = np.asarray(values)
                if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                    raise ValueError(f"Auxiliary array {name!r} contains NaN/Inf")
                chunks = True if arr.ndim > 0 else None
                auxiliary_group.create_dataset(
                    name,
                    data=arr,
                    compression=compression if arr.ndim > 0 else None,
                    chunks=chunks,
                    shuffle=arr.ndim > 0,
                )
            handle.attrs.modify("complete", True)
            handle.flush()


def read_wsi_output_record(path: Path) -> WSIOutputRecord:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise RuntimeError(f"Incomplete WSI output file: {path}")
        slide_id = str(handle.attrs.get("slide_id", path.stem))
        embedding = np.asarray(handle["slide_embedding"][:])
        coords = np.asarray(handle["coords"][:]) if "coords" in handle else None
        attention = {name: np.asarray(handle["attention"][name][:]) for name in handle["attention"]}
        auxiliary = (
            {name: np.asarray(handle["auxiliary"][name][:]) for name in handle["auxiliary"]}
            if "auxiliary" in handle
            else {}
        )
        metadata = {key: handle.attrs[key] for key in handle.attrs.keys()}
    return WSIOutputRecord(slide_id, embedding, coords, attention, auxiliary, metadata)


def output_is_complete(path: Path, expected_schema: str) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return bool(handle.attrs.get("complete", False)) and str(handle.attrs.get("schema")) == expected_schema
    except OSError:
        return False
