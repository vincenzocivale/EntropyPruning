"""HDF5 I/O for offline Tile-EAF and WSI-EAF teacher caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cache_contracts import TileCacheSpec, WSICacheSpec


def _deps():
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("cache I/O requires h5py and numpy") from exc
    return h5py, np


def _jsonable_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


class TileCacheWriter:
    """Append tile-teacher batches to one slide-level cache without keeping all tiles in RAM."""

    def __init__(
        self,
        path: str | Path,
        spec: TileCacheSpec,
        *,
        slide_id: str,
        case_id: str,
        compression: str | None = "lzf",
    ) -> None:
        h5py, _ = _deps()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, "w")
        self.handle.attrs["schema_version"] = spec.schema_version
        self.handle.attrs["kind"] = "tile_eaf"
        self.handle.attrs["slide_id"] = slide_id
        self.handle.attrs["case_id"] = case_id
        self.handle.attrs["spec_json"] = _jsonable_metadata(spec.metadata())
        self.compression = compression
        self._datasets: dict[str, Any] = {}
        self.count = 0

    def _append_array(self, name: str, value: Any, *, dtype: str | None = None) -> None:
        _, np = _deps()
        array = np.asarray(value, dtype=dtype)
        if array.ndim < 1:
            raise ValueError(f"{name} must have a batch dimension")
        if name not in self._datasets:
            shape = (0,) + array.shape[1:]
            maxshape = (None,) + array.shape[1:]
            self._datasets[name] = self.handle.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                chunks=True,
                compression=self.compression,
                dtype=array.dtype,
            )
        dataset = self._datasets[name]
        if dataset.shape[1:] != array.shape[1:]:
            raise ValueError(
                f"Shape changed for {name}: {dataset.shape[1:]} vs {array.shape[1:]}"
            )
        start = dataset.shape[0]
        dataset.resize(start + array.shape[0], axis=0)
        dataset[start:] = array

    def append(
        self,
        *,
        coords: Any,
        early_tokens: Any,
        final_attention: Any,
        tile_embeddings: Any,
    ) -> None:
        _, np = _deps()
        arrays = {
            "coords": np.asarray(coords),
            "early_tokens": np.asarray(early_tokens),
            "final_attention": np.asarray(final_attention),
            "tile_embeddings": np.asarray(tile_embeddings),
        }
        batch_sizes = {name: value.shape[0] for name, value in arrays.items()}
        if len(set(batch_sizes.values())) != 1:
            raise ValueError(f"Batch-size mismatch: {batch_sizes}")
        self._append_array("coords", arrays["coords"], dtype="int32")
        self._append_array("early_tokens", arrays["early_tokens"], dtype="float16")
        self._append_array("final_attention", arrays["final_attention"], dtype="float16")
        self._append_array("tile_embeddings", arrays["tile_embeddings"], dtype="float16")
        self.count += next(iter(batch_sizes.values()))
        self.handle.attrs["n_tiles"] = self.count
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "TileCacheWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_wsi_cache(
    path: str | Path,
    spec: WSICacheSpec,
    *,
    slide_id: str,
    case_id: str,
    coords: Any,
    tile_embeddings: Any,
    tile_scores: Any,
    wsi_embedding: Any,
    raw_attention: Any | None = None,
    compression: str | None = "lzf",
) -> Path:
    h5py, np = _deps()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords, dtype="int32")
    tile_embeddings = np.asarray(tile_embeddings, dtype="float16")
    tile_scores = np.asarray(tile_scores, dtype="float16")
    wsi_embedding = np.asarray(wsi_embedding, dtype="float16")
    n = coords.shape[0]
    if tile_embeddings.shape[0] != n or tile_scores.shape[0] != n:
        raise ValueError(
            "WSI cache first dimension must match for coords/tile_embeddings/tile_scores"
        )
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = spec.schema_version
        handle.attrs["kind"] = "wsi_eaf"
        handle.attrs["slide_id"] = slide_id
        handle.attrs["case_id"] = case_id
        handle.attrs["n_tiles"] = n
        handle.attrs["spec_json"] = _jsonable_metadata(spec.metadata())
        handle.create_dataset("coords", data=coords, compression=compression)
        handle.create_dataset("tile_embeddings", data=tile_embeddings, compression=compression)
        handle.create_dataset("tile_scores", data=tile_scores, compression=compression)
        handle.create_dataset("wsi_embedding", data=wsi_embedding, compression=compression)
        if raw_attention is not None:
            if not spec.store_raw_attention:
                raise ValueError("raw_attention supplied but store_raw_attention=False")
            handle.create_dataset(
                "raw_attention",
                data=np.asarray(raw_attention, dtype="float16"),
                compression=compression,
            )
    return path


def validate_cache(path: str | Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    h5py, _ = _deps()
    path = Path(path)
    with h5py.File(path, "r") as handle:
        kind = str(handle.attrs.get("kind", ""))
        if expected_kind and kind != expected_kind:
            raise ValueError(f"Expected {expected_kind}, found {kind}")
        if kind == "tile_eaf":
            required = {"coords", "early_tokens", "final_attention", "tile_embeddings"}
        elif kind == "wsi_eaf":
            required = {"coords", "tile_embeddings", "tile_scores", "wsi_embedding"}
        else:
            raise ValueError(f"Unknown cache kind: {kind}")
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"Missing cache datasets: {sorted(missing)}")
        n = handle["coords"].shape[0]
        for name in required - {"wsi_embedding"}:
            if handle[name].shape[0] != n:
                raise ValueError(f"First-dimension mismatch for {name}")
        return {
            "kind": kind,
            "slide_id": str(handle.attrs.get("slide_id", "")),
            "case_id": str(handle.attrs.get("case_id", "")),
            "n_tiles": n,
            "datasets": {name: tuple(handle[name].shape) for name in handle.keys()},
            "spec": json.loads(str(handle.attrs["spec_json"])),
        }
