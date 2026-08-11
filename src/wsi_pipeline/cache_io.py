"""HDF5 I/O for offline Tile-EAF and WSI-EAF teacher caches."""

from __future__ import annotations

import json
import os
import tempfile
import time
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


def open_h5_with_retry(path: Path, mode: str = "r", *, attempts: int = 6, base_delay: float = 0.2):
    """Open an HDF5 file, retrying briefly on transient "unable to lock file" errors.

    Observed in practice immediately after a DataLoader with worker processes tears
    down: reopening the just-closed, just-renamed cache file for its own post-write
    integrity check can race an OS-level advisory-lock release and fail with
    ``OSError: ... unable to lock file ... errno = 11``. The file itself is fine (a
    fresh open moments later succeeds); this only smooths over that race rather than
    treating a transient lock as a corrupt cache.
    """
    h5py, _ = _deps()
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            return h5py.File(path, mode)
        except OSError as exc:
            if "unable to lock file" not in str(exc).lower() or attempt == attempts - 1:
                raise
            last_exc = exc
            time.sleep(base_delay * (2**attempt))
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises


def _jsonable_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _mkstemp_beside(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    return Path(tmp_name)


class TileCacheWriter:
    """Append tile-teacher batches to one slide-level cache without keeping all tiles in RAM.

    Writes go to a sibling ``.tmp`` file; the final path only appears via an atomic
    ``os.replace`` when ``close()``/``__exit__`` runs *without* an exception, and only
    once the ``complete`` HDF5 attribute has been set. A crash, OOM, or Ctrl-C mid-slide
    therefore never leaves a corrupt or partial file at the real cache path — resume
    logic only ever sees either nothing or a fully-written, ``complete=True`` cache.
    """

    def __init__(
        self,
        path: str | Path,
        spec: TileCacheSpec,
        *,
        slide_id: str,
        case_id: str,
        compression: str | None = "lzf",
        expected_n: int | None = None,
    ) -> None:
        h5py, _ = _deps()
        self.path = Path(path)
        self._tmp_path = _mkstemp_beside(self.path)
        self._closed = False
        self.handle = h5py.File(self._tmp_path, "w")
        self.handle.attrs["schema_version"] = spec.schema_version
        self.handle.attrs["kind"] = "tile_eaf"
        self.handle.attrs["complete"] = False
        self.handle.attrs["slide_id"] = slide_id
        self.handle.attrs["case_id"] = case_id
        self.handle.attrs["spec_json"] = _jsonable_metadata(spec.metadata())
        self.compression = compression
        if spec.dtype not in {"float16", "float32"}:
            raise ValueError(f"Unsupported tile-cache dtype: {spec.dtype}")
        self.storage_dtype = spec.dtype
        self.expected_n = int(expected_n) if expected_n is not None else None
        if self.expected_n is not None and self.expected_n < 0:
            raise ValueError("expected_n must be non-negative")
        self._datasets: dict[str, Any] = {}
        self.count = 0

    def _append_array(self, name: str, value: Any, *, dtype: str | None = None) -> None:
        _, np = _deps()
        array = np.asarray(value, dtype=dtype)
        if array.ndim < 1:
            raise ValueError(f"{name} must have a batch dimension")
        if name not in self._datasets:
            initial = self.expected_n if self.expected_n is not None else 0
            shape = (initial,) + array.shape[1:]
            maxshape = shape if self.expected_n is not None else (None,) + array.shape[1:]
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
        start = self.count
        stop = start + array.shape[0]
        if self.expected_n is not None:
            if stop > self.expected_n:
                raise ValueError(
                    f"{name} would exceed expected_n={self.expected_n}: stop={stop}"
                )
        else:
            dataset.resize(stop, axis=0)
        dataset[start:stop] = array

    def append(
        self,
        *,
        coords: Any,
        final_attention: Any,
        tile_embeddings: Any,
    ) -> None:
        """Append one batch. Deliberately takes no ``early_tokens`` -- it is not part
        of the permanent cache (CACHE_SCHEMA_VERSION v2); EAF Tile training recomputes
        it online instead. See ``HookedViTTileTeacherAdapter.extract_early``."""
        _, np = _deps()
        arrays = {
            "coords": np.asarray(coords),
            "final_attention": np.asarray(final_attention),
            "tile_embeddings": np.asarray(tile_embeddings),
        }
        batch_sizes = {name: value.shape[0] for name, value in arrays.items()}
        if len(set(batch_sizes.values())) != 1:
            raise ValueError(f"Batch-size mismatch: {batch_sizes}")
        self._append_array("coords", arrays["coords"], dtype="int32")
        self._append_array(
            "final_attention", arrays["final_attention"], dtype=self.storage_dtype
        )
        self._append_array(
            "tile_embeddings", arrays["tile_embeddings"], dtype=self.storage_dtype
        )
        self.count += next(iter(batch_sizes.values()))
        self.handle.attrs["n_tiles"] = self.count

    def close(self, *, mark_complete: bool = True) -> None:
        """Finalize the cache. ``mark_complete=False`` discards the tmp file instead."""
        if self._closed:
            return
        self._closed = True
        if mark_complete:
            if self.expected_n is not None and self.count != self.expected_n:
                self.handle.close()
                self._tmp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Incomplete cache: wrote {self.count}, expected {self.expected_n}"
                )
            self.handle.attrs["complete"] = True
        self.handle.flush()
        self.handle.close()
        if mark_complete:
            os.replace(self._tmp_path, self.path)
        else:
            self._tmp_path.unlink(missing_ok=True)

    def __enter__(self) -> "TileCacheWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Only publish the cache at its real path if the batch loop finished cleanly;
        # on any exception (including OOM) the tmp file is dropped and `self.path`
        # never appears, so resume logic never mistakes a partial run for a valid cache.
        self.close(mark_complete=exc_type is None)


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
    coords = np.asarray(coords, dtype="int32")
    tile_embeddings = np.asarray(tile_embeddings, dtype="float16")
    tile_scores = np.asarray(tile_scores, dtype="float16")
    wsi_embedding = np.asarray(wsi_embedding, dtype="float16")
    n = coords.shape[0]
    if tile_embeddings.shape[0] != n or tile_scores.shape[0] != n:
        raise ValueError(
            "WSI cache first dimension must match for coords/tile_embeddings/tile_scores"
        )
    tmp_path = _mkstemp_beside(path)
    try:
        with h5py.File(tmp_path, "w") as handle:
            handle.attrs["schema_version"] = spec.schema_version
            handle.attrs["kind"] = "wsi_eaf"
            handle.attrs["complete"] = False
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
            handle.attrs.modify("complete", True)
            handle.flush()
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def validate_cache(path: str | Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    path = Path(path)
    with open_h5_with_retry(path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"Cache is not marked complete (partial/interrupted write): {path}")
        kind = str(handle.attrs.get("kind", ""))
        if expected_kind and kind != expected_kind:
            raise ValueError(f"Expected {expected_kind}, found {kind}")
        if kind == "tile_eaf":
            required = {"coords", "final_attention", "tile_embeddings"}
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


def n_coords_in_registry(coords_path: str | Path) -> int:
    """Row count of a TRIDENT ``*_patches.h5`` coordinate file (``coords`` dataset)."""
    with open_h5_with_retry(Path(coords_path), "r") as handle:
        return int(handle["coords"].shape[0])


def tile_cache_status(
    path: str | Path,
    *,
    coords_path: str | Path,
    spec: TileCacheSpec,
) -> dict[str, Any]:
    """Resume/integrity check: is the tile cache at ``path`` valid and complete?

    Returns ``{"ok": bool, "reason": str | None, ...}``. Never raises; a missing,
    corrupt, incomplete, wrong-encoder, wrong-schema, or tile-count-mismatched cache is
    reported as ``ok=False`` with a human-readable ``reason`` rather than an exception,
    so callers can uniformly decide "rebuild this slide" without a try/except per check.
    """
    path = Path(path)
    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    try:
        info = validate_cache(path, expected_kind="tile_eaf")
    except (OSError, ValueError, KeyError) as exc:
        return {"ok": False, "reason": f"invalid_or_corrupt: {exc}"}
    if info["spec"].get("cache_id") != spec.cache_id:
        return {
            "ok": False,
            "reason": (
                f"spec_mismatch: cache built with cache_id={info['spec'].get('cache_id')!r}, "
                f"expected {spec.cache_id!r} (encoder/layer/attention/dtype changed)"
            ),
        }
    try:
        expected_n = n_coords_in_registry(coords_path)
    except (OSError, KeyError) as exc:
        return {"ok": False, "reason": f"coords_unreadable: {exc}"}
    if info["n_tiles"] != expected_n:
        return {
            "ok": False,
            "reason": f"tile_count_mismatch: cache has {info['n_tiles']}, coords has {expected_n}",
        }
    return {"ok": True, "reason": None, "n_tiles": info["n_tiles"], "spec": info["spec"]}
