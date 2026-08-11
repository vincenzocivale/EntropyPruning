"""Validated indexes joining canonical slide manifests to compact tile caches."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from src.data.wsi.manifest import read_manifest

from .cache_io import validate_cache


INDEX_FIELDS = ("slide_id", "cache_path", "n_tiles", "cache_id")


def build_tile_cache_index(
    slides_path: str | Path,
    cache_roots: Iterable[str | Path],
    output_path: str | Path,
    *,
    expected_cache_id: str | None = None,
    data_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Create a strict one-to-one slide/cache index and verify coordinate order."""
    slides_path = Path(slides_path).expanduser().resolve()
    slides = read_manifest(slides_path)
    wanted = {row.slide_id: row for row in slides if row.coords_path}
    if not wanted:
        raise ValueError(f"No slides with coords_path in {slides_path}")

    candidates: dict[str, list[tuple[Path, dict]]] = {}
    seen_cache_paths: set[Path] = set()
    for root_value in cache_roots:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Cache root does not exist: {root}")
        for path in root.rglob("*.h5"):
            path = path.resolve()
            if path in seen_cache_paths:
                continue
            seen_cache_paths.add(path)
            try:
                info = validate_cache(path, expected_kind="tile_eaf")
            except (OSError, ValueError, KeyError):
                continue
            slide_id = info["slide_id"]
            cache_id = str(info["spec"].get("cache_id", ""))
            if slide_id not in wanted:
                continue
            if expected_cache_id is not None and cache_id != expected_cache_id:
                continue
            candidates.setdefault(slide_id, []).append((path, info))

    missing = sorted(set(wanted) - set(candidates))
    duplicates = sorted(slide_id for slide_id, values in candidates.items() if len(values) != 1)
    if missing or duplicates:
        raise RuntimeError(
            f"Cache index is not one-to-one: missing={len(missing)} {missing[:5]}, "
            f"duplicates={len(duplicates)} {duplicates[:5]}"
        )

    observed_cache_ids = {
        str(values[0][1]["spec"].get("cache_id", ""))
        for values in candidates.values()
    }
    if len(observed_cache_ids) != 1:
        raise RuntimeError(
            f"Cache index would mix cache identities: {sorted(observed_cache_ids)}"
        )

    rows: list[dict[str, str]] = []
    for slide_id in sorted(wanted):
        cache_path, info = candidates[slide_id][0]
        coords_path = Path(wanted[slide_id].coords_path).expanduser()
        if not coords_path.is_absolute():
            base = Path(data_root).expanduser().resolve() if data_root else slides_path.parent
            coords_path = (base / coords_path).resolve()
        with h5py.File(coords_path, "r") as coords_handle, h5py.File(cache_path, "r") as cache_handle:
            source_coords = np.asarray(coords_handle["coords"][:, :2])
            cached_coords = np.asarray(cache_handle["coords"][:, :2])
        if not np.array_equal(source_coords, cached_coords):
            raise RuntimeError(
                f"Coordinate order mismatch for {slide_id}: {coords_path} vs {cache_path}"
            )
        rows.append(
            {
                "slide_id": slide_id,
                "cache_path": str(cache_path),
                "n_tiles": str(info["n_tiles"]),
                "cache_id": str(info["spec"].get("cache_id", "")),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    return rows


def read_tile_cache_index(path: str | Path) -> dict[str, Path]:
    """Read an index, rejecting duplicate slide identifiers."""
    path = Path(path).expanduser().resolve()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, Path] = {}
    for row in rows:
        slide_id = row.get("slide_id", "")
        cache_path = Path(row.get("cache_path", "")).expanduser()
        if not cache_path.is_absolute():
            cache_path = (path.parent / cache_path).resolve()
        if not slide_id or slide_id in mapping:
            raise ValueError(f"Invalid or duplicate slide_id in {path}: {slide_id!r}")
        mapping[slide_id] = cache_path
    return mapping
