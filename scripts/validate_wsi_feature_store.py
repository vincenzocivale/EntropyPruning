#!/usr/bin/env python
"""Validate an HDF5 WSI feature store.

The script checks that bags stored in ``H5WSIFeatureStore`` are usable for
WSI-level tile attention forecasting before launching training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import H5WSIFeatureStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a WSI HDF5 feature store.")

    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--require-attention", action="store_true")
    parser.add_argument("--require-coords", action="store_true")
    parser.add_argument("--max-errors", type=int, default=50)

    args = parser.parse_args()

    if args.feature_dim is not None and args.feature_dim <= 0:
        raise ValueError("--feature-dim must be positive when provided.")
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be positive.")

    return args


def _validate_slide(
    store: H5WSIFeatureStore,
    slide_id: str,
    *,
    expected_feature_dim: int | None,
    require_attention: bool,
    require_coords: bool,
) -> tuple[list[str], dict[str, int | str | bool | None]]:
    errors: list[str] = []
    info: dict[str, int | str | bool | None] = {
        "slide_id": slide_id,
        "n_tiles": None,
        "feature_dim": None,
        "has_attention": None,
        "has_coords": None,
    }

    try:
        bag = store.read(slide_id)
    except Exception as exc:  # noqa: BLE001 - report validation failure, continue.
        return [f"failed to read slide: {type(exc).__name__}: {exc}"], info

    info["n_tiles"] = bag.n_tiles
    info["feature_dim"] = bag.feature_dim
    info["has_attention"] = bag.attention is not None
    info["has_coords"] = bag.coords is not None

    if bag.tile_features.ndim != 2:
        errors.append(f"tile_features must be 2D; got {tuple(bag.tile_features.shape)}")

    if not torch.is_floating_point(bag.tile_features):
        errors.append("tile_features must be floating point")

    if not torch.isfinite(bag.tile_features).all():
        errors.append("tile_features contain NaN or Inf")

    if expected_feature_dim is not None and bag.feature_dim != expected_feature_dim:
        errors.append(
            f"feature_dim mismatch: expected {expected_feature_dim}, got {bag.feature_dim}"
        )

    if require_coords and bag.coords is None:
        errors.append("coords are required but missing")

    if bag.coords is not None:
        if bag.coords.ndim != 2:
            errors.append(f"coords must be 2D; got {tuple(bag.coords.shape)}")
        elif bag.coords.shape[0] != bag.n_tiles:
            errors.append(
                f"coords length mismatch: expected {bag.n_tiles}, got {bag.coords.shape[0]}"
            )
        elif bag.coords.shape[1] not in (2, 4):
            errors.append(f"coords width must be 2 or 4; got {bag.coords.shape[1]}")

    if require_attention and bag.attention is None:
        errors.append("attention is required but missing")

    if bag.attention is not None:
        if bag.attention.ndim != 1:
            errors.append(f"attention must be 1D; got {tuple(bag.attention.shape)}")
        elif bag.attention.shape[0] != bag.n_tiles:
            errors.append(
                f"attention length mismatch: expected {bag.n_tiles}, got {bag.attention.shape[0]}"
            )

        if not torch.is_floating_point(bag.attention):
            errors.append("attention must be floating point")
        else:
            if not torch.isfinite(bag.attention).all():
                errors.append("attention contains NaN or Inf")
            if (bag.attention < 0).any():
                errors.append("attention contains negative values")
            if bag.attention.sum() <= 0:
                errors.append("attention mass must be positive")

    return errors, info


def main() -> int:
    args = parse_args()

    store = H5WSIFeatureStore(args.feature_store)
    slide_ids = store.slide_ids()

    errors: list[dict[str, str]] = []
    n_tiles_total = 0
    min_tiles = None
    max_tiles = None
    feature_dims: set[int] = set()
    n_with_attention = 0
    n_with_coords = 0

    if not slide_ids:
        errors.append({"slide_id": "<store>", "error": "feature store contains no slides"})

    for slide_id in slide_ids:
        slide_errors, info = _validate_slide(
            store,
            slide_id,
            expected_feature_dim=args.feature_dim,
            require_attention=args.require_attention,
            require_coords=args.require_coords,
        )

        n_tiles = info["n_tiles"]
        feature_dim = info["feature_dim"]

        if isinstance(n_tiles, int):
            n_tiles_total += n_tiles
            min_tiles = n_tiles if min_tiles is None else min(min_tiles, n_tiles)
            max_tiles = n_tiles if max_tiles is None else max(max_tiles, n_tiles)

        if isinstance(feature_dim, int):
            feature_dims.add(feature_dim)

        if info["has_attention"] is True:
            n_with_attention += 1
        if info["has_coords"] is True:
            n_with_coords += 1

        for error in slide_errors:
            if len(errors) < args.max_errors:
                errors.append({"slide_id": slide_id, "error": error})

    valid = len(errors) == 0

    summary = {
        "valid": valid,
        "feature_store": str(args.feature_store),
        "n_slides": len(slide_ids),
        "n_tiles_total": n_tiles_total,
        "min_tiles": min_tiles,
        "max_tiles": max_tiles,
        "feature_dims": sorted(feature_dims),
        "n_with_attention": n_with_attention,
        "n_with_coords": n_with_coords,
        "require_attention": args.require_attention,
        "require_coords": args.require_coords,
        "expected_feature_dim": args.feature_dim,
        "n_errors": len(errors),
        "errors": errors,
    }

    print(json.dumps(summary, indent=2), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
