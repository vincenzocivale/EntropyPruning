#!/usr/bin/env python
"""Create a synthetic HDF5 WSI feature store for smoke tests and demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import H5WSIFeatureStore, WSIBag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic HDF5 WSI feature store."
    )

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-slides", type=int, default=24)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--min-tiles", type=int, default=32)
    parser.add_argument("--max-tiles", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    args = parser.parse_args()

    if args.n_slides <= 0:
        raise ValueError("--n-slides must be positive.")
    if args.feature_dim <= 0:
        raise ValueError("--feature-dim must be positive.")
    if args.min_tiles <= 0:
        raise ValueError("--min-tiles must be positive.")
    if args.max_tiles < args.min_tiles:
        raise ValueError("--max-tiles must be >= --min-tiles.")
    if args.n_classes <= 0:
        raise ValueError("--n-classes must be positive.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")

    return args


def _make_synthetic_bag(
    *,
    slide_index: int,
    n_tiles: int,
    feature_dim: int,
    n_classes: int,
    target_direction: torch.Tensor,
    generator: torch.Generator,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim, generator=generator)

    # Add a small class-dependent shift so labels are not completely arbitrary.
    label = slide_index % n_classes
    class_shift = (label - (n_classes - 1) / 2.0) * 0.15
    tile_features[:, 0] += class_shift

    # Synthetic attention target: softmax over a fixed linear projection.
    # This makes the target learnable from tile_features.
    logits = tile_features @ target_direction
    attention = torch.softmax(logits, dim=0)

    # Synthetic grid-like coordinates.
    side = int(torch.ceil(torch.sqrt(torch.tensor(float(n_tiles)))).item())
    ys = torch.arange(n_tiles, dtype=torch.long) // side
    xs = torch.arange(n_tiles, dtype=torch.long) % side
    coords = torch.stack([xs, ys], dim=1)

    return WSIBag(
        slide_id=f"synthetic_slide_{slide_index:05d}",
        tile_features=tile_features,
        coords=coords,
        label=label,
        attention=attention,
        metadata={
            "source": "synthetic",
            "attention_rule": "softmax(tile_features @ target_direction)",
        },
    )


def main() -> int:
    args = parse_args()

    if args.output.exists() and args.overwrite:
        args.output.unlink()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    target_direction = torch.linspace(-1.0, 1.0, args.feature_dim)

    store = H5WSIFeatureStore(args.output)

    n_tiles_total = 0
    min_tiles_seen = None
    max_tiles_seen = None

    for slide_index in range(args.n_slides):
        n_tiles = int(
            torch.randint(
                low=args.min_tiles,
                high=args.max_tiles + 1,
                size=(1,),
                generator=generator,
            ).item()
        )
        bag = _make_synthetic_bag(
            slide_index=slide_index,
            n_tiles=n_tiles,
            feature_dim=args.feature_dim,
            n_classes=args.n_classes,
            target_direction=target_direction,
            generator=generator,
        )
        store.write(bag)

        n_tiles_total += n_tiles
        min_tiles_seen = n_tiles if min_tiles_seen is None else min(min_tiles_seen, n_tiles)
        max_tiles_seen = n_tiles if max_tiles_seen is None else max(max_tiles_seen, n_tiles)

    summary = {
        "output": str(args.output),
        "n_slides": args.n_slides,
        "feature_dim": args.feature_dim,
        "min_tiles": min_tiles_seen,
        "max_tiles": max_tiles_seen,
        "n_tiles_total": n_tiles_total,
        "n_classes": args.n_classes,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
