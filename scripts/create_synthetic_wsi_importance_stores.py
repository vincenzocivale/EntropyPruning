#!/usr/bin/env python
"""Create synthetic early/late/importance-target HDF5 WSI feature stores.

This produces three paired stores in one deterministic pass:

- ``--early-output``: early-layer-style tile features (e.g. a cheap,
  low-dimensional encoder stage), used as the forecaster's *selection* input.
- ``--late-output``: late-layer-style tile features (e.g. an expensive,
  high-dimensional encoder stage), used as the *materialize* store when
  building a pruned feature store.
- ``--importance-output``: a scalar tile-importance target correlated with
  the *late* features, ``importance_i = softmax(w^T late_feature_i +
  noise_i)``, written as an EAF target store (``tile_features`` set to the
  target reshaped to ``[n_tiles, 1]``, ``attention`` set to the target
  itself), consistent with ``scripts/import_wsi_importance_targets.py``.

All three stores are generated from a single draw of ``n_tiles``/coords per
slide, so they share identical slide ids, tile counts, and tile coordinates
by construction. Generating them from three independent
``create_synthetic_wsi_feature_store.py`` calls would not guarantee this:
different ``--feature-dim`` values consume the shared RNG differently and
desynchronize the per-slide tile counts across runs. This script exists
purely for tests and smoke runs; real experiments should use TRIDENT-derived
or otherwise real feature stores (see ``docs/wsi_tile_importance_forecasting.md``).
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

from src.data.wsi import H5WSIFeatureStore, WSIBag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paired synthetic early/late/importance-target HDF5 WSI "
            "feature stores for tests and smoke runs."
        )
    )

    parser.add_argument("--early-output", type=Path, required=True)
    parser.add_argument("--late-output", type=Path, required=True)
    parser.add_argument("--importance-output", type=Path, required=True)

    parser.add_argument("--n-slides", type=int, default=24)
    parser.add_argument("--early-feature-dim", type=int, default=32)
    parser.add_argument("--late-feature-dim", type=int, default=128)
    parser.add_argument("--min-tiles", type=int, default=32)
    parser.add_argument("--max-tiles", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help=(
            "Std of Gaussian noise added to the late-feature projection "
            "before the softmax that defines the importance target."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_slides <= 0:
        raise ValueError("--n-slides must be positive.")
    if args.early_feature_dim <= 0:
        raise ValueError("--early-feature-dim must be positive.")
    if args.late_feature_dim <= 0:
        raise ValueError("--late-feature-dim must be positive.")
    if args.min_tiles <= 0:
        raise ValueError("--min-tiles must be positive.")
    if args.max_tiles < args.min_tiles:
        raise ValueError("--max-tiles must be >= --min-tiles.")
    if args.n_classes <= 0:
        raise ValueError("--n-classes must be positive.")
    if args.noise_std < 0:
        raise ValueError("--noise-std must be non-negative.")

    outputs = [args.early_output, args.late_output, args.importance_output]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError(
            "--early-output, --late-output, and --importance-output must be "
            "distinct paths."
        )

    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output file(s) already exist; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )


def _make_coords(n_tiles: int) -> torch.Tensor:
    side = int(torch.ceil(torch.sqrt(torch.tensor(float(n_tiles)))).item())
    ys = torch.arange(n_tiles, dtype=torch.long) // side
    xs = torch.arange(n_tiles, dtype=torch.long) % side
    return torch.stack([xs, ys], dim=1)


def main() -> int:
    args = parse_args()

    for path in (args.early_output, args.late_output, args.importance_output):
        if path.exists() and args.overwrite:
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    # Fixed projection direction correlating the importance target with the
    # late-layer features: importance_i = softmax(w^T late_feature_i + noise_i).
    importance_direction = torch.linspace(-1.0, 1.0, args.late_feature_dim)

    early_store = H5WSIFeatureStore(args.early_output)
    late_store = H5WSIFeatureStore(args.late_output)
    importance_store = H5WSIFeatureStore(args.importance_output)

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
        slide_id = f"synthetic_slide_{slide_index:05d}"
        label = slide_index % args.n_classes
        coords = _make_coords(n_tiles)

        early_features = torch.randn(n_tiles, args.early_feature_dim, generator=generator)
        late_features = torch.randn(n_tiles, args.late_feature_dim, generator=generator)
        noise = torch.randn(n_tiles, generator=generator) * args.noise_std

        logits = late_features @ importance_direction + noise
        importance = torch.softmax(logits, dim=0)

        early_store.write(
            WSIBag(
                slide_id=slide_id,
                tile_features=early_features,
                coords=coords,
                label=label,
                metadata={"source": "synthetic_early"},
            )
        )
        late_store.write(
            WSIBag(
                slide_id=slide_id,
                tile_features=late_features,
                coords=coords,
                label=label,
                metadata={"source": "synthetic_late"},
            )
        )
        importance_store.write(
            WSIBag(
                slide_id=slide_id,
                tile_features=importance.unsqueeze(1),
                coords=coords,
                label=label,
                attention=importance,
                metadata={
                    "target_source": "synthetic_importance_from_late",
                    "target_type": "tile_importance",
                    "importance_rule": "softmax(late_feature @ w + noise)",
                    "noise_std": args.noise_std,
                },
            )
        )

        n_tiles_total += n_tiles
        min_tiles_seen = n_tiles if min_tiles_seen is None else min(min_tiles_seen, n_tiles)
        max_tiles_seen = n_tiles if max_tiles_seen is None else max(max_tiles_seen, n_tiles)

    summary = {
        "early_output": str(args.early_output),
        "late_output": str(args.late_output),
        "importance_output": str(args.importance_output),
        "n_slides": args.n_slides,
        "early_feature_dim": args.early_feature_dim,
        "late_feature_dim": args.late_feature_dim,
        "min_tiles": min_tiles_seen,
        "max_tiles": max_tiles_seen,
        "n_tiles_total": n_tiles_total,
        "n_classes": args.n_classes,
        "noise_std": args.noise_std,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
