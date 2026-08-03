#!/usr/bin/env python
"""Inspect an EAF WSI HDF5 feature store."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import H5WSIFeatureStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an EAF WSI HDF5 feature store.")

    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum example slide ids to include for each category.",
    )

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative.")


def _label_to_key(label: Any) -> str:
    if label is None:
        return "<missing>"

    if isinstance(label, torch.Tensor):
        if label.numel() == 1:
            return str(label.detach().cpu().item())
        return f"tensor_shape_{tuple(label.shape)}"

    return str(label)


def _shape_key(shape: torch.Size | tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in tuple(shape))


def _mean(values: list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def inspect_feature_store(
    feature_store: Path,
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    store = H5WSIFeatureStore(feature_store)
    slide_ids = store.slide_ids()

    if not slide_ids:
        return {
            "feature_store": str(feature_store),
            "valid": False,
            "reason": "feature store contains no slides",
            "n_slides": 0,
        }

    tile_counts: list[int] = []
    feature_dims: Counter[int] = Counter()
    coord_shapes: Counter[str] = Counter()
    attention_shapes: Counter[str] = Counter()
    label_distribution: Counter[str] = Counter()
    metadata_source_distribution: Counter[str] = Counter()

    slides_with_coords: list[str] = []
    slides_without_coords: list[str] = []
    slides_with_attention: list[str] = []
    slides_without_attention: list[str] = []
    slides_with_label: list[str] = []
    slides_without_label: list[str] = []
    slides_with_nonfinite_features: list[str] = []
    slides_with_nonfinite_attention: list[str] = []
    slides_with_negative_attention: list[str] = []
    slides_with_zero_attention_mass: list[str] = []

    for slide_id in slide_ids:
        bag = store.read(slide_id)

        tile_counts.append(bag.n_tiles)
        feature_dims[int(bag.tile_features.shape[1])] += 1

        if not torch.isfinite(bag.tile_features).all():
            slides_with_nonfinite_features.append(slide_id)

        if bag.coords is None:
            slides_without_coords.append(slide_id)
        else:
            slides_with_coords.append(slide_id)
            coord_shapes[_shape_key(tuple(bag.coords.shape[1:]))] += 1

        if bag.attention is None:
            slides_without_attention.append(slide_id)
        else:
            slides_with_attention.append(slide_id)
            attention_shapes[_shape_key(tuple(bag.attention.shape))] += 1

            if not torch.isfinite(bag.attention).all():
                slides_with_nonfinite_attention.append(slide_id)
            if (bag.attention < 0).any():
                slides_with_negative_attention.append(slide_id)
            if bag.attention.sum() <= 0:
                slides_with_zero_attention_mass.append(slide_id)

        if bag.label is None:
            slides_without_label.append(slide_id)
        else:
            slides_with_label.append(slide_id)
            label_distribution[_label_to_key(bag.label)] += 1

        source = "<missing>"
        if bag.metadata is not None:
            source = str(bag.metadata.get("source", "<missing>"))
        metadata_source_distribution[source] += 1

    n_slides = len(slide_ids)

    summary = {
        "feature_store": str(feature_store),
        "valid": True,
        "n_slides": n_slides,
        "tile_count": {
            "min": min(tile_counts),
            "mean": _mean(tile_counts),
            "median": _median(tile_counts),
            "max": max(tile_counts),
            "total": sum(tile_counts),
        },
        "feature_dims": dict(sorted(feature_dims.items())),
        "coords": {
            "n_with": len(slides_with_coords),
            "n_without": len(slides_without_coords),
            "fraction_with": _percent(len(slides_with_coords), n_slides),
            "shapes_excluding_tile_dim": dict(sorted(coord_shapes.items())),
            "examples_without": slides_without_coords[:max_examples],
        },
        "attention": {
            "n_with": len(slides_with_attention),
            "n_without": len(slides_without_attention),
            "fraction_with": _percent(len(slides_with_attention), n_slides),
            "shapes": dict(sorted(attention_shapes.items())),
            "examples_without": slides_without_attention[:max_examples],
        },
        "labels": {
            "n_with": len(slides_with_label),
            "n_without": len(slides_without_label),
            "fraction_with": _percent(len(slides_with_label), n_slides),
            "distribution": dict(sorted(label_distribution.items())),
            "examples_without": slides_without_label[:max_examples],
        },
        "metadata": {
            "source_distribution": dict(sorted(metadata_source_distribution.items())),
        },
        "quality_flags": {
            "n_with_nonfinite_features": len(slides_with_nonfinite_features),
            "n_with_nonfinite_attention": len(slides_with_nonfinite_attention),
            "n_with_negative_attention": len(slides_with_negative_attention),
            "n_with_zero_attention_mass": len(slides_with_zero_attention_mass),
            "examples_with_nonfinite_features": slides_with_nonfinite_features[:max_examples],
            "examples_with_nonfinite_attention": slides_with_nonfinite_attention[:max_examples],
            "examples_with_negative_attention": slides_with_negative_attention[:max_examples],
            "examples_with_zero_attention_mass": slides_with_zero_attention_mass[:max_examples],
        },
        "examples": {
            "slide_ids": list(slide_ids[:max_examples]),
        },
    }

    return summary


def main() -> int:
    args = parse_args()

    summary = inspect_feature_store(
        args.feature_store,
        max_examples=args.max_examples,
    )

    rendered = json.dumps(summary, indent=2)
    print(rendered, flush=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
