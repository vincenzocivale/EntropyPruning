#!/usr/bin/env python
"""Create a WSI feature store after morphology-aware tile coarsening.

This script operates on already extracted tile embeddings, including embeddings
produced by an unchanged EAF-pruned tile encoder. It materializes only selected
real tiles, preserving the standard ``tile_features + coords`` interface expected
by ABMIL, TITAN, and other WSI encoders.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi.bag import WSIBag
from src.data.wsi.h5_feature_store import H5WSIFeatureStore
from src.data.wsi.morphology_coarsening import (
    COARSENING_STRATEGIES,
    MorphologyCoarseningConfig,
    coarsen_wsi_tiles,
    config_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a coarsened HDF5 WSI feature store using random, spatial, "
            "morphology-only, or morphology-and-topology-preserving selection."
        )
    )
    parser.add_argument("--input-feature-store", type=Path, required=True)
    parser.add_argument("--output-feature-store", type=Path, required=True)
    parser.add_argument("--strategy", choices=COARSENING_STRATEGIES, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slide-ids-file", type=Path, default=None)
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--spatial-mode", choices=("auto", "grid", "knn"), default="auto")
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--spatial-k", type=int, default=8)
    parser.add_argument("--knn-chunk-size", type=int, default=512)

    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--merge-similarity", type=float, default=None)
    threshold_group.add_argument("--merge-quantile", type=float, default=0.25)
    parser.add_argument("--min-region-size", type=int, default=1)

    parser.add_argument("--size-exponent", type=float, default=0.5)
    parser.add_argument("--heterogeneity-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--rarity-weight", type=float, default=2.0)
    parser.add_argument("--boundary-fraction", type=float, default=0.35)

    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--max-fps-iterations", type=int, default=512)
    parser.add_argument("--diagnostic-top-fraction", type=float, default=0.10)
    parser.add_argument("--diagnostic-chunk-size", type=int, default=512)

    args = parser.parse_args()
    _validate_paths(args)
    return args


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.input_feature_store.exists():
        raise FileNotFoundError(
            f"input feature store not found: {args.input_feature_store}"
        )
    if args.input_feature_store.resolve() == args.output_feature_store.resolve():
        raise ValueError("input and output feature stores must be different files.")
    if args.output_feature_store.exists() and not args.overwrite:
        raise FileExistsError(
            f"output feature store already exists: {args.output_feature_store}"
        )
    if args.slide_ids_file is not None and not args.slide_ids_file.exists():
        raise FileNotFoundError(f"slide id file not found: {args.slide_ids_file}")


def _read_slide_ids(path: Path) -> tuple[str, ...]:
    slide_ids = []
    for raw_line in path.read_text().splitlines():
        value = raw_line.strip()
        if value and not value.startswith("#"):
            slide_ids.append(value)
    if not slide_ids:
        raise ValueError(f"slide id file is empty: {path}")
    return tuple(slide_ids)


def _config_from_args(args: argparse.Namespace) -> MorphologyCoarseningConfig:
    merge_quantile = 0.25 if args.merge_quantile is None else args.merge_quantile
    return MorphologyCoarseningConfig(
        keep_ratio=args.keep_ratio,
        strategy=args.strategy,
        seed=args.seed,
        spatial_mode=args.spatial_mode,
        connectivity=args.connectivity,
        spatial_k=args.spatial_k,
        knn_chunk_size=args.knn_chunk_size,
        merge_similarity=args.merge_similarity,
        merge_quantile=merge_quantile,
        min_region_size=args.min_region_size,
        size_exponent=args.size_exponent,
        heterogeneity_weight=args.heterogeneity_weight,
        boundary_weight=args.boundary_weight,
        rarity_weight=args.rarity_weight,
        boundary_fraction=args.boundary_fraction,
        projection_dim=args.projection_dim,
        max_fps_iterations=args.max_fps_iterations,
        diagnostic_top_fraction=args.diagnostic_top_fraction,
        diagnostic_chunk_size=args.diagnostic_chunk_size,
    )


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "slide_id",
        "strategy",
        "requested_keep_ratio",
        "effective_keep_ratio",
        "n_tiles_input",
        "n_tiles_kept",
        "n_regions",
        "region_coverage",
        "significant_region_coverage",
        "rare_region_coverage",
        "projected_feature_coverage",
        "mean_normalized_spatial_distance",
        "boundary_top_recall",
        "attention_mass_retained",
        "oracle_attention_mass_at_k",
        "relative_attention_mass_retained",
        "attention_top_recall",
        "spatial_edge_mode",
        "neighbor_edge_count",
        "merge_threshold",
    ]
    all_keys = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = _config_from_args(args)

    if args.output_feature_store.exists() and args.overwrite:
        args.output_feature_store.unlink()
    args.output_feature_store.parent.mkdir(parents=True, exist_ok=True)

    report_csv = args.report_csv or args.output_feature_store.with_suffix(
        args.output_feature_store.suffix + ".coarsening.csv"
    )
    summary_json = args.summary_json or args.output_feature_store.with_suffix(
        args.output_feature_store.suffix + ".coarsening.json"
    )
    for path in (report_csv, summary_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"report already exists: {path}")

    input_store = H5WSIFeatureStore(args.input_feature_store)
    output_store = H5WSIFeatureStore(args.output_feature_store)
    available = input_store.slide_ids()
    if not available:
        raise ValueError(f"input store contains no slides: {args.input_feature_store}")
    slide_ids = (
        available
        if args.slide_ids_file is None
        else _read_slide_ids(args.slide_ids_file)
    )
    missing = sorted(set(slide_ids) - set(available))
    if missing:
        raise KeyError(
            f"{len(missing)} requested slide(s) are absent from the input store, "
            f"e.g. {missing[:10]}"
        )

    print(
        json.dumps(
            {
                "event": "start",
                "input_feature_store": str(args.input_feature_store),
                "output_feature_store": str(args.output_feature_store),
                "n_slides": len(slide_ids),
                "config": config_to_dict(config),
            }
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for slide_index, slide_id in enumerate(slide_ids):
        bag = input_store.read(slide_id)
        result = coarsen_wsi_tiles(
            bag.tile_features,
            bag.coords,
            config,
            attention=bag.attention,
        )
        selected = result.selected_indices
        metadata = dict(bag.metadata) if bag.metadata is not None else {}
        metadata.update(
            {
                "coarsened_by": "MorphologyTopologyCoarsener",
                "coarsening_strategy": config.strategy,
                "coarsening_config": config_to_dict(config),
                "coarsening_input_n_tiles": bag.n_tiles,
                "coarsening_output_n_tiles": int(selected.numel()),
                "coarsening_preserved_original_tile_order": True,
            }
        )
        output_store.write(
            WSIBag(
                slide_id=bag.slide_id,
                tile_features=bag.tile_features[selected],
                coords=bag.coords[selected] if bag.coords is not None else None,
                label=bag.label,
                attention=bag.attention[selected] if bag.attention is not None else None,
                metadata=metadata,
            )
        )
        row = {"slide_id": slide_id, **result.diagnostics}
        rows.append(row)
        if (slide_index + 1) % 25 == 0 or slide_index + 1 == len(slide_ids):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "slides_done": slide_index + 1,
                        "slides_total": len(slide_ids),
                    }
                ),
                flush=True,
            )

    _write_csv(report_csv, rows)
    summary = {
        "event": "done",
        "input_feature_store": str(args.input_feature_store),
        "output_feature_store": str(args.output_feature_store),
        "report_csv": str(report_csv),
        "n_slides": len(rows),
        "config": config_to_dict(config),
        "n_tiles_input_total": sum(int(row["n_tiles_input"]) for row in rows),
        "n_tiles_kept_total": sum(int(row["n_tiles_kept"]) for row in rows),
        "mean_effective_keep_ratio": _mean_numeric(rows, "effective_keep_ratio"),
        "mean_projected_feature_coverage": _mean_numeric(
            rows, "projected_feature_coverage"
        ),
        "mean_normalized_spatial_distance": _mean_numeric(
            rows, "mean_normalized_spatial_distance"
        ),
        "mean_region_coverage": _mean_numeric(rows, "region_coverage"),
        "mean_significant_region_coverage": _mean_numeric(
            rows, "significant_region_coverage"
        ),
        "mean_rare_region_coverage": _mean_numeric(rows, "rare_region_coverage"),
        "mean_boundary_top_recall": _mean_numeric(rows, "boundary_top_recall"),
        "mean_attention_mass_retained": _mean_numeric(
            rows, "attention_mass_retained"
        ),
        "mean_relative_attention_mass_retained": _mean_numeric(
            rows, "relative_attention_mass_retained"
        ),
        "mean_attention_top_recall": _mean_numeric(rows, "attention_top_recall"),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
