#!/usr/bin/env python3
"""Analyze how WSI-model attention relates to tile-level embeddings.

This command is deliberately label-free. It consumes existing tile feature
stores and existing attention artifacts; it does not train a forecaster or run
any downstream task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow direct execution from the repository checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.wsi_attention_embedding import (  # noqa: E402
    aggregate_slide_metrics,
    analyze_attention_embedding_relation,
    flatten_numeric_metrics,
)
from src.data.wsi import (  # noqa: E402
    EmbeddedAttentionSource,
    FeatureStoreAttentionSource,
    H5WSIFeatureStore,
    ManifestAttentionSource,
    align_attention_to_bag,
)


def _parse_axis_selection(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(-?\d+)\s*=\s*(-?\d+)\s*", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "attention selections must use AXIS=INDEX, for example --attention-select 0=-1"
        )
    return int(match.group(1)), int(match.group(2))


def _load_slide_ids(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        first = handle.readline()
        handle.seek(0)
        if "," in first or first.strip() == "slide_id":
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "slide_id" not in reader.fieldnames:
                raise ValueError(f"slide list CSV must contain a slide_id column: {path}")
            rows = [(row.get("slide_id") or "").strip() for row in reader]
        else:
            rows = [line.strip() for line in handle]
    rows = [slide_id for slide_id in rows if slide_id]
    if len(rows) != len(set(rows)):
        raise ValueError(f"slide list contains duplicate slide_id values: {path}")
    return rows


def _safe_filename(slide_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", slide_id).strip("._") or "slide"
    digest = hashlib.sha1(slide_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe}__{digest}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a label-free audit of WSI attention against tile-level embedding geometry."
        )
    )
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--attention-store",
        type=Path,
        help=(
            "Existing HDF5 target/attention store. If omitted together with "
            "--attention-manifest, read the embedded attention field from --feature-store."
        ),
    )
    source.add_argument(
        "--attention-manifest",
        type=Path,
        help="CSV mapping slide_id to attention_path (legacy target_path is accepted).",
    )

    parser.add_argument("--alignment", choices=("auto", "index", "coords"), default="auto")
    parser.add_argument("--attention-key")
    parser.add_argument("--coords-key")
    parser.add_argument("--tile-axis", type=int)
    parser.add_argument(
        "--tile-slice-start",
        type=int,
        help=(
            "Explicit start of the tile slice on a token axis containing CLS/special tokens; "
            "for [CLS, tile_0, ...], use 1."
        ),
    )
    parser.add_argument(
        "--attention-reduction",
        choices=("mean", "sum", "max", "l2"),
        default="mean",
    )
    parser.add_argument(
        "--attention-select",
        action="append",
        default=[],
        type=_parse_axis_selection,
        metavar="AXIS=INDEX",
        help="Select a layer/head/query axis before reducing the remaining non-tile axes.",
    )
    parser.add_argument(
        "--attention-normalization",
        choices=("auto", "none", "softmax", "l1", "minmax"),
        default="auto",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=0,
        help="Compute mean cosine distance to k nearest reference embeddings; 0 disables it.",
    )
    parser.add_argument("--knn-reference-size", type=int, default=4096)
    parser.add_argument("--knn-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slide-list", type=Path)
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_slides is not None and args.max_slides <= 0:
        raise ValueError("--max-slides must be positive.")
    if args.knn_k < 0:
        raise ValueError("--knn-k must be non-negative.")
    if args.knn_reference_size <= 0 or args.knn_chunk_size <= 0:
        raise ValueError("kNN reference/chunk sizes must be positive.")
    axes = [axis for axis, _ in args.attention_select]
    if len(axes) != len(set(axes)):
        raise ValueError("each --attention-select axis may be specified only once.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --overwrite to update it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_dir = output_dir / "tiles"
    detail_dir = output_dir / "slide_details"
    tile_dir.mkdir(exist_ok=True)
    detail_dir.mkdir(exist_ok=True)

    feature_store = H5WSIFeatureStore(args.feature_store, read_only=True)
    if args.attention_store is not None:
        attention_store = H5WSIFeatureStore(args.attention_store, read_only=True)
        attention_source = FeatureStoreAttentionSource(attention_store)
        source_description = {"kind": "h5_attention_store", "path": str(args.attention_store)}
    elif args.attention_manifest is not None:
        selections = dict(args.attention_select)
        attention_source = ManifestAttentionSource(
            args.attention_manifest,
            attention_key=args.attention_key,
            coords_key=args.coords_key,
            tile_axis=args.tile_axis,
            tile_slice_start=args.tile_slice_start,
            reduction=args.attention_reduction,
            selections=selections,
        )
        source_description = {"kind": "manifest", "path": str(args.attention_manifest)}
    else:
        attention_source = EmbeddedAttentionSource(feature_store)
        source_description = {"kind": "embedded_feature_store_attention"}

    available = feature_store.slide_ids()
    if args.slide_list is not None:
        requested = _load_slide_ids(args.slide_list)
        missing = sorted(set(requested).difference(available))
        if missing:
            raise KeyError(
                f"{len(missing)} requested slide(s) are absent from the feature store, "
                f"including {missing[:10]}"
            )
        slide_ids = requested
    else:
        slide_ids = list(available)
    if args.max_slides is not None:
        slide_ids = slide_ids[: args.max_slides]
    if not slide_ids:
        raise ValueError("no slides selected for analysis.")

    config = {
        "feature_store": str(args.feature_store.resolve()),
        "attention_source": source_description,
        "alignment": args.alignment,
        "attention_key": args.attention_key,
        "coords_key": args.coords_key,
        "tile_axis": args.tile_axis,
        "tile_slice_start": args.tile_slice_start,
        "attention_reduction": args.attention_reduction,
        "attention_select": {str(axis): index for axis, index in args.attention_select},
        "attention_normalization": args.attention_normalization,
        "knn_k": args.knn_k,
        "knn_reference_size": args.knn_reference_size,
        "knn_chunk_size": args.knn_chunk_size,
        "seed": args.seed,
        "n_selected_slides": len(slide_ids),
        "label_free": True,
        "downstream_evaluation": False,
    }
    _atomic_json(output_dir / "config.json", config)

    flat_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for position, slide_id in enumerate(slide_ids):
        try:
            bag = feature_store.read(slide_id)
            attention = attention_source.read(slide_id, n_tiles=bag.n_tiles)
            bag, attention = align_attention_to_bag(bag, attention, mode=args.alignment)
            summary, arrays = analyze_attention_embedding_relation(
                bag.tile_features,
                attention.values,
                bag.coords,
                attention_normalization=args.attention_normalization,
                knn_k=args.knn_k,
                knn_reference_size=args.knn_reference_size,
                knn_chunk_size=args.knn_chunk_size,
                seed=args.seed + position,
            )
            detail = {
                "slide_id": slide_id,
                "feature_dim": bag.feature_dim,
                "attention_metadata": attention.metadata,
                **summary,
            }
            safe_name = _safe_filename(slide_id)
            _atomic_json(detail_dir / f"{safe_name}.json", detail)
            _atomic_npz(tile_dir / f"{safe_name}.npz", arrays)
            flat_rows.append(
                {
                    "slide_id": slide_id,
                    "feature_dim": bag.feature_dim,
                    **flatten_numeric_metrics(summary),
                }
            )
            print(f"[{position + 1}/{len(slide_ids)}] analyzed {slide_id}", flush=True)
        except Exception as exc:  # deliberate per-slide boundary for large cohorts
            if not args.continue_on_error:
                raise
            errors.append(
                {
                    "slide_id": slide_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(
                f"[{position + 1}/{len(slide_ids)}] ERROR {slide_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    _write_csv(output_dir / "slide_metrics.csv", flat_rows)
    _write_csv(output_dir / "errors.csv", errors)
    numeric_rows = [
        {key: value for key, value in row.items() if key not in {"slide_id", "feature_dim"}}
        for row in flat_rows
    ]
    aggregate = aggregate_slide_metrics(numeric_rows)
    aggregate["n_requested_slides"] = len(slide_ids)
    aggregate["n_failed_slides"] = len(errors)
    aggregate["n_successful_slides"] = len(flat_rows)
    _atomic_json(output_dir / "aggregate.json", aggregate)

    print(
        f"completed: {len(flat_rows)} successful, {len(errors)} failed; output={output_dir}",
        flush=True,
    )
    return 0 if flat_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
