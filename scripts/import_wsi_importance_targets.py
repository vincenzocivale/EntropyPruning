#!/usr/bin/env python
"""Import precomputed tile-importance targets into an EAF WSI feature store.

Consumes a manifest of ``slide_id,target_path,coords_path,label,target_source,
target_type`` rows, where each ``target_path`` is a per-slide artifact
(HDF5, ``.pt``/``.pth``, ``.npy``, or ``.npz``) containing a 1D tile
importance vector — e.g. ABMIL attention, a WSI foundation model tile score,
or any other precomputed target. TRIDENT is never a hard dependency: this
script only expects TRIDENT-*style* per-slide artifacts (one file per
slide), not a live TRIDENT installation.

The output is an ordinary ``H5WSIFeatureStore`` usable as a
``--target-feature-store`` for ``scripts/train_wsi_importance_forecaster.py``
or validated with ``scripts/validate_wsi_feature_store.py --require-attention``.
Since a target store has no real per-tile embedding, ``tile_features`` is set
to the target vector reshaped to ``[n_tiles, 1]`` — this makes accidental
misuse as an *input* feature store fail loudly on a feature-dim mismatch
rather than silently succeed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import (
    H5WSIFeatureStore,
    WSIBag,
    read_wsi_importance_coords_tensor,
    read_wsi_importance_target_tensor,
)

_NORMALIZATION_MODES = ("none", "sum", "minmax", "softmax")


@dataclass(frozen=True)
class ImportanceTargetManifestRow:
    """One manifest row describing a precomputed importance target."""

    slide_id: str
    target_path: Path
    coords_path: Path | None
    label: int | float | None
    target_source: str | None
    target_type: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import precomputed per-slide tile-importance targets into an "
            "EAF HDF5 WSI feature store."
        )
    )

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-feature-store", type=Path, required=True)

    parser.add_argument(
        "--target-key",
        type=str,
        default=None,
        help=(
            "Dataset/array key for the importance value inside each "
            "target_path file. If omitted, common names are tried "
            "(attention, importance, tile_importance, score, scores)."
        ),
    )
    parser.add_argument(
        "--coords-key",
        type=str,
        default=None,
        help="Dataset/array key for coordinates inside each coords_path file.",
    )
    parser.add_argument("--require-coords", action="store_true")
    parser.add_argument(
        "--normalize",
        type=str,
        choices=_NORMALIZATION_MODES,
        default="none",
        help="Optional per-slide normalization applied before writing.",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--created-by",
        type=str,
        default=None,
        help="Free-form provenance label stored in bag metadata.",
    )
    parser.add_argument("--trident-job-dir", type=str, default=None)
    parser.add_argument("--patch-encoder", type=str, default=None)
    parser.add_argument("--slide-encoder", type=str, default=None)
    parser.add_argument("--mag", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=None)

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_feature_store.exists() and not args.overwrite:
        raise FileExistsError(
            f"output feature store already exists: {args.output_feature_store}"
        )
    if args.patch_size is not None and args.patch_size <= 0:
        raise ValueError("--patch-size must be positive when provided.")


def _resolve_path(raw_value: str, *, base_dir: Path) -> Path:
    path = Path(raw_value)
    return path if path.is_absolute() else base_dir / path


def _parse_label(raw_value: str | None) -> int | float | None:
    if raw_value is None:
        return None

    value = raw_value.strip()
    if value == "":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"label must be numeric when provided; got {raw_value!r}") from exc


def _read_manifest(path: Path) -> tuple[ImportanceTargetManifestRow, ...]:
    base_dir = path.parent
    rows: list[ImportanceTargetManifestRow] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")

        required = {"slide_id", "target_path"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "manifest is missing required columns: " + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            slide_id = (row.get("slide_id") or "").strip()
            target_raw = (row.get("target_path") or "").strip()
            coords_raw = (row.get("coords_path") or "").strip()
            target_source = (row.get("target_source") or "").strip() or None
            target_type = (row.get("target_type") or "").strip() or None

            if not slide_id:
                raise ValueError(f"manifest row {row_number} has empty slide_id.")
            if not target_raw:
                raise ValueError(f"manifest row {row_number} has empty target_path.")

            rows.append(
                ImportanceTargetManifestRow(
                    slide_id=slide_id,
                    target_path=_resolve_path(target_raw, base_dir=base_dir),
                    coords_path=(
                        _resolve_path(coords_raw, base_dir=base_dir) if coords_raw else None
                    ),
                    label=_parse_label(row.get("label")),
                    target_source=target_source,
                    target_type=target_type,
                )
            )

    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")

    slide_ids = [row.slide_id for row in rows]
    duplicates = sorted({slide_id for slide_id in slide_ids if slide_ids.count(slide_id) > 1})
    if duplicates:
        raise ValueError(
            "manifest contains duplicate slide ids: " + ", ".join(duplicates[:10])
        )

    return tuple(rows)


def _normalize_target(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return tensor

    if mode == "sum":
        total = tensor.sum()
        if total <= 0:
            raise ValueError("cannot sum-normalize a target with non-positive total mass.")
        return tensor / total

    if mode == "minmax":
        low = tensor.min()
        high = tensor.max()
        if high <= low:
            raise ValueError("cannot min-max normalize a constant target.")
        return (tensor - low) / (high - low)

    if mode == "softmax":
        return torch.softmax(tensor, dim=0)

    raise ValueError(f"unknown normalization mode: {mode!r}")


def main() -> int:
    args = parse_args()

    if args.output_feature_store.exists() and args.overwrite:
        args.output_feature_store.unlink()

    args.output_feature_store.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(args.manifest)
    output_store = H5WSIFeatureStore(args.output_feature_store)

    n_slides = 0
    n_tiles_total = 0
    n_with_coords = 0
    n_with_label = 0

    print(
        json.dumps(
            {
                "event": "start",
                "manifest": str(args.manifest),
                "output_feature_store": str(args.output_feature_store),
                "n_records": len(rows),
                "normalize": args.normalize,
            }
        ),
        flush=True,
    )

    for row in rows:
        target = read_wsi_importance_target_tensor(row.target_path, key=args.target_key)
        target = _normalize_target(target, args.normalize)

        coords = None
        if row.coords_path is not None:
            coords = read_wsi_importance_coords_tensor(row.coords_path, key=args.coords_key)
            if coords.shape[0] != target.shape[0]:
                raise ValueError(
                    f"slide {row.slide_id} has {target.shape[0]} target values "
                    f"but {coords.shape[0]} coordinates."
                )
        elif args.require_coords:
            raise ValueError(
                f"slide {row.slide_id} is missing coords but --require-coords was set."
            )

        metadata = {
            "target_source": row.target_source,
            "target_type": row.target_type or "tile_importance",
            "target_key": args.target_key,
            "target_path": str(row.target_path),
            "coords_source": str(row.coords_path) if row.coords_path is not None else None,
            "created_by": args.created_by,
            "normalization": args.normalize,
            "trident_job_dir": args.trident_job_dir,
            "patch_encoder": args.patch_encoder,
            "slide_encoder": args.slide_encoder,
            "mag": args.mag,
            "patch_size": args.patch_size,
        }

        bag = WSIBag(
            slide_id=row.slide_id,
            tile_features=target.unsqueeze(1),
            coords=coords,
            label=row.label,
            attention=target,
            metadata=metadata,
        )
        output_store.write(bag)

        n_slides += 1
        n_tiles_total += bag.n_tiles
        n_with_coords += int(coords is not None)
        n_with_label += int(row.label is not None)

    summary = {
        "event": "done",
        "manifest": str(args.manifest),
        "output_feature_store": str(args.output_feature_store),
        "n_slides": n_slides,
        "n_tiles_total": n_tiles_total,
        "n_with_coords": n_with_coords,
        "n_with_label": n_with_label,
    }
    print(json.dumps(summary, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
