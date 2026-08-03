#!/usr/bin/env python
"""Import TRIDENT-style feature HDF5 files into an EAF WSI feature store."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import (
    H5WSIFeatureStore,
    TridentSlideRecord,
    load_trident_slide_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import TRIDENT-style per-slide HDF5 features into EAF."
    )

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-feature-store", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--feature-dataset", type=str, default=None)
    parser.add_argument("--coords-dataset", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.feature_dim is not None and args.feature_dim <= 0:
        raise ValueError("--feature-dim must be positive when provided.")
    if args.output_feature_store.exists() and not args.overwrite:
        raise FileExistsError(
            f"output feature store already exists: {args.output_feature_store}"
        )


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


def _read_manifest(path: Path) -> tuple[TridentSlideRecord, ...]:
    base_dir = path.parent
    records: list[TridentSlideRecord] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")

        required = {"slide_id", "features_path"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "manifest is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            slide_id = (row.get("slide_id") or "").strip()
            features_raw = (row.get("features_path") or "").strip()
            coords_raw = (row.get("coords_path") or "").strip()

            if not slide_id:
                raise ValueError(f"manifest row {row_number} has empty slide_id.")
            if not features_raw:
                raise ValueError(f"manifest row {row_number} has empty features_path.")

            metadata = {
                "trident_manifest": str(path),
                "trident_manifest_row": row_number,
            }

            records.append(
                TridentSlideRecord(
                    slide_id=slide_id,
                    features_path=_resolve_path(features_raw, base_dir=base_dir),
                    coords_path=(
                        _resolve_path(coords_raw, base_dir=base_dir)
                        if coords_raw
                        else None
                    ),
                    label=_parse_label(row.get("label")),
                    metadata=metadata,
                )
            )

    if not records:
        raise ValueError(f"manifest contains no rows: {path}")

    slide_ids = [record.slide_id for record in records]
    duplicates = sorted({slide_id for slide_id in slide_ids if slide_ids.count(slide_id) > 1})
    if duplicates:
        raise ValueError(
            "manifest contains duplicate slide ids: "
            + ", ".join(duplicates[:10])
        )

    return tuple(records)


def main() -> int:
    args = parse_args()

    if args.output_feature_store.exists() and args.overwrite:
        args.output_feature_store.unlink()

    args.output_feature_store.parent.mkdir(parents=True, exist_ok=True)

    records = _read_manifest(args.manifest)
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
                "n_records": len(records),
            }
        ),
        flush=True,
    )

    for record in records:
        bag = load_trident_slide_record(
            record,
            feature_dataset_name=args.feature_dataset,
            coords_dataset_name=args.coords_dataset,
            expected_feature_dim=args.feature_dim,
        )
        output_store.write(bag)

        n_slides += 1
        n_tiles_total += bag.n_tiles
        n_with_coords += int(bag.coords is not None)
        n_with_label += int(bag.label is not None)

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
