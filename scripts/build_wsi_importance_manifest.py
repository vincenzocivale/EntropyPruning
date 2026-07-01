#!/usr/bin/env python
"""Build a manifest for importing precomputed WSI tile-importance targets.

Scans a directory of per-slide target artifacts (HDF5, ``.pt``/``.pth``,
``.npy``, or ``.npz`` — e.g. TRIDENT-style per-slide files, or any other
precomputed tile-importance export) and, optionally, a matching directory of
coordinate files and a labels CSV, and writes a manifest consumable by
``scripts/import_wsi_importance_targets.py``:

    slide_id,target_path,coords_path,label,target_source,target_type
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


_ALLOWED_SUFFIXES = {".h5", ".hdf5", ".pt", ".pth", ".npy", ".npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manifest from a directory of precomputed tile-importance "
            "target files."
        )
    )

    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)

    parser.add_argument("--coords-dir", type=Path, default=None)
    parser.add_argument("--labels-csv", type=Path, default=None)

    parser.add_argument(
        "--target-glob",
        type=str,
        default="*.h5",
        help="Glob used inside --targets-dir.",
    )
    parser.add_argument(
        "--target-source",
        type=str,
        default="",
        help="Constant target_source value written to every manifest row.",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="tile_importance",
        help="Constant target_type value written to every manifest row.",
    )
    parser.add_argument(
        "--require-coords",
        action="store_true",
        help="Fail if a target file has no matching coords file.",
    )
    parser.add_argument(
        "--require-labels",
        action="store_true",
        help="Fail if a slide has no label in --labels-csv.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths instead of paths relative to the manifest location.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output manifest if it already exists.",
    )

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.targets_dir.exists():
        raise FileNotFoundError(f"targets directory not found: {args.targets_dir}")
    if not args.targets_dir.is_dir():
        raise NotADirectoryError(f"targets path is not a directory: {args.targets_dir}")

    if args.coords_dir is not None:
        if not args.coords_dir.exists():
            raise FileNotFoundError(f"coords directory not found: {args.coords_dir}")
        if not args.coords_dir.is_dir():
            raise NotADirectoryError(f"coords path is not a directory: {args.coords_dir}")

    if args.require_coords and args.coords_dir is None:
        raise ValueError("--require-coords requires --coords-dir.")

    if args.labels_csv is not None and not args.labels_csv.exists():
        raise FileNotFoundError(f"labels CSV not found: {args.labels_csv}")

    if args.require_labels and args.labels_csv is None:
        raise ValueError("--require-labels requires --labels-csv.")

    if args.output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"output manifest already exists: {args.output_manifest}")


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in _ALLOWED_SUFFIXES


def _slide_id_from_path(path: Path) -> str:
    return path.stem


def _discover_files(directory: Path, glob_pattern: str) -> dict[str, Path]:
    paths = sorted(path for path in directory.glob(glob_pattern) if path.is_file())

    records: dict[str, Path] = {}
    duplicates: list[str] = []

    for path in paths:
        if not _is_supported(path):
            continue

        slide_id = _slide_id_from_path(path)
        if slide_id in records:
            duplicates.append(slide_id)
        records[slide_id] = path

    if not records:
        raise ValueError(
            f"no supported target files found in {directory} using glob "
            f"{glob_pattern!r}; supported suffixes are {sorted(_ALLOWED_SUFFIXES)}."
        )

    if duplicates:
        raise ValueError(
            "duplicate target slide ids: " + ", ".join(sorted(set(duplicates))[:10])
        )

    return records


def _discover_coords(coords_dir: Path | None) -> dict[str, Path]:
    if coords_dir is None:
        return {}

    records: dict[str, Path] = {}
    duplicates: list[str] = []

    for path in sorted(coords_dir.iterdir()):
        if not path.is_file() or not _is_supported(path):
            continue

        slide_id = _slide_id_from_path(path)
        if slide_id in records:
            duplicates.append(slide_id)
        records[slide_id] = path

    if duplicates:
        raise ValueError(
            "duplicate coords slide ids: " + ", ".join(sorted(set(duplicates))[:10])
        )

    return records


def _read_labels(labels_csv: Path | None) -> dict[str, str]:
    if labels_csv is None:
        return {}

    with labels_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"labels CSV has no header: {labels_csv}")

        required = {"slide_id", "label"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "labels CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        labels: dict[str, str] = {}
        duplicates: list[str] = []

        for row_number, row in enumerate(reader, start=2):
            slide_id = (row.get("slide_id") or "").strip()
            label = (row.get("label") or "").strip()

            if not slide_id:
                raise ValueError(f"labels CSV row {row_number} has empty slide_id.")
            if not label:
                raise ValueError(f"labels CSV row {row_number} has empty label.")

            if slide_id in labels:
                duplicates.append(slide_id)
            labels[slide_id] = label

    if duplicates:
        raise ValueError(
            "labels CSV contains duplicate slide ids: " + ", ".join(sorted(set(duplicates))[:10])
        )

    return labels


def _format_path(path: Path, *, manifest_path: Path, absolute: bool) -> str:
    resolved = path.resolve()

    if absolute:
        return str(resolved)

    base_dir = manifest_path.resolve().parent
    return os.path.relpath(resolved, base_dir)


def main() -> int:
    args = parse_args()

    targets = _discover_files(args.targets_dir, args.target_glob)
    coords = _discover_coords(args.coords_dir)
    labels = _read_labels(args.labels_csv)

    rows = []
    missing_coords = []
    missing_labels = []

    for slide_id in sorted(targets):
        target_path = targets[slide_id]
        coord_path = coords.get(slide_id)
        label = labels.get(slide_id, "")

        if args.require_coords and coord_path is None:
            missing_coords.append(slide_id)
        if args.require_labels and label == "":
            missing_labels.append(slide_id)

        rows.append(
            {
                "slide_id": slide_id,
                "target_path": _format_path(
                    target_path,
                    manifest_path=args.output_manifest,
                    absolute=args.absolute_paths,
                ),
                "coords_path": (
                    _format_path(
                        coord_path,
                        manifest_path=args.output_manifest,
                        absolute=args.absolute_paths,
                    )
                    if coord_path is not None
                    else ""
                ),
                "label": label,
                "target_source": args.target_source,
                "target_type": args.target_type,
            }
        )

    if missing_coords:
        raise ValueError("missing coords for slides: " + ", ".join(missing_coords[:10]))

    if missing_labels:
        raise ValueError("missing labels for slides: " + ", ".join(missing_labels[:10]))

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with args.output_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slide_id",
                "target_path",
                "coords_path",
                "label",
                "target_source",
                "target_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"wrote {len(rows)} rows to {args.output_manifest}; "
        f"with_coords={sum(1 for row in rows if row['coords_path'])}; "
        f"with_labels={sum(1 for row in rows if row['label'])}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
