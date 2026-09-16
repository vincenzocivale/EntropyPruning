#!/usr/bin/env python3
"""Audit downloaded HEST WSI files and materialize an EAF/TRIDENT dataset view."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image


DEFAULT_WSI_ROOT = Path("/data2/home/vcivale/projects/imaging/data/WSI")
VALID_EXTENSIONS = {".tif", ".tiff", ".btf", ".bigtiff"}


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def find_single_wsi(wsi_dir: Path, sample_id: str) -> Path | None:
    matches = sorted(
        path
        for path in wsi_dir.glob(f"{sample_id}.*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple WSI candidates for {sample_id}: {matches}")
    return matches[0] if matches else None


def safe_float(value: object) -> float | None:
    text = str(value).strip()
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    try:
        result = float(match.group(0))
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def thumbnail_metrics(slide: openslide.OpenSlide, max_size: int = 1024) -> tuple[float, float]:
    width, height = slide.dimensions
    scale = min(max_size / max(width, height), 1.0)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = slide.get_thumbnail(size).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    max_rgb = array.max(axis=2)
    min_rgb = array.min(axis=2)
    saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-6)
    luminance = array.mean(axis=2)
    tissue = (saturation > 0.05) & (luminance < 0.95)
    return float(tissue.mean()), float(array.std())


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Conflicting symlink: {link} -> {link.resolve()} != {target}")
        return
    if link.exists():
        raise RuntimeError(f"Path already exists and is not a symlink: {link}")
    link.symlink_to(os.path.relpath(target, start=link.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-root", type=Path, default=DEFAULT_WSI_ROOT)
    parser.add_argument("--dataset-id", default="hest_eaf_thunder_clean_v1")
    parser.add_argument("--release", default="v1.3.0")
    parser.add_argument("--min-width", type=int, default=2048)
    parser.add_argument("--min-height", type=int, default=2048)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.01)
    parser.add_argument("--min-thumbnail-std", type=float, default=0.02)
    parser.add_argument("--max-mpp", type=float, default=1.20)
    parser.add_argument("--write-thumbnails", action="store_true")
    args = parser.parse_args()

    dataset_root = args.wsi_root / "datasets/pretraining" / args.dataset_id
    manifest_root = dataset_root / "manifests"
    eligible = pd.read_csv(manifest_root / "eligible.csv", dtype=str).fillna("")
    source_root = args.wsi_root / "sources/huggingface/hest" / args.release
    wsi_dir = source_root / "wsis"
    raw_view = dataset_root / "views/raw_flat"
    thumbnail_dir = dataset_root / "artifacts/qc/thumbnails"
    coord_dir = dataset_root / "artifacts/trident/20x_512px_0px_overlap/patches"

    audit_rows: list[dict] = []
    slide_rows: list[dict] = []
    trident_rows: list[dict] = []

    for index, row in eligible.iterrows():
        sample_id = str(row["id"]).strip().upper()
        path = find_single_wsi(wsi_dir, sample_id)
        audit = {"id": sample_id, "wsi_path": "", "status": "missing", "reason": "missing_wsi"}
        if path is None:
            audit_rows.append(audit)
            continue

        audit["wsi_path"] = str(path.relative_to(args.wsi_root))
        try:
            slide = openslide.OpenSlide(str(path))
            width, height = slide.dimensions
            level_count = slide.level_count
            mpp_x = safe_float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
            mpp_y = safe_float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y))
            metadata_mpp = safe_float(row.get("pixel_size_um", ""))
            effective_mpp = mpp_x or mpp_y or metadata_mpp
            tissue_fraction, thumbnail_std = thumbnail_metrics(slide)

            reasons: list[str] = []
            if width < args.min_width or height < args.min_height:
                reasons.append("small_dimensions")
            if level_count < 2:
                reasons.append("non_pyramidal")
            if tissue_fraction < args.min_tissue_fraction:
                reasons.append("low_tissue_fraction")
            if thumbnail_std < args.min_thumbnail_std:
                reasons.append("low_pixel_variance")
            if effective_mpp is None:
                reasons.append("missing_mpp")
            elif effective_mpp > args.max_mpp:
                reasons.append("mpp_too_coarse")

            if args.write_thumbnails:
                thumbnail_dir.mkdir(parents=True, exist_ok=True)
                slide.get_thumbnail((1024, 1024)).convert("RGB").save(
                    thumbnail_dir / f"HEST__{sample_id}.jpg", quality=90
                )
            slide.close()

            status = "pass" if not reasons else "quarantine"
            audit.update(
                {
                    "status": status,
                    "reason": "|".join(reasons),
                    "width": width,
                    "height": height,
                    "level_count": level_count,
                    "mpp_x": "" if mpp_x is None else mpp_x,
                    "mpp_y": "" if mpp_y is None else mpp_y,
                    "metadata_mpp": "" if metadata_mpp is None else metadata_mpp,
                    "effective_mpp": "" if effective_mpp is None else effective_mpp,
                    "tissue_fraction": tissue_fraction,
                    "thumbnail_std": thumbnail_std,
                    "size_bytes": path.stat().st_size,
                }
            )
        except Exception as exc:  # OpenSlide/decoder errors must quarantine the file.
            audit.update({"status": "quarantine", "reason": f"open_error:{type(exc).__name__}:{exc}"})

        audit_rows.append(audit)
        if audit["status"] != "pass":
            continue

        flat_name = f"HEST__{sample_id}{path.suffix.lower()}"
        flat_path = raw_view / flat_name
        ensure_symlink(flat_path, path)
        slide_id = Path(flat_name).stem
        expected_coords = coord_dir / f"{slide_id}_patches.h5"
        coords_available = expected_coords.is_file() and expected_coords.stat().st_size > 0
        case_id = str(row.get("patient_id", "")).strip() or f"HEST-{sample_id}"
        sampling_group = str(row.get("sampling_group", "")).strip() or case_id

        slide_rows.append(
            {
                "dataset_id": args.dataset_id,
                "source_provider": "HuggingFace",
                "source_family": "HEST",
                "cohort": str(row.get("source_cohort", "")).strip() or "HEST",
                "case_id": case_id,
                "sampling_group": sampling_group,
                "slide_id": slide_id,
                "file_name": flat_name,
                "raw_path": str(path.relative_to(args.wsi_root)),
                "raw_available": "true",
                "coords_path": str(expected_coords.relative_to(args.wsi_root)),
                "coords_available": str(coords_available).lower(),
                "preprocessing_status": "coords_ready" if coords_available else "coords_pending",
                "include_in_pretraining": "true",
                "labels_used": "false",
                "organ": str(row.get("organ", "")),
                "technology": str(row.get("technology", "")),
                "species": str(row.get("species", "")),
                "preservation": str(row.get("preservation", "")),
                "oncotree_code": str(row.get("oncotree_code", "")),
                "source_publication": str(row.get("source_publication", "")),
                "hest_id": sample_id,
                "license": "CC-BY-NC-SA-4.0",
            }
        )
        if not coords_available:
            trident_rows.append({"wsi": flat_name})

        if (index + 1) % 50 == 0:
            print(f"Audited {index + 1:,}/{len(eligible):,}")

    audit_frame = pd.DataFrame(audit_rows)
    slides_frame = pd.DataFrame(slide_rows)
    trident_frame = pd.DataFrame(trident_rows, columns=["wsi"])
    atomic_csv(manifest_root / "wsi_audit.csv", audit_frame)
    atomic_csv(manifest_root / "slides.csv", slides_frame)
    atomic_csv(manifest_root / "trident_pending.csv", trident_frame)

    print("=== HEST WSI AUDIT ===")
    print(f"Eligible inventory:     {len(eligible):,}")
    print(f"Audit pass:             {(audit_frame['status'] == 'pass').sum():,}")
    print(f"Quarantine/missing:     {(audit_frame['status'] != 'pass').sum():,}")
    print(f"Coordinates ready:      {(slides_frame['coords_available'] == 'true').sum() if not slides_frame.empty else 0:,}")
    print(f"TRIDENT pending:        {len(trident_frame):,}")
    print("Manifest:", manifest_root / "slides.csv")


if __name__ == "__main__":
    main()
