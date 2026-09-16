#!/usr/bin/env python3
"""Combine TCGA-clean and HEST-clean canonical WSI manifests for EAF training."""
from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_WSI_ROOT = Path("/data2/home/vcivale/projects/imaging/data/WSI")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def hash_unit(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def deterministic_split(source_family: str, group: str, val_fraction: float) -> str:
    value = hash_unit(f"split::{source_family}::{group}")
    return "val" if value < val_fraction else "train"


def normalize(frame: pd.DataFrame, source_family: str) -> pd.DataFrame:
    result = frame.copy().fillna("")
    result["source_family"] = source_family
    if "sampling_group" not in result.columns:
        result["sampling_group"] = result.get("case_id", result.get("slide_id", ""))
    if "include_in_pretraining" in result.columns:
        result = result[result["include_in_pretraining"].astype(str).str.lower() == "true"]
    if "raw_available" in result.columns:
        result = result[result["raw_available"].astype(str).str.lower() == "true"]
    return result


def select_hest_groups(hest: pd.DataFrame, target_slides: int, seed: int) -> pd.DataFrame:
    """Select complete donor/patient groups deterministically, never splitting serial sections."""
    if target_slides >= len(hest):
        return hest.copy()
    groups = []
    for group, group_frame in hest.groupby("sampling_group", sort=False):
        groups.append(
            (
                hash_unit(f"hest-select::{seed}::{group}"),
                str(group),
                group_frame,
            )
        )
    groups.sort(key=lambda item: (item[0], item[1]))
    selected = []
    count = 0
    for _, _, group_frame in groups:
        if selected and count >= target_slides:
            break
        selected.append(group_frame)
        count += len(group_frame)
    return pd.concat(selected, ignore_index=True, sort=False) if selected else hest.iloc[0:0].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-root", type=Path, default=DEFAULT_WSI_ROOT)
    parser.add_argument("--tcga-dataset-id", default="tcga_eaf_thunder_clean_v1")
    parser.add_argument("--hest-dataset-id", default="hest_eaf_thunder_clean_v1")
    parser.add_argument("--output-dataset-id", default="eaf_multisource_clean_v1")
    parser.add_argument("--require-coords", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument(
        "--hest-share",
        type=float,
        default=0.30,
        help="Target HEST share among selected slides. Complete sampling groups are retained.",
    )
    parser.add_argument("--selection-seed", type=int, default=17)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    if not 0.0 < args.hest_share < 1.0:
        raise ValueError("--hest-share must be between 0 and 1")

    tcga_path = args.wsi_root / "datasets/pretraining" / args.tcga_dataset_id / "manifests/slides.csv"
    hest_path = args.wsi_root / "datasets/pretraining" / args.hest_dataset_id / "manifests/slides.csv"
    output_root = args.wsi_root / "datasets/pretraining" / args.output_dataset_id / "manifests"

    tcga = normalize(pd.read_csv(tcga_path, dtype=str), "TCGA")
    hest_all = normalize(pd.read_csv(hest_path, dtype=str), "HEST")
    if args.require_coords:
        for name, frame in (("TCGA", tcga), ("HEST", hest_all)):
            if "coords_available" not in frame.columns:
                raise RuntimeError(f"{name} manifest lacks coords_available.")
        tcga = tcga[tcga["coords_available"].astype(str).str.lower() == "true"].copy()
        hest_all = hest_all[hest_all["coords_available"].astype(str).str.lower() == "true"].copy()

    target_hest = round(len(tcga) * args.hest_share / (1.0 - args.hest_share))
    hest = select_hest_groups(hest_all, target_hest, args.selection_seed)
    combined = pd.concat([tcga, hest], ignore_index=True, sort=False).fillna("")

    duplicate_slide_ids = combined.loc[combined["slide_id"].duplicated(keep=False), "slide_id"].unique().tolist()
    if duplicate_slide_ids:
        raise RuntimeError(f"Duplicate slide IDs across sources: {duplicate_slide_ids[:20]}")

    combined["split"] = [
        deterministic_split(str(row.source_family), str(row.sampling_group), args.val_fraction)
        for row in combined.itertuples(index=False)
    ]
    combined["selection_seed"] = args.selection_seed
    combined["target_hest_share"] = args.hest_share
    combined = combined.sort_values(["split", "source_family", "cohort", "case_id", "slide_id"]).reset_index(drop=True)
    atomic_csv(output_root / "slides.csv", combined)
    atomic_csv(output_root / "train.csv", combined[combined["split"] == "train"])
    atomic_csv(output_root / "val.csv", combined[combined["split"] == "val"])
    atomic_csv(output_root / "hest_available.csv", hest_all)
    atomic_csv(output_root / "hest_selected.csv", hest)

    actual_share = len(hest) / len(combined) if len(combined) else 0.0
    print("=== EAF MULTISOURCE CLEAN ===")
    print(f"TCGA slides:          {len(tcga):,}")
    print(f"HEST available:       {len(hest_all):,}")
    print(f"HEST selected:        {len(hest):,}")
    print(f"Combined slides:      {len(combined):,}")
    print(f"Actual HEST share:    {actual_share:.3f}")
    print(f"Groups:               {combined['sampling_group'].nunique():,}")
    print("By source:")
    for source, count in sorted(Counter(combined["source_family"]).items()):
        print(f"  {source:10s} {count:6,d}")
    print("By split/source:")
    print(combined.groupby(["split", "source_family"]).size().to_string())
    print("Output:", output_root / "slides.csv")


if __name__ == "__main__":
    main()
