#!/usr/bin/env python3
"""Build a conservative HEST-1k WSI-only inventory for task-agnostic EAF.

The script downloads/reads only the HEST metadata table. It excludes:
- non-human samples (unless --include-mouse is passed),
- HEST-Benchmark sample IDs,
- reserved organs (Lung by default),
- samples whose metadata match source-level blocklist terms,
- explicit sample-ID blocklists.

It does not download spatial transcriptomics data, patches, masks, or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_WSI_ROOT = Path("/data2/home/vcivale/projects/imaging/data/WSI")
DEFAULT_DATASET_ID = "hest_eaf_thunder_clean_v1"
DEFAULT_RELEASE = "v1.3.0"
DEFAULT_METADATA_URI = "hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_terms(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def norm_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def find_column(columns: Iterable[str], aliases: Iterable[str], *, required: bool = False) -> str | None:
    lookup = {norm_column(col): col for col in columns}
    for alias in aliases:
        found = lookup.get(norm_column(alias))
        if found is not None:
            return found
    if required:
        raise RuntimeError(
            f"Missing required metadata column. Tried aliases={list(aliases)}; "
            f"available={list(columns)}"
        )
    return None


def values(frame: pd.DataFrame, column: str | None, default: str = "") -> pd.Series:
    if column is None:
        return pd.Series([default] * len(frame), index=frame.index, dtype="string")
    return frame[column].fillna(default).astype(str).str.strip()


def load_hest_benchmark_ids() -> set[str]:
    """Read official HEST-Benchmark sample IDs from MahmoodLab/hest-bench."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to construct the HEST-Benchmark firewall."
        ) from exc

    dataset = load_dataset("MahmoodLab/hest-bench")
    ids: set[str] = set()
    for split in dataset.values():
        if "sample_id" not in split.column_names:
            continue
        ids.update(str(value).strip() for value in split["sample_id"] if str(value).strip())
    if not ids:
        raise RuntimeError("Official HEST-Benchmark query returned zero sample IDs.")
    return ids


def source_text(frame: pd.DataFrame) -> pd.Series:
    # Intentionally scans all metadata fields: dataset naming is not fully standardized
    # across the 180 source cohorts.
    return frame.fillna("").astype(str).agg(" | ".join, axis=1).str.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=DEFAULT_METADATA_URI)
    parser.add_argument("--wsi-root", type=Path, default=DEFAULT_WSI_ROOT)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--include-mouse", action="store_true")
    parser.add_argument(
        "--exclude-organ",
        action="append",
        default=["Lung"],
        help="Repeatable. Lung is excluded by default to preserve NSCLC OOD evaluation.",
    )
    parser.add_argument("--blocked-source-terms", type=Path, default=Path("catalog/hest_overlap/blocked_source_terms.txt"))
    parser.add_argument("--blocked-ids", type=Path, default=Path("catalog/hest_overlap/blocked_hest_ids.txt"))
    parser.add_argument("--skip-hest-benchmark-firewall", action="store_true")
    args = parser.parse_args()

    dataset_root = args.wsi_root / "datasets/pretraining" / args.dataset_id
    manifest_root = dataset_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    metadata.columns = [str(col).strip() for col in metadata.columns]
    if metadata.empty:
        raise RuntimeError("HEST metadata table is empty.")

    id_col = find_column(metadata.columns, ["id", "sample_id", "hest_id"], required=True)
    species_col = find_column(metadata.columns, ["species", "organism"])
    organ_col = find_column(metadata.columns, ["organ", "tissue", "primary_site"])
    technology_col = find_column(metadata.columns, ["technology", "tech", "assay"])
    oncotree_col = find_column(metadata.columns, ["oncotree_code", "oncotree", "cancer_type"])
    patient_col = find_column(metadata.columns, ["patient", "patient_id", "donor", "donor_id", "subject_id"])
    cohort_col = find_column(metadata.columns, ["dataset", "cohort", "source_dataset", "dataset_title", "study"])
    publication_col = find_column(metadata.columns, ["publication", "citation", "doi", "pubmed_id"])
    preservation_col = find_column(metadata.columns, ["preservation_method", "preservation", "sample_prep"])
    mpp_col = find_column(metadata.columns, ["pixel_size_um", "pixel_size", "mpp", "microns_per_pixel"])

    canonical = pd.DataFrame(index=metadata.index)
    canonical["id"] = values(metadata, id_col).str.upper()
    canonical["species"] = values(metadata, species_col)
    canonical["organ"] = values(metadata, organ_col)
    canonical["technology"] = values(metadata, technology_col)
    canonical["oncotree_code"] = values(metadata, oncotree_col)
    canonical["patient_id"] = values(metadata, patient_col)
    canonical["source_cohort"] = values(metadata, cohort_col)
    canonical["source_publication"] = values(metadata, publication_col)
    canonical["preservation"] = values(metadata, preservation_col)
    canonical["pixel_size_um"] = values(metadata, mpp_col)

    if canonical["id"].eq("").any():
        raise RuntimeError("One or more HEST rows have an empty sample ID.")
    duplicate_ids = canonical.loc[canonical["id"].duplicated(keep=False), "id"].unique().tolist()
    if duplicate_ids:
        raise RuntimeError(f"Duplicate HEST sample IDs: {duplicate_ids[:20]}")

    benchmark_ids: set[str] = set()
    if not args.skip_hest_benchmark_firewall:
        benchmark_ids = {value.upper() for value in load_hest_benchmark_ids()}
    blocked_ids = {value.upper() for value in read_terms(args.blocked_ids)}
    blocked_terms = [value.lower() for value in read_terms(args.blocked_source_terms)]
    all_text = source_text(metadata)

    exclusion_reasons: list[str] = []
    eligibility: list[str] = []
    for idx, row in canonical.iterrows():
        reasons: list[str] = []
        sample_id = row["id"]
        species = row["species"].lower()
        organ = row["organ"].lower()
        oncotree = row["oncotree_code"].lower()

        if not args.include_mouse:
            is_human = any(token in species for token in ("homo sapiens", "human", "h. sapiens"))
            if species and not is_human:
                reasons.append("non_human")
            elif not species:
                reasons.append("missing_species")

        for excluded_organ in args.exclude_organ:
            token = excluded_organ.strip().lower()
            if token and (token in organ or token in oncotree):
                reasons.append(f"reserved_organ:{token}")

        if sample_id in benchmark_ids:
            reasons.append("hest_benchmark_overlap")
        if sample_id in blocked_ids:
            reasons.append("explicit_id_blocklist")

        matched_terms = [term for term in blocked_terms if term and term in all_text.loc[idx]]
        if matched_terms:
            reasons.append("blocked_source:" + ",".join(sorted(set(matched_terms))))

        if not row["organ"]:
            reasons.append("missing_organ")

        metadata_errors = {"missing_species", "missing_organ"}
        if any(reason in metadata_errors for reason in reasons):
            status = "quarantine"
        elif reasons:
            status = "exclude"
        else:
            status = "eligible"

        eligibility.append(status)
        exclusion_reasons.append("|".join(reasons))

    canonical["eligibility"] = eligibility
    canonical["exclusion_reason"] = exclusion_reasons
    canonical["dataset_id"] = args.dataset_id
    canonical["source_provider"] = "HuggingFace"
    canonical["source_family"] = "HEST"
    canonical["release"] = args.release
    canonical["hf_repo_id"] = "MahmoodLab/hest"
    canonical["raw_glob"] = canonical["id"].map(lambda sid: f"sources/huggingface/hest/{args.release}/wsis/{sid}.*")
    canonical["sampling_group"] = canonical.apply(
        lambda row: row["patient_id"] if row["patient_id"] else f"{row['source_cohort']}::{row['id']}", axis=1
    )

    # Retain all original metadata with a stable prefix for later audits.
    original = metadata.copy()
    original.columns = [f"hest_{norm_column(col)}" for col in original.columns]
    inventory = pd.concat([canonical, original], axis=1)

    eligible = inventory[inventory["eligibility"] == "eligible"].copy()
    excluded = inventory[inventory["eligibility"] == "exclude"].copy()
    quarantine = inventory[inventory["eligibility"] == "quarantine"].copy()

    atomic_csv(manifest_root / "metadata_snapshot.csv", metadata)
    atomic_csv(manifest_root / "inventory_all.csv", inventory)
    atomic_csv(manifest_root / "eligible.csv", eligible)
    atomic_csv(manifest_root / "excluded.csv", excluded)
    atomic_csv(manifest_root / "quarantine.csv", quarantine)
    atomic_text(manifest_root / "download_ids.txt", "\n".join(eligible["id"].tolist()) + "\n")
    atomic_text(manifest_root / "hest_benchmark_ids.txt", "\n".join(sorted(benchmark_ids)) + "\n")

    source_counts = Counter(eligible["source_cohort"].replace("", "UNKNOWN"))
    organ_counts = Counter(eligible["organ"].replace("", "UNKNOWN"))
    dataset_yaml = {
        "dataset_id": args.dataset_id,
        "created_utc": utc_now(),
        "release": args.release,
        "source": {"provider": "HuggingFace", "repo_id": "MahmoodLab/hest"},
        "license": "CC-BY-NC-SA-4.0",
        "policy": {
            "wsi_only": True,
            "spatial_expression_downloaded": False,
            "labels_used": False,
            "human_only": not args.include_mouse,
            "excluded_organs": args.exclude_organ,
            "hest_benchmark_firewall": not args.skip_hest_benchmark_firewall,
            "source_term_firewall": blocked_terms,
        },
        "counts": {
            "metadata_rows": int(len(inventory)),
            "eligible": int(len(eligible)),
            "excluded": int(len(excluded)),
            "quarantine": int(len(quarantine)),
            "hest_benchmark_ids": int(len(benchmark_ids)),
        },
        "manifests": {
            "inventory": "manifests/inventory_all.csv",
            "eligible": "manifests/eligible.csv",
            "excluded": "manifests/excluded.csv",
            "quarantine": "manifests/quarantine.csv",
            "download_ids": "manifests/download_ids.txt",
        },
        "eligible_by_organ": dict(sorted(organ_counts.items())),
        "eligible_by_source_cohort": dict(sorted(source_counts.items())),
    }
    atomic_text(dataset_root / "dataset.json", json.dumps(dataset_yaml, indent=2, sort_keys=True) + "\n")

    print("=== HEST EAF INVENTORY ===")
    print(f"Metadata rows:        {len(inventory):,}")
    print(f"Eligible WSI:         {len(eligible):,}")
    print(f"Excluded:             {len(excluded):,}")
    print(f"Quarantine:           {len(quarantine):,}")
    print(f"HEST-Benchmark IDs:   {len(benchmark_ids):,}")
    print("\nEligible by organ:")
    for key, count in sorted(organ_counts.items()):
        print(f"  {key:24s} {count:5,d}")
    print("\nOutput:", dataset_root)


if __name__ == "__main__":
    main()
