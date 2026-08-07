from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


WSI_ROOT = Path("/data2/home/vcivale/projects/imaging/data/WSI")

DATASET_ROOT = (
    WSI_ROOT
    / "datasets/pretraining/tcga_eaf_thunder_clean_v1"
)

MANIFEST_ROOT = DATASET_ROOT / "manifests"
ELIGIBLE_PATH = MANIFEST_ROOT / "eligible_all.csv"

RAW_VIEW = DATASET_ROOT / "views/raw_flat"
COORD_VIEW = (
    DATASET_ROOT
    / "artifacts/trident/20x_512px_0px_overlap/patches"
)

LEAKAGE_SOURCE = WSI_ROOT / "catalog/thunder_overlap"
LEAKAGE_SNAPSHOT = MANIFEST_ROOT / "leakage_snapshot"

BLOCKLISTS = [
    "tcga_uniform_cases.txt",
    "tcga_crc_msi_cases.txt",
    "tcga_tils_cases.txt",
    "ccrcc_tcga_cases.txt",
    "all_thunder_tcga_cases.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, path)


def ensure_symlink(link: Path, target: Path) -> None:
    if not target.is_file():
        raise FileNotFoundError(target)

    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(
                f"Symlink incompatibile:\n  {link}\n  {target}"
            )
        return

    if link.exists():
        raise RuntimeError(f"Destinazione già esistente: {link}")

    relative = os.path.relpath(target, start=link.parent)
    link.symlink_to(relative)


def main() -> None:
    with ELIGIBLE_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        eligible = list(csv.DictReader(handle))

    if len(eligible) != 1860:
        raise RuntimeError(
            f"Attese 1.860 slide eligible, trovate {len(eligible)}"
        )

    RAW_VIEW.mkdir(parents=True, exist_ok=True)
    COORD_VIEW.mkdir(parents=True, exist_ok=True)
    LEAKAGE_SNAPSHOT.mkdir(parents=True, exist_ok=True)

    snapshot = {}

    for name in BLOCKLISTS:
        source = LEAKAGE_SOURCE / name

        if not source.is_file():
            raise FileNotFoundError(source)

        destination = LEAKAGE_SNAPSHOT / name
        shutil.copy2(source, destination)

        snapshot[name] = {
            "sha256": sha256(destination),
            "n_cases": sum(
                1
                for line in destination.read_text().splitlines()
                if line.strip()
            ),
        }

    rows = []
    raw_ready = []
    download_pending = []
    coords_ready = []
    coords_pending = []

    seen_file_names = set()
    seen_file_ids = set()

    for source_row in eligible:
        file_id = source_row["file_id"].strip()
        file_name = source_row["file_name"].strip()
        case_id = source_row["case_id"].strip().upper()
        project_id = source_row["project_id"].strip().upper()

        if file_id in seen_file_ids:
            raise RuntimeError(f"file_id duplicato: {file_id}")
        seen_file_ids.add(file_id)

        if file_name in seen_file_names:
            raise RuntimeError(f"file_name duplicato: {file_name}")
        seen_file_names.add(file_name)

        existing_raw_rel = source_row.get(
            "local_raw_path", ""
        ).strip()
        existing_coords_rel = source_row.get(
            "local_coords_path", ""
        ).strip()

        existing_raw = (
            WSI_ROOT / existing_raw_rel
            if existing_raw_rel
            else None
        )
        existing_coords = (
            WSI_ROOT / existing_coords_rel
            if existing_coords_rel
            else None
        )

        raw_available = bool(
            existing_raw is not None
            and existing_raw.is_file()
        )
        coords_available = bool(
            existing_coords is not None
            and existing_coords.is_file()
        )

        raw_destination_rel = source_row["raw_destination"].strip()
        canonical_raw = WSI_ROOT / raw_destination_rel

        if raw_available:
            ensure_symlink(
                RAW_VIEW / file_name,
                existing_raw,
            )
            raw_path = existing_raw_rel
            raw_status = "reuse_local"
        else:
            raw_path = raw_destination_rel
            raw_status = "download_pending"

        if coords_available:
            coord_name = existing_coords.name
            ensure_symlink(
                COORD_VIEW / coord_name,
                existing_coords,
            )
            coords_path = str(
                (COORD_VIEW / coord_name).relative_to(WSI_ROOT)
            )
            preprocessing_status = "coords_ready"
        else:
            coords_path = ""
            preprocessing_status = (
                "coords_pending"
                if raw_available
                else "awaiting_download"
            )

        row = {
            "dataset_id": "tcga_eaf_thunder_clean_v1",
            "source_provider": "GDC",
            "program": "TCGA",
            "cohort": project_id,
            "case_id": case_id,
            "file_id": file_id,
            "file_name": file_name,
            "slide_id": Path(file_name).stem,
            "slide_group": "diagnostic",
            "raw_path": raw_path,
            "raw_destination": raw_destination_rel,
            "raw_available": str(raw_available).lower(),
            "raw_status": raw_status,
            "coords_path": coords_path,
            "coords_available": str(coords_available).lower(),
            "preprocessing_status": preprocessing_status,
            "include_in_pretraining": "true",
            "labels_used": "false",
            "thunder_overlap": "false",
            "reserved_wsi_project": "false",
            "file_size": source_row["file_size"],
            "md5sum": source_row["md5sum"],
            "access": source_row["access"],
        }

        rows.append(row)

        if raw_available:
            raw_ready.append(row)
        else:
            download_pending.append(row)

        if coords_available:
            coords_ready.append(row)
        else:
            coords_pending.append(row)

    fields = list(rows[0].keys())

    atomic_csv(
        MANIFEST_ROOT / "slides.csv",
        rows,
        fields,
    )
    atomic_csv(
        MANIFEST_ROOT / "raw_ready.csv",
        raw_ready,
        fields,
    )
    atomic_csv(
        MANIFEST_ROOT / "download_pending.csv",
        download_pending,
        fields,
    )
    atomic_csv(
        MANIFEST_ROOT / "coords_ready.csv",
        coords_ready,
        fields,
    )
    atomic_csv(
        MANIFEST_ROOT / "coords_pending.csv",
        coords_pending,
        fields,
    )

    with (
        MANIFEST_ROOT / "trident_pending_local.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wsi"])
        writer.writeheader()

        for row in coords_pending:
            if row["raw_available"] == "true":
                writer.writerow({"wsi": row["file_name"]})

    leakage_report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "tcga_eaf_thunder_clean_v1",
        "exclusion_unit": "TCGA case/patient",
        "eligible_slides": len(rows),
        "eligible_cases": len({r["case_id"] for r in rows}),
        "thunder_overlap_after_filter": 0,
        "reserved_projects": ["TCGA-LUAD", "TCGA-LUSC"],
        "blocklists": snapshot,
    }

    report_path = MANIFEST_ROOT / "leakage_report.json"
    report_path.write_text(
        json.dumps(leakage_report, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    print("=== TCGA THUNDER-CLEAN MATERIALIZED ===")
    print(f"Slide totali:          {len(rows):,}")
    print(f"Casi totali:           {len({r['case_id'] for r in rows}):,}")
    print(f"Raw già disponibili:   {len(raw_ready):,}")
    print(f"Raw da scaricare:      {len(download_pending):,}")
    print(f"Coordinate pronte:     {len(coords_ready):,}")
    print(f"Coordinate pending:    {len(coords_pending):,}")
    print(f"Symlink raw:           {len(list(RAW_VIEW.glob('*.svs'))):,}")
    print(f"Symlink coordinate:    {len(list(COORD_VIEW.glob('*.h5'))):,}")

    print("\n=== PER COORTE ===")
    for cohort, count in sorted(
        Counter(r["cohort"] for r in rows).items()
    ):
        print(f"{cohort:12s} {count:5,d}")

    print("\nManifest:", MANIFEST_ROOT / "slides.csv")
    print("Leakage report:", report_path)


if __name__ == "__main__":
    main()
