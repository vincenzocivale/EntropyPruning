from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


GDC_API = "https://api.gdc.cancer.gov"

WSI_ROOT = Path(
    "/data2/home/vcivale/projects/imaging/data/WSI"
)

SOURCE_DATASET_ROOT = (
    WSI_ROOT
    / "datasets/pretraining/tcga_eaf_multicohort_v1"
)

OUTPUT_DATASET_ROOT = (
    WSI_ROOT
    / "datasets/pretraining/tcga_eaf_thunder_clean_v1"
)

MANIFEST_ROOT = OUTPUT_DATASET_ROOT / "manifests"

BLOCKED_CASES_PATH = (
    WSI_ROOT
    / "catalog/thunder_overlap/all_thunder_tcga_cases.txt"
)

RESERVED_PROJECTS_PATH = (
    WSI_ROOT
    / "catalog/reserved_wsi_benchmark_projects.txt"
)

LOCAL_SLIDES_CATALOG = WSI_ROOT / "catalog/slides.csv"

TCGA_CASE_RE = re.compile(
    r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$"
)

MIN_EXPECTED_BLOCKED_CASES = 7900

REQUEST_TIMEOUT = 180
MAX_RETRIES = 5
PAGE_SIZE = 1000


OUTPUT_FIELDS = [
    "dataset_id",
    "source_provider",
    "program",
    "project_id",
    "case_id",
    "case_uuid",
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "state",
    "access",
    "data_category",
    "data_type",
    "data_format",
    "experimental_strategy",
    "sample_types",
    "sample_submitter_ids",
    "raw_destination",
    "local_raw_path",
    "local_coords_path",
    "available_locally",
    "coords_available_locally",
    "eligibility",
    "exclusion_reason",
    "download_status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_nonempty_lines(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    return {
        line.strip().upper()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
    }


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            delimiter=delimiter,
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


class GDCClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": (
                    "EntropyPruning-TCGA-clean-inventory/1.0"
                ),
            }
        )

    def post(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{GDC_API}/{endpoint.lstrip('/')}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()

                if "data" not in result:
                    raise RuntimeError(
                        f"Risposta GDC priva di data: {result}"
                    )

                return result

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
            ) as error:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Richiesta GDC fallita dopo "
                        f"{MAX_RETRIES} tentativi: {url}"
                    ) from error

                delay = min(2 ** attempt, 30)
                print(
                    f"[GDC] tentativo {attempt} fallito: "
                    f"{error!r}; retry tra {delay}s"
                )
                time.sleep(delay)

        raise AssertionError("Unreachable")


def load_local_catalog() -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    if not LOCAL_SLIDES_CATALOG.is_file():
        return {}, {}

    with LOCAL_SLIDES_CATALOG.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        rows = list(csv.DictReader(handle))

    by_file_id: dict[str, dict[str, str]] = {}
    by_file_name: dict[str, dict[str, str]] = {}

    for row in rows:
        file_id = row.get("file_id", "").strip()
        file_name = row.get("file_name", "").strip()

        if file_id:
            by_file_id[file_id] = row

        if file_name:
            by_file_name[file_name] = row

    return by_file_id, by_file_name


def list_tcga_projects(
    client: GDCClient,
) -> list[str]:
    payload = {
        "filters": {
            "op": "=",
            "content": {
                "field": "program.name",
                "value": "TCGA",
            },
        },
        "format": "JSON",
        "fields": (
            "project_id,name,primary_site,"
            "disease_type,state,released"
        ),
        "size": "100",
        "sort": "project_id:asc",
    }

    response = client.post("projects", payload)
    hits = response["data"].get("hits", [])

    projects = sorted(
        {
            str(hit["project_id"]).upper()
            for hit in hits
            if str(hit.get("project_id", "")).startswith(
                "TCGA-"
            )
        }
    )

    if len(projects) < 30:
        raise RuntimeError(
            f"Numero inatteso di progetti TCGA: "
            f"{len(projects)}"
        )

    return projects


def query_project_slides(
    client: GDCClient,
    project_id: str,
) -> list[dict[str, Any]]:
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project_id,
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_type",
                    "value": "Slide Image",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.experimental_strategy",
                    "value": "Diagnostic Slide",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_format",
                    "value": "SVS",
                },
            },
        ],
    }

    fields = ",".join(
        [
            "file_id",
            "file_name",
            "file_size",
            "md5sum",
            "state",
            "access",
            "data_category",
            "data_type",
            "data_format",
            "experimental_strategy",
            "cases.case_id",
            "cases.submitter_id",
            "cases.project.project_id",
            "cases.samples.sample_type",
            "cases.samples.submitter_id",
        ]
    )

    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        payload = {
            "filters": filters,
            "format": "JSON",
            "fields": fields,
            "size": str(PAGE_SIZE),
            "from": str(offset),
            "sort": "file_id:asc",
        }

        response = client.post("files", payload)
        data = response["data"]
        hits = data.get("hits", [])
        pagination = data.get("pagination", {})
        total = int(pagination.get("total", len(hits)))

        rows.extend(hits)
        offset += len(hits)

        if not hits or offset >= total:
            break

    return rows


def flatten_hit(
    hit: dict[str, Any],
    blocked_cases: set[str],
    reserved_projects: set[str],
    local_by_file_id: dict[str, dict[str, str]],
    local_by_file_name: dict[str, dict[str, str]],
) -> dict[str, Any]:
    file_id = str(
        hit.get("file_id")
        or hit.get("id")
        or ""
    ).strip()

    file_name = str(hit.get("file_name", "")).strip()

    cases = hit.get("cases") or []

    case_ids = sorted(
        {
            str(case.get("submitter_id", ""))
            .strip()
            .upper()
            for case in cases
            if str(case.get("submitter_id", "")).strip()
        }
    )

    case_uuids = sorted(
        {
            str(
                case.get("case_id")
                or case.get("id")
                or ""
            ).strip()
            for case in cases
            if str(
                case.get("case_id")
                or case.get("id")
                or ""
            ).strip()
        }
    )

    projects = sorted(
        {
            str(
                (case.get("project") or {}).get(
                    "project_id",
                    "",
                )
            )
            .strip()
            .upper()
            for case in cases
            if str(
                (case.get("project") or {}).get(
                    "project_id",
                    "",
                )
            ).strip()
        }
    )

    samples = [
        sample
        for case in cases
        for sample in (case.get("samples") or [])
    ]

    sample_types = sorted(
        {
            str(sample.get("sample_type", "")).strip()
            for sample in samples
            if str(sample.get("sample_type", "")).strip()
        }
    )

    sample_submitter_ids = sorted(
        {
            str(sample.get("submitter_id", "")).strip()
            for sample in samples
            if str(sample.get("submitter_id", "")).strip()
        }
    )

    local = (
        local_by_file_id.get(file_id)
        or local_by_file_name.get(file_name)
        or {}
    )

    local_raw_path = local.get("raw_path", "").strip()
    local_coords_path = local.get("coords_path", "").strip()

    raw_exists = bool(
        local_raw_path
        and (WSI_ROOT / local_raw_path).is_file()
    )

    coords_exists = bool(
        local_coords_path
        and (WSI_ROOT / local_coords_path).is_file()
    )

    reasons: list[str] = []

    if not file_id:
        reasons.append("missing_file_id")

    if not file_name:
        reasons.append("missing_file_name")

    if len(case_ids) != 1:
        reasons.append(
            f"ambiguous_case_count:{len(case_ids)}"
        )

    if len(projects) != 1:
        reasons.append(
            f"ambiguous_project_count:{len(projects)}"
        )

    case_id = case_ids[0] if len(case_ids) == 1 else ""
    case_uuid = (
        case_uuids[0]
        if len(case_uuids) == 1
        else "|".join(case_uuids)
    )
    project_id = (
        projects[0]
        if len(projects) == 1
        else ""
    )

    if case_id and not TCGA_CASE_RE.fullmatch(case_id):
        reasons.append("invalid_tcga_case_barcode")

    if case_id in blocked_cases:
        reasons.append("thunder_case_overlap")

    if project_id in reserved_projects:
        reasons.append("reserved_wsi_benchmark_project")

    metadata_errors = [
        reason
        for reason in reasons
        if reason.startswith("missing_")
        or reason.startswith("ambiguous_")
        or reason == "invalid_tcga_case_barcode"
    ]

    if metadata_errors:
        eligibility = "quarantine"
    elif reasons:
        eligibility = "exclude"
    else:
        eligibility = "eligible"

    raw_destination = (
        f"sources/gdc/tcga/{project_id}/diagnostic/"
        f"{file_name}"
        if project_id and file_name
        else ""
    )

    access = str(hit.get("access", "")).strip()

    if eligibility != "eligible":
        download_status = "not_applicable"
    elif raw_exists:
        download_status = "reuse_local"
    elif access.lower() == "open":
        download_status = "download_missing"
    else:
        download_status = "access_review_required"

    return {
        "dataset_id": "tcga_eaf_thunder_clean_v1",
        "source_provider": "GDC",
        "program": "TCGA",
        "project_id": project_id,
        "case_id": case_id,
        "case_uuid": case_uuid,
        "file_id": file_id,
        "file_name": file_name,
        "file_size": int(hit.get("file_size") or 0),
        "md5sum": str(hit.get("md5sum", "")).strip(),
        "state": str(hit.get("state", "")).strip(),
        "access": access,
        "data_category": str(
            hit.get("data_category", "")
        ).strip(),
        "data_type": str(hit.get("data_type", "")).strip(),
        "data_format": str(
            hit.get("data_format", "")
        ).strip(),
        "experimental_strategy": str(
            hit.get("experimental_strategy", "")
        ).strip(),
        "sample_types": "|".join(sample_types),
        "sample_submitter_ids": "|".join(
            sample_submitter_ids
        ),
        "raw_destination": raw_destination,
        "local_raw_path": local_raw_path,
        "local_coords_path": local_coords_path,
        "available_locally": str(raw_exists).lower(),
        "coords_available_locally": str(
            coords_exists
        ).lower(),
        "eligibility": eligibility,
        "exclusion_reason": "|".join(reasons),
        "download_status": download_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-small-blocklist",
        action="store_true",
        help=(
            "Permette di proseguire anche se la blocklist "
            "THUNDER contiene meno casi del previsto."
        ),
    )
    args = parser.parse_args()

    blocked_cases = read_nonempty_lines(
        BLOCKED_CASES_PATH
    )
    reserved_projects = read_nonempty_lines(
        RESERVED_PROJECTS_PATH
    )

    if (
        len(blocked_cases) < MIN_EXPECTED_BLOCKED_CASES
        and not args.allow_small_blocklist
    ):
        raise RuntimeError(
            f"Blocklist THUNDER troppo piccola: "
            f"{len(blocked_cases)} casi; attesi almeno "
            f"{MIN_EXPECTED_BLOCKED_CASES}."
        )

    local_by_file_id, local_by_file_name = (
        load_local_catalog()
    )

    client = GDCClient()
    projects = list_tcga_projects(client)

    print("=== GDC TCGA DIAGNOSTIC SLIDE INVENTORY ===")
    print(f"Progetti TCGA: {len(projects)}")
    print(f"Casi THUNDER bloccati: {len(blocked_cases):,}")
    print(
        "Progetti WSI riservati:",
        ", ".join(sorted(reserved_projects)),
    )

    flattened_rows: list[dict[str, Any]] = []

    for index, project_id in enumerate(
        projects,
        start=1,
    ):
        hits = query_project_slides(
            client,
            project_id,
        )

        print(
            f"[{index:02d}/{len(projects):02d}] "
            f"{project_id}: {len(hits):,} slide"
        )

        for hit in hits:
            flattened_rows.append(
                flatten_hit(
                    hit,
                    blocked_cases=blocked_cases,
                    reserved_projects=reserved_projects,
                    local_by_file_id=local_by_file_id,
                    local_by_file_name=local_by_file_name,
                )
            )

    # Deduplicazione per UUID file.
    by_file_id: dict[str, dict[str, Any]] = {}
    duplicate_file_ids: list[str] = []

    for row in flattened_rows:
        file_id = str(row["file_id"])

        if file_id in by_file_id:
            duplicate_file_ids.append(file_id)
            continue

        by_file_id[file_id] = row

    if duplicate_file_ids:
        raise RuntimeError(
            f"UUID file duplicati nella risposta GDC: "
            f"{duplicate_file_ids[:10]}"
        )

    rows = sorted(
        by_file_id.values(),
        key=lambda row: (
            str(row["project_id"]),
            str(row["case_id"]),
            str(row["file_name"]),
        ),
    )

    eligible = [
        row
        for row in rows
        if row["eligibility"] == "eligible"
    ]

    reuse_local = [
        row
        for row in eligible
        if row["download_status"] == "reuse_local"
    ]

    download_missing = [
        row
        for row in eligible
        if row["download_status"] == "download_missing"
    ]

    access_review = [
        row
        for row in eligible
        if row["download_status"]
        == "access_review_required"
    ]

    excluded_thunder = [
        row
        for row in rows
        if "thunder_case_overlap"
        in row["exclusion_reason"].split("|")
    ]

    excluded_reserved = [
        row
        for row in rows
        if "reserved_wsi_benchmark_project"
        in row["exclusion_reason"].split("|")
    ]

    quarantine = [
        row
        for row in rows
        if row["eligibility"] == "quarantine"
    ]

    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "gdc_inventory_all_diagnostic.csv": rows,
        "eligible_all.csv": eligible,
        "eligible_reuse_local.csv": reuse_local,
        "eligible_download_missing.csv": download_missing,
        "eligible_access_review.csv": access_review,
        "excluded_thunder_cases.csv": excluded_thunder,
        "excluded_reserved_projects.csv": excluded_reserved,
        "quarantine_metadata.csv": quarantine,
    }

    for filename, data in outputs.items():
        atomic_write_csv(
            MANIFEST_ROOT / filename,
            data,
            OUTPUT_FIELDS,
        )

    gdc_manifest_rows = [
        {
            "id": row["file_id"],
            "filename": row["file_name"],
            "md5": row["md5sum"],
            "size": row["file_size"],
            "state": row["state"],
        }
        for row in download_missing
    ]

    atomic_write_csv(
        MANIFEST_ROOT / "gdc_manifest_missing.tsv",
        gdc_manifest_rows,
        ["id", "filename", "md5", "size", "state"],
        delimiter="\t",
    )

    eligible_cases = {
        row["case_id"]
        for row in eligible
        if row["case_id"]
    }

    local_cases = {
        row["case_id"]
        for row in reuse_local
        if row["case_id"]
    }

    download_cases = {
        row["case_id"]
        for row in download_missing
        if row["case_id"]
    }

    project_counts = Counter(
        row["project_id"]
        for row in eligible
    )

    total_eligible_bytes = sum(
        int(row["file_size"])
        for row in eligible
    )

    missing_bytes = sum(
        int(row["file_size"])
        for row in download_missing
    )

    blocklist_sha256 = sha256_file(
        BLOCKED_CASES_PATH
    )

    dataset_yaml = f"""\
dataset_id: tcga_eaf_thunder_clean_v1
version: 1
created_utc: {utc_now()}
role: self_supervised_pretraining

source:
  provider: GDC
  program: TCGA
  query:
    data_type: Slide Image
    experimental_strategy: Diagnostic Slide
    data_format: SVS

leakage_firewall:
  unit: case
  thunder_blocked_cases: {len(blocked_cases)}
  thunder_blocklist: catalog/thunder_overlap/all_thunder_tcga_cases.txt
  thunder_blocklist_sha256: {blocklist_sha256}
  reserved_projects:
{os.linesep.join(f"    - {p}" for p in sorted(reserved_projects))}

counts:
  gdc_diagnostic_slides_total: {len(rows)}
  eligible_slides: {len(eligible)}
  eligible_cases: {len(eligible_cases)}
  reusable_local_slides: {len(reuse_local)}
  reusable_local_cases: {len(local_cases)}
  missing_download_slides: {len(download_missing)}
  missing_download_cases: {len(download_cases)}
  access_review_slides: {len(access_review)}
  quarantined_slides: {len(quarantine)}
  eligible_size_bytes: {total_eligible_bytes}
  missing_size_bytes: {missing_bytes}

policy:
  labels_used: false
  exclusion_level: patient
  multiple_slides_per_clean_case_allowed: true
  train_validation_split_level: patient
  raw_files_are_not_duplicated: true

manifests:
  inventory: manifests/gdc_inventory_all_diagnostic.csv
  eligible: manifests/eligible_all.csv
  reuse_local: manifests/eligible_reuse_local.csv
  download_missing: manifests/eligible_download_missing.csv
  gdc_download_manifest: manifests/gdc_manifest_missing.tsv
  excluded_thunder: manifests/excluded_thunder_cases.csv
  excluded_reserved: manifests/excluded_reserved_projects.csv
  quarantine: manifests/quarantine_metadata.csv
"""

    atomic_write_text(
        OUTPUT_DATASET_ROOT / "dataset.yaml",
        dataset_yaml,
    )

    print("\n=== TCGA THUNDER-CLEAN INVENTORY ===")
    print(f"Slide GDC totali:          {len(rows):,}")
    print(f"Slide eligible:            {len(eligible):,}")
    print(f"Casi eligible:             {len(eligible_cases):,}")
    print(f"Slide già locali:          {len(reuse_local):,}")
    print(f"Casi già locali:           {len(local_cases):,}")
    print(f"Slide da scaricare:        {len(download_missing):,}")
    print(f"Casi da scaricare:         {len(download_cases):,}")
    print(f"Slide access review:       {len(access_review):,}")
    print(f"Slide quarantena:          {len(quarantine):,}")
    print(
        "Spazio eligible totale:   "
        f"{total_eligible_bytes / 2**40:.3f} TiB"
    )
    print(
        "Spazio download mancante: "
        f"{missing_bytes / 2**40:.3f} TiB"
    )

    print("\n=== ELIGIBLE PER PROJECT ===")
    for project_id, count in sorted(
        project_counts.items()
    ):
        print(f"{project_id:12s} {count:6,d}")

    print("\nOutput:", OUTPUT_DATASET_ROOT)
    print(
        "GDC manifest:",
        MANIFEST_ROOT / "gdc_manifest_missing.tsv",
    )


if __name__ == "__main__":
    main()
