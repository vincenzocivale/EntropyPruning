"""Discovery, planning, and download helpers for public unlabeled WSI cohorts.

The module intentionally separates discovery from download. Discovery writes an
immutable JSON plan with exact item identifiers and byte estimates. Downloading
then consumes that plan, which makes large acquisitions reviewable, resumable,
and reproducible.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
PLAN_SCHEMA_VERSION = 1


class AcquisitionError(RuntimeError):
    """Raised when a discovery or acquisition request cannot be completed."""


@dataclass(frozen=True)
class AcquisitionItem:
    provider: str
    item_id: str
    file_name: str
    size_bytes: int
    cohort: str = ""
    case_id: str = ""
    study_uid: str = ""
    md5: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionPlan:
    provider: str
    source_id: str
    filters: dict[str, Any]
    items: list[AcquisitionItem]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: int = PLAN_SCHEMA_VERSION

    @property
    def total_bytes(self) -> int:
        return sum(max(0, item.size_bytes) for item in self.items)

    @property
    def cohorts(self) -> dict[str, int]:
        return dict(sorted(Counter(item.cohort or "unknown" for item in self.items).items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "source_id": self.source_id,
            "created_at": self.created_at,
            "filters": self.filters,
            "summary": {
                "n_items": len(self.items),
                "total_bytes": self.total_bytes,
                "total_gib": self.total_bytes / 2**30,
                "cohorts": self.cohorts,
            },
            "items": [asdict(item) for item in self.items],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AcquisitionPlan":
        version = int(payload.get("schema_version", 0))
        if version != PLAN_SCHEMA_VERSION:
            raise AcquisitionError(
                f"Unsupported acquisition plan schema {version}; expected {PLAN_SCHEMA_VERSION}."
            )
        return cls(
            provider=str(payload["provider"]),
            source_id=str(payload.get("source_id", payload["provider"])),
            created_at=str(payload.get("created_at", "")),
            filters=dict(payload.get("filters", {})),
            items=[AcquisitionItem(**item) for item in payload.get("items", [])],
            schema_version=version,
        )


def human_bytes(value: int) -> str:
    value_f = float(max(0, value))
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    unit = units[0]
    for unit in units:
        if value_f < 1024.0 or unit == units[-1]:
            break
        value_f /= 1024.0
    return f"{value_f:.2f} {unit}"


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", 0)) != 1:
        raise AcquisitionError(f"Unsupported catalog schema in {path}")
    return payload


def save_plan(plan: AcquisitionPlan, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "acquisition_plan.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_plan(path: Path) -> AcquisitionPlan:
    with path.open(encoding="utf-8") as handle:
        return AcquisitionPlan.from_dict(json.load(handle))


def _gdc_filters(projects: Sequence[str] | None, diagnostic_only: bool) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "op": "in",
            "content": {"field": "files.data_category", "value": ["Biospecimen"]},
        },
        {
            "op": "in",
            "content": {"field": "files.data_type", "value": ["Slide Image"]},
        },
        {
            "op": "in",
            "content": {"field": "files.data_format", "value": ["SVS"]},
        },
        {
            "op": "in",
            "content": {"field": "files.access", "value": ["open"]},
        },
    ]
    if projects:
        content.append(
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": list(projects),
                },
            }
        )
    if diagnostic_only:
        content.append(
            {
                "op": "in",
                "content": {
                    "field": "files.experimental_strategy",
                    "value": ["Diagnostic Slide"],
                },
            }
        )
    return {"op": "and", "content": content}


def _first_case(hit: Mapping[str, Any]) -> Mapping[str, Any]:
    cases = hit.get("cases") or []
    return cases[0] if cases else {}


def _normalize_gdc_hit(hit: Mapping[str, Any]) -> AcquisitionItem:
    case = _first_case(hit)
    project = case.get("project") or {}
    samples = case.get("samples") or []
    sample_types = sorted(
        {str(sample.get("sample_type", "")) for sample in samples if sample.get("sample_type")}
    )
    return AcquisitionItem(
        provider="gdc",
        item_id=str(hit["file_id"]),
        file_name=str(hit["file_name"]),
        size_bytes=int(hit.get("file_size") or 0),
        cohort=str(project.get("project_id", "")),
        case_id=str(case.get("submitter_id", "")),
        md5=str(hit.get("md5sum", "")),
        metadata={
            "experimental_strategy": hit.get("experimental_strategy", ""),
            "sample_types": sample_types,
            "state": hit.get("state", ""),
        },
    )


def discover_gdc(
    *,
    projects: Sequence[str] | None = None,
    all_tcga: bool = False,
    diagnostic_only: bool = True,
    page_size: int = 1000,
    timeout: int = 120,
    session: Any | None = None,
) -> list[AcquisitionItem]:
    """Return all matching open-access SVS files from the GDC Files API."""
    if projects and all_tcga:
        raise AcquisitionError("Use either explicit projects or all_tcga, not both.")
    if page_size <= 0:
        raise AcquisitionError("page_size must be positive")

    if session is None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
            raise AcquisitionError("GDC discovery requires requests>=2.31") from exc
        session = requests.Session()

    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "state",
        "experimental_strategy",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.sample_type",
    ]
    offset = 0
    total: int | None = None
    items: list[AcquisitionItem] = []
    filters = _gdc_filters(projects, diagnostic_only)

    while total is None or offset < total:
        payload = {
            "filters": filters,
            "fields": ",".join(fields),
            "format": "JSON",
            "from": offset,
            "size": page_size,
            "sort": "file_id:asc",
        }
        response = session.post(GDC_FILES_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json().get("data", {})
        hits = body.get("hits", [])
        pagination = body.get("pagination", {})
        total = int(pagination.get("total", len(hits)))
        if not hits:
            break
        for hit in hits:
            item = _normalize_gdc_hit(hit)
            if all_tcga and not item.cohort.startswith("TCGA-"):
                continue
            items.append(item)
        offset += len(hits)

    deduped = {item.item_id: item for item in items}
    return [deduped[key] for key in sorted(deduped)]


def discover_idc_pathology_collections(client: Any | None = None) -> list[dict[str, Any]]:
    """List IDC collections containing Slide Microscopy (DICOM Modality=SM)."""
    if client is None:
        client = _make_idc_client()
    frame = client.sql_query(
        """
        SELECT
          collection_id,
          COUNT(*) AS n_series,
          SUM(series_size_MB) AS approximate_size_MB
        FROM (
          SELECT DISTINCT collection_id, SeriesInstanceUID, series_size_MB
          FROM index
          WHERE Modality = 'SM'
        )
        GROUP BY collection_id
        ORDER BY collection_id
        """
    )
    return frame.to_dict(orient="records")


def _make_idc_client() -> Any:
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise AcquisitionError(
            "IDC support requires idc-index. Install requirements-wsi-acquisition.txt."
        ) from exc
    return IDCClient.client()


def _sql_string_list(values: Sequence[str]) -> str:
    if not values:
        raise AcquisitionError("At least one IDC collection is required.")
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def discover_idc(
    *, collections: Sequence[str], client: Any | None = None
) -> list[AcquisitionItem]:
    """Return DICOM WSI series from selected IDC collections."""
    if client is None:
        client = _make_idc_client()
    collection_sql = _sql_string_list([value.lower() for value in collections])
    frame = client.sql_query(
        f"""
        SELECT DISTINCT
          collection_id,
          PatientID,
          StudyInstanceUID,
          SeriesInstanceUID,
          series_size_MB
        FROM index
        WHERE Modality = 'SM'
          AND lower(collection_id) IN ({collection_sql})
        ORDER BY collection_id, PatientID, SeriesInstanceUID
        """
    )
    items: list[AcquisitionItem] = []
    for row in frame.to_dict(orient="records"):
        series_uid = str(row["SeriesInstanceUID"])
        size_mb = float(row.get("series_size_MB") or 0.0)
        items.append(
            AcquisitionItem(
                provider="idc",
                item_id=series_uid,
                file_name=f"{series_uid}.dicom-series",
                size_bytes=int(round(size_mb * 1_000_000)),
                cohort=str(row.get("collection_id", "")),
                case_id=str(row.get("PatientID", "")),
                study_uid=str(row.get("StudyInstanceUID", "")),
                metadata={"modality": "SM", "series_size_MB": size_mb},
            )
        )
    return items


def select_budgeted_items(
    items: Sequence[AcquisitionItem],
    *,
    max_items: int | None = None,
    max_bytes: int | None = None,
    per_cohort: int | None = None,
    seed: int = 0,
) -> list[AcquisitionItem]:
    """Deterministically select a cohort-balanced subset under count/byte limits."""
    if max_items is not None and max_items < 0:
        raise AcquisitionError("max_items cannot be negative")
    if max_bytes is not None and max_bytes < 0:
        raise AcquisitionError("max_bytes cannot be negative")
    if per_cohort is not None and per_cohort < 0:
        raise AcquisitionError("per_cohort cannot be negative")

    groups: dict[str, list[AcquisitionItem]] = defaultdict(list)
    for item in items:
        groups[item.cohort or "unknown"].append(item)

    rng = random.Random(seed)
    for cohort in groups:
        groups[cohort] = sorted(groups[cohort], key=lambda item: item.item_id)
        rng.shuffle(groups[cohort])
        if per_cohort is not None:
            groups[cohort] = groups[cohort][:per_cohort]

    selected: list[AcquisitionItem] = []
    selected_bytes = 0
    positions = {cohort: 0 for cohort in groups}
    cohorts = sorted(groups)

    while cohorts:
        progressed = False
        next_cohorts: list[str] = []
        for cohort in cohorts:
            position = positions[cohort]
            if position >= len(groups[cohort]):
                continue
            item = groups[cohort][position]
            positions[cohort] += 1
            next_cohorts.append(cohort)

            if max_items is not None and len(selected) >= max_items:
                return selected
            if max_bytes is not None and selected_bytes + item.size_bytes > max_bytes:
                continue
            selected.append(item)
            selected_bytes += item.size_bytes
            progressed = True
        cohorts = next_cohorts
    return selected


def write_provider_manifests(plan: AcquisitionPlan, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    slides_path = output_dir / "slides.csv"
    with slides_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "slide_id",
                "provider",
                "source_id",
                "cohort",
                "case_id",
                "item_id",
                "file_name",
                "size_bytes",
                "study_uid",
            ]
        )
        for item in plan.items:
            writer.writerow(
                [
                    Path(item.file_name).stem,
                    item.provider,
                    plan.source_id,
                    item.cohort,
                    item.case_id,
                    item.item_id,
                    item.file_name,
                    item.size_bytes,
                    item.study_uid,
                ]
            )
    artifacts["slides_csv"] = slides_path

    if plan.provider == "gdc":
        manifest_path = output_dir / "gdc_manifest.tsv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["id", "filename", "md5", "size", "state"])
            for item in plan.items:
                writer.writerow(
                    [item.item_id, item.file_name, item.md5, item.size_bytes, "released"]
                )
        artifacts["gdc_manifest"] = manifest_path
    elif plan.provider == "idc":
        series_path = output_dir / "idc_series.csv"
        with series_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["collection_id", "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "size_bytes"]
            )
            for item in plan.items:
                writer.writerow(
                    [item.cohort, item.case_id, item.study_uid, item.item_id, item.size_bytes]
                )
        artifacts["idc_series"] = series_path
    return artifacts


def build_plan(
    *,
    provider: str,
    source_id: str,
    all_items: Sequence[AcquisitionItem],
    filters: Mapping[str, Any],
    max_items: int | None,
    max_gib: float | None,
    per_cohort: int | None,
    seed: int,
) -> AcquisitionPlan:
    max_bytes = None if max_gib is None else int(max_gib * 2**30)
    selected = select_budgeted_items(
        all_items,
        max_items=max_items,
        max_bytes=max_bytes,
        per_cohort=per_cohort,
        seed=seed,
    )
    plan_filters = dict(filters)
    plan_filters.update(
        {
            "max_items": max_items,
            "max_gib": max_gib,
            "per_cohort": per_cohort,
            "seed": seed,
            "discovered_items": len(all_items),
            "discovered_bytes": sum(item.size_bytes for item in all_items),
        }
    )
    return AcquisitionPlan(
        provider=provider,
        source_id=source_id,
        filters=plan_filters,
        items=selected,
    )


def download_gdc(
    plan: AcquisitionPlan,
    *,
    plan_path: Path,
    output_dir: Path,
    processes: int = 4,
    gdc_client: str = "gdc-client",
) -> None:
    if plan.provider != "gdc":
        raise AcquisitionError("The supplied plan is not a GDC plan.")
    executable = shutil.which(gdc_client)
    if executable is None:
        raise AcquisitionError(
            "gdc-client was not found. Install the official GDC Data Transfer Tool, "
            "or use scripts/download_gdc_manifest_simple.py with the generated manifest."
        )
    manifest = plan_path.parent / "gdc_manifest.tsv"
    if not manifest.exists():
        write_provider_manifests(plan, plan_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "download",
        "-m",
        str(manifest),
        "-d",
        str(output_dir),
        "-n",
        str(processes),
    ]
    subprocess.run(command, check=True)


def download_idc(plan: AcquisitionPlan, *, output_dir: Path, client: Any | None = None) -> None:
    if plan.provider != "idc":
        raise AcquisitionError("The supplied plan is not an IDC plan.")
    if client is None:
        client = _make_idc_client()
    output_dir.mkdir(parents=True, exist_ok=True)
    client.download_dicom_series(
        seriesInstanceUID=[item.item_id for item in plan.items],
        downloadDir=str(output_dir),
    )


def print_plan_summary(plan: AcquisitionPlan) -> str:
    discovered_items = plan.filters.get("discovered_items")
    discovered_bytes = int(plan.filters.get("discovered_bytes") or 0)
    lines = [
        f"Provider: {plan.provider}",
        f"Source: {plan.source_id}",
        f"Selected: {len(plan.items):,} items / {human_bytes(plan.total_bytes)}",
    ]
    if discovered_items is not None:
        lines.append(
            f"Discovered before budget: {int(discovered_items):,} items / {human_bytes(discovered_bytes)}"
        )
    lines.append("Cohorts: " + ", ".join(f"{key}={value}" for key, value in plan.cohorts.items()))
    return "\n".join(lines)
