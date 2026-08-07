"""Acquisition adapters for the large EAF pretraining corpora HISTAI, GTEx and HEST."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from .layout import DatasetRole, StoreLayout
from .manifest import (
    SlideRecord,
    assert_no_path_under,
    group_disjoint_split,
    read_manifest,
    write_manifest,
)

HISTAI_DATASET = "histai_eaf_wsi_v1"
GTEX_DATASET = "gtex_eaf_wsi_v1"
HEST_DATASET = "hest_eaf_wsi_v1"
STRICT_DATASET = "eaf_wsi_pretrain_strict_v1"

# Discovery filters reused from the pre-refactor `wsi_prepare_strict_pretraining.py`
# scan-hest command: match raw WSI containers, skip derived/preprocessed artifacts.
HEST_WSI_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".dcm"}
HEST_EXCLUDED_PATH_PARTS = {
    "thumbnails",
    "thumbnail",
    "spatial_plots",
    "spatial_plot",
    "tissue_seg",
    "patches",
    "trident",
    "features",
}

HISTAI_SUBSETS = (
    "HISTAI-mixed",
    "HISTAI-skin-b2",
    "HISTAI-skin-b1",
    "HISTAI-colorectal-b1",
    "HISTAI-breast",
    "HISTAI-thorax",
    "HISTAI-gastrointestinal",
    "HISTAI-hematologic",
    "HISTAI-colorectal-b2",
)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty plan")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ensure_symlink(link_path: Path, target: Path) -> None:
    """Create/refresh a compatibility symlink; never copies pixel data."""
    target = target.resolve()
    if link_path.is_symlink() or link_path.exists():
        if link_path.resolve() == target:
            return
        link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)


def _histai_case_id(repo_path: str) -> str | None:
    return next((part for part in Path(repo_path).parts if part.startswith("case_")), None)


def _is_histai_he(repo_path: str) -> bool:
    return bool(
        re.match(
            r"^slide_(?:[^_]+_)?H&E_\d+\.tiff$",
            Path(repo_path).name,
            flags=re.IGNORECASE,
        )
    )


def _histai_magnification(repo_path: str) -> str:
    match = re.search(
        r"slide_([^_]+)_h&e_\d+\.tiff$",
        Path(repo_path).name.lower(),
    )
    return match.group(1) if match else ""


def _histai_rank(repo_path: str) -> tuple[int, str]:
    mag = _histai_magnification(repo_path)
    if mag in {"20x", "x20", "20"}:
        return (0, repo_path)
    if not mag:
        return (1, repo_path)
    return (2, repo_path)


def plan_histai(
    data_root: str | Path,
    *,
    token: str | None = None,
    subsets: Iterable[str] | None = None,
    force: bool = False,
) -> Path:
    """Freeze one deterministic H&E WSI per ``(subset, case_id)``.

    Incremental and gating-resilient:

    - Subsets already present in an existing ``plan.csv`` are left untouched
      (not re-listed) unless ``force=True``, so a repeated call never clobbers a
      plan that already encodes the one-H&E-per-case selection for that subset.
    - A subset whose repository cannot be listed (HF gate not yet accepted, 403,
      network error, ...) does not fail the whole plan. It is skipped and its
      status is recorded in ``manifests/histai_subset_access.json`` so it stays
      absent from the plan (i.e. effectively ``downloaded=0`` once
      ``download_histai`` is run) instead of raising for every other subset.
    """

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to plan HISTAI") from exc

    layout = StoreLayout.from_root(data_root)
    layout.ensure_base_dirs()
    chosen = tuple(subsets) if subsets else HISTAI_SUBSETS
    invalid = sorted(set(chosen) - set(HISTAI_SUBSETS))
    if invalid:
        raise ValueError(f"Unknown HISTAI subsets: {invalid}")

    dataset_dir = layout.dataset_dir(DatasetRole.PRETRAINING, HISTAI_DATASET)
    plan = dataset_dir / "manifests" / "plan.csv"
    access_path = dataset_dir / "manifests" / "histai_subset_access.json"

    existing_rows = _read_rows(plan) if plan.exists() else []
    existing_subsets = {row["subset"] for row in existing_rows}
    to_list = list(chosen) if force else [s for s in chosen if s not in existing_subsets]

    if not to_list:
        print(
            f"[eaf-data] HISTAI plan already covers {sorted(chosen)}; "
            f"not regenerating {plan} (pass force=True to re-list)",
            flush=True,
        )
        return plan if plan.exists() else _write_rows(plan, existing_rows)

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    access: dict[str, dict] = (
        json.loads(access_path.read_text()) if access_path.exists() else {}
    )
    kept_rows = [row for row in existing_rows if row["subset"] not in to_list]
    new_rows: list[dict[str, str]] = []
    for subset in to_list:
        repo_id = f"histai/{subset}"
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        except Exception as exc:  # gated/not-yet-accessible subset: skip, don't fail the run
            access[subset] = {"accessible": False, "error": repr(exc)}
            print(f"[eaf-data] {subset}: inaccessible ({exc}); left out of plan", flush=True)
            continue

        by_case: dict[str, list[str]] = defaultdict(list)
        for repo_path in files:
            case_id = _histai_case_id(repo_path)
            if case_id and _is_histai_he(repo_path):
                by_case[case_id].append(repo_path)
        for case_id in sorted(by_case):
            repo_path = min(by_case[case_id], key=_histai_rank)
            new_rows.append(
                {
                    "source": "histai",
                    "subset": subset,
                    "repo_id": repo_id,
                    "case_id": case_id,
                    "repo_path": repo_path,
                    "native_magnification": _histai_magnification(repo_path),
                }
            )
        access[subset] = {"accessible": True, "n_cases": len(by_case)}
        print(
            f"[eaf-data] {subset}: {len(by_case)} H&E cases selected",
            flush=True,
        )

    access_path.parent.mkdir(parents=True, exist_ok=True)
    access_path.write_text(json.dumps(access, indent=2, sort_keys=True) + "\n")

    rows = kept_rows + new_rows
    if not rows:
        raise RuntimeError(
            "No HISTAI subset was accessible; see "
            f"{access_path} for per-subset errors"
        )
    return _write_rows(plan, rows)


def download_histai(
    data_root: str | Path,
    *,
    subsets: Iterable[str] | None = None,
    workers: int = 4,
    token: str | None = None,
) -> Path:
    """Incrementally download only planned HISTAI subsets with a global progress bar."""

    try:
        from huggingface_hub import hf_hub_download
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub, hf_xet and tqdm") from exc

    layout = StoreLayout.from_root(data_root)
    dataset_dir = layout.dataset_dir(DatasetRole.PRETRAINING, HISTAI_DATASET)
    plan = dataset_dir / "manifests" / "plan.csv"
    if not plan.exists():
        raise FileNotFoundError(f"Missing HISTAI plan: {plan}")
    all_rows = _read_rows(plan)
    requested = set(subsets or ())
    rows = [row for row in all_rows if not requested or row["subset"] in requested]
    if not rows:
        raise ValueError("No HISTAI rows selected")

    auth = token or os.environ.get("HF_TOKEN")
    failures: list[str] = []

    def download_one(row: dict[str, str]) -> None:
        local_dir = layout.sources / "histai" / row["subset"]
        expected = local_dir / row["repo_path"]
        if expected.is_file():
            return
        local_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=row["repo_id"],
            repo_type="dataset",
            filename=row["repo_path"],
            local_dir=local_dir,
            token=auth,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_rows = {pool.submit(download_one, row): row for row in rows}
        for future in tqdm(
            as_completed(future_rows), total=len(future_rows), desc="HISTAI", unit="WSI"
        ):
            row = future_rows[future]
            try:
                future.result()
            except Exception as exc:  # keep the rest of an incremental run alive
                failures.append(f"{row['repo_id']}:{row['repo_path']}::{exc}")

    views_raw_flat = dataset_dir / "views" / "raw_flat"
    records: list[SlideRecord] = []
    for row in all_rows:
        local = layout.sources / "histai" / row["subset"] / row["repo_path"]
        slide_stem = Path(row["repo_path"]).stem
        slide_id = f"{row['subset']}__{row['case_id']}__{slide_stem}"
        exists = local.is_file()
        if exists:
            # `slide_H&E_0.tiff` repeats across cases; namespace the compatibility
            # view by subset + case_id to avoid raw_flat collisions (matches the
            # pre-refactor script). `raw_path` below still points at the single
            # physical copy under sources/, the view is a convenience symlink only.
            _ensure_symlink(views_raw_flat / f"{slide_id}{local.suffix}", local)
        records.append(
            SlideRecord(
                slide_id=slide_id,
                case_id=f"{row['subset']}::{row['case_id']}",
                source="histai",
                cohort=row["subset"],
                subset=row["subset"],
                raw_path=str(local),
                downloaded="1" if exists else "0",
                metadata_json=(
                    '{"repo_id":"%s","repo_path":"%s","native_magnification":"%s"}'
                    % (row["repo_id"], row["repo_path"], row["native_magnification"])
                ),
            )
        )
    manifest = write_manifest(dataset_dir / "manifests" / "slides.csv", records)
    if failures:
        failure_path = dataset_dir / "manifests" / "download_failures.txt"
        failure_path.write_text("\n".join(failures) + "\n")
        print(f"[eaf-data] HISTAI failures: {len(failures)} -> {failure_path}")
    return manifest


def plan_gtex(data_root: str | Path, *, max_series: int | None = None) -> Path:
    """Freeze the IDC GTEx Slide Microscopy series list.

    The unit of a WSI is ``SeriesInstanceUID``. ``PatientID`` is retained as the
    donor-level grouping key. Tissue metadata can be joined later without changing the
    series identity.
    """

    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise RuntimeError("Install idc-index to plan GTEx") from exc

    layout = StoreLayout.from_root(data_root)
    layout.ensure_base_dirs()
    client = IDCClient.client()
    query = """
SELECT PatientID, StudyInstanceUID, SeriesInstanceUID, series_size_MB
FROM index
WHERE collection_id = 'gtex' AND Modality = 'SM'
ORDER BY PatientID, SeriesInstanceUID
"""
    frame = client.sql_query(query)
    if max_series is not None:
        frame = frame.head(max_series)
    rows = [
        {
            "source": "gtex",
            "patient_id": str(row.PatientID),
            "study_uid": str(row.StudyInstanceUID),
            "series_uid": str(row.SeriesInstanceUID),
            "series_size_MB": str(float(row.series_size_MB)),
        }
        for row in frame.itertuples(index=False)
    ]
    plan = (
        layout.dataset_dir(DatasetRole.PRETRAINING, GTEX_DATASET)
        / "manifests"
        / "plan.csv"
    )
    return _write_rows(plan, rows)


def _find_gtex_representative(source_root: Path, series_uid: str) -> Path | None:
    matches = sorted(source_root.glob(f"**/SM_{series_uid}/*.dcm"))
    return matches[0] if matches else None


def download_gtex(
    data_root: str | Path,
    *,
    workers: int = 2,
    limit: int | None = None,
) -> Path:
    """Incrementally download planned GTEx DICOM series using the official IDC CLI."""

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("Install tqdm") from exc

    layout = StoreLayout.from_root(data_root)
    dataset_dir = layout.dataset_dir(DatasetRole.PRETRAINING, GTEX_DATASET)
    plan = dataset_dir / "manifests" / "plan.csv"
    if not plan.exists():
        raise FileNotFoundError(f"Missing GTEx plan: {plan}")
    all_rows = _read_rows(plan)
    rows = all_rows[:limit] if limit is not None else all_rows
    source_root = layout.sources / "gtex"
    source_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    def download_one(row: dict[str, str]) -> None:
        if _find_gtex_representative(source_root, row["series_uid"]):
            return
        subprocess.run(
            [
                "idc",
                "download",
                row["series_uid"],
                "--download-dir",
                str(source_root),
            ],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_rows = {pool.submit(download_one, row): row for row in rows}
        for future in tqdm(
            as_completed(future_rows), total=len(future_rows), desc="GTEx", unit="WSI"
        ):
            row = future_rows[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{row['series_uid']}::{exc}")

    records: list[SlideRecord] = []
    for row in all_rows:
        representative = _find_gtex_representative(source_root, row["series_uid"])
        records.append(
            SlideRecord(
                slide_id=row["series_uid"],
                case_id=row["patient_id"],
                patient_id=row["patient_id"],
                source="gtex",
                cohort="GTEx",
                subset="IDC",
                raw_path=str(representative or ""),
                study_uid=row["study_uid"],
                series_uid=row["series_uid"],
                downloaded="1" if representative else "0",
                metadata_json='{"series_size_MB":%s}' % row["series_size_MB"],
            )
        )
    manifest = write_manifest(dataset_dir / "manifests" / "slides.csv", records)
    if failures:
        failure_path = dataset_dir / "manifests" / "download_failures.txt"
        failure_path.write_text("\n".join(failures) + "\n")
        print(f"[eaf-data] GTEx failures: {len(failures)} -> {failure_path}")
    return manifest


def _discover_hest_wsi(root: Path) -> list[Path]:
    """Find raw HEST WSI containers under ``root`` without copying anything."""
    root = root.expanduser().resolve()
    candidates: list[Path] = []
    preferred: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in HEST_WSI_SUFFIXES:
            continue
        lower_parts = {part.lower() for part in path.parts}
        if lower_parts & HEST_EXCLUDED_PATH_PARTS:
            continue
        candidates.append(path)
        if "wsis" in lower_parts or "raw_wsi" in lower_parts:
            preferred.append(path)
    return sorted(preferred if preferred else candidates)


def register_hest(
    data_root: str | Path,
    hest_root: str | Path,
    *,
    dataset_name: str = HEST_DATASET,
) -> Path:
    """Register an existing HEST raw-WSI tree in place, without copying it.

    This is a lightweight sibling to the mature, already-built
    ``hest_eaf_thunder_clean_v1`` corpus (left untouched); it exists so
    ``eaf_wsi_pretrain_strict_v1`` can reference HEST slides without requiring
    that separate, more elaborate pipeline to run first.
    """

    layout = StoreLayout.from_root(data_root)
    hest_root = Path(hest_root).expanduser().resolve()
    if not hest_root.exists():
        raise FileNotFoundError(f"Existing HEST root does not exist: {hest_root}")

    slides = _discover_hest_wsi(hest_root)
    if not slides:
        raise RuntimeError(f"No WSI files discovered under {hest_root}")

    records = [
        SlideRecord(
            slide_id=slide.stem,
            # Replace with a real patient/study grouping if a HEST clinical
            # mapping becomes available; one slide per case is the safe default.
            case_id=slide.stem,
            source="hest",
            cohort="hest",
            subset="existing",
            raw_path=str(slide),
            downloaded="1",
        )
        for slide in slides
    ]
    dataset_dir = layout.dataset_dir(DatasetRole.PRETRAINING, dataset_name)
    manifest = write_manifest(dataset_dir / "manifests" / "slides.csv", records)
    print(f"[eaf-data] HEST registered read-only: {hest_root} ({len(records)} WSIs) -> {manifest}")
    return manifest


def build_strict_corpus(
    data_root: str | Path,
    *,
    tcga_root: str | Path | None = None,
    sources: Iterable[str] = ("histai", "gtex", "hest"),
    seed: int = 17,
) -> Path:
    """Build ``eaf_wsi_pretrain_strict_v1``: a dataset-disjoint union of
    downloaded/registered pretraining sources, with TCGA excluded by construction.

    The EAF training corpus policy is HISTAI + GTEx + HEST only; TCGA is never
    included here even though it is registered/preserved elsewhere on disk
    (see "WSI EAF Training Data Policy" in CLAUDE.md). A source missing its
    manifest (e.g. GTEx before it has been planned/downloaded) simply
    contributes zero rows rather than failing the whole build.

    Only rows with ``downloaded == "1"`` and an existing ``raw_path`` are included.
    No pixels are copied or moved; the strict manifest references the single
    physical copy already registered under each source's own dataset directory.
    """

    layout = StoreLayout.from_root(data_root)
    source_datasets = {
        "histai": HISTAI_DATASET,
        "hest": HEST_DATASET,
        "gtex": GTEX_DATASET,
    }
    requested = list(sources)
    unknown = sorted(set(requested) - set(source_datasets))
    if unknown:
        raise ValueError(f"Unknown strict-corpus sources: {unknown}")

    if tcga_root is None:
        default_tcga = layout.sources / "gdc" / "tcga"
        tcga_root = default_tcga if default_tcga.exists() else None

    rows: list[SlideRecord] = []
    seen_paths: set[Path] = set()
    included_counts: dict[str, int] = {}
    for name in requested:
        manifest_path = layout.manifest_path(DatasetRole.PRETRAINING, source_datasets[name])
        if not manifest_path.exists():
            included_counts[name] = 0
            continue
        n_included = 0
        for record in read_manifest(manifest_path):
            if record.downloaded != "1" or not record.raw_path:
                continue
            raw_path = Path(record.raw_path).expanduser().resolve()
            if not raw_path.is_file():
                continue
            if raw_path in seen_paths:
                continue
            seen_paths.add(raw_path)
            rows.append(record)
            n_included += 1
        included_counts[name] = n_included

    if not rows:
        raise RuntimeError(
            "No downloaded/registered slides found across "
            f"{requested}; run plan/download or register-hest first"
        )

    # Leakage guard: fail loudly rather than silently pulling preserved TCGA into
    # the strict pretraining union. TCGA itself is never read/written here.
    assert_no_path_under(rows, tcga_root, label="preserved TCGA root")

    split_rows = group_disjoint_split(rows, seed=seed)
    strict_dir = layout.dataset_dir(DatasetRole.PRETRAINING, STRICT_DATASET)
    manifest = write_manifest(strict_dir / "manifests" / "slides.csv", split_rows)

    by_split: dict[str, int] = defaultdict(int)
    for row in split_rows:
        by_split[row.split] += 1
    provenance = {
        "sources_included": included_counts,
        "n_total": len(split_rows),
        "by_split": dict(by_split),
        "tcga_root_excluded": str(tcga_root) if tcga_root else None,
        "seed": seed,
    }
    (strict_dir / "manifests" / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    print(f"[eaf-data] strict manifest: {manifest} ({len(split_rows)} WSIs)")
    for name, count in sorted(included_counts.items()):
        print(f"  - {name}: {count}")
    for split, count in sorted(by_split.items()):
        print(f"  - {split}: {count}")
    if tcga_root:
        print(f"[eaf-data] verified: no strict-v1 path is under preserved TCGA root {tcga_root}")
    return manifest
