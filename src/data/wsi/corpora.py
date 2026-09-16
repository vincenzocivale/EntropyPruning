"""HISTAI-only acquisition for task-agnostic EAF training."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from .layout import DatasetRole, StoreLayout
from .manifest import SlideRecord, write_manifest

HISTAI_DATASET = "histai_eaf_wsi_v1"

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
        raise ValueError("Cannot write an empty HISTAI plan")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ensure_symlink(link_path: Path, target: Path) -> None:
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
    match = re.search(r"slide_([^_]+)_h&e_\d+\.tiff$", Path(repo_path).name.lower())
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
    """Freeze one deterministic H&E WSI per HISTAI case."""
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
        return plan

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    access = json.loads(access_path.read_text()) if access_path.exists() else {}
    kept_rows = [row for row in existing_rows if row["subset"] not in to_list]
    new_rows: list[dict[str, str]] = []
    for subset in to_list:
        repo_id = f"histai/{subset}"
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        except Exception as exc:
            access[subset] = {"accessible": False, "error": repr(exc)}
            print(f"[eaf-data] {subset}: inaccessible ({exc}); skipped", flush=True)
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
        print(f"[eaf-data] {subset}: {len(by_case)} H&E cases selected", flush=True)

    access_path.parent.mkdir(parents=True, exist_ok=True)
    access_path.write_text(json.dumps(access, indent=2, sort_keys=True) + "\n")
    rows = kept_rows + new_rows
    if not rows:
        raise RuntimeError(f"No HISTAI subset was accessible; see {access_path}")
    return _write_rows(plan, rows)


def download_histai(
    data_root: str | Path,
    *,
    subsets: Iterable[str] | None = None,
    workers: int = 4,
    token: str | None = None,
) -> Path:
    """Incrementally download only the H&E WSI frozen by ``plan_histai``."""
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
        for future in tqdm(as_completed(future_rows), total=len(future_rows), desc="HISTAI", unit="WSI"):
            row = future_rows[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{row['repo_id']}:{row['repo_path']}::{exc}")

    views_raw_flat = dataset_dir / "views" / "raw_flat"
    records: list[SlideRecord] = []
    for row in all_rows:
        local = layout.sources / "histai" / row["subset"] / row["repo_path"]
        slide_stem = Path(row["repo_path"]).stem
        slide_id = f"{row['subset']}__{row['case_id']}__{slide_stem}"
        exists = local.is_file()
        if exists:
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
