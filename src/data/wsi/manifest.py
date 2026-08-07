"""Canonical slide manifest helpers shared by pretraining and downstream datasets."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterable


@dataclass
class SlideRecord:
    slide_id: str
    case_id: str
    source: str
    cohort: str = ""
    subset: str = ""
    stain: str = "H&E"
    raw_path: str = ""
    coords_path: str = ""
    split: str = ""
    tissue: str = ""
    patient_id: str = ""
    study_uid: str = ""
    series_uid: str = ""
    downloaded: str = "1"
    metadata_json: str = "{}"

    def metadata(self) -> dict:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metadata_json for {self.slide_id}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"metadata_json must be an object for {self.slide_id}")
        return value


MANIFEST_FIELDS = [field.name for field in fields(SlideRecord)]


def write_manifest(path: str | Path, records: Iterable[SlideRecord]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    validate_manifest(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path


def read_manifest(path: str | Path) -> list[SlideRecord]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        unknown = set(reader.fieldnames or ()) - set(MANIFEST_FIELDS)
        if unknown:
            # Preserve forward compatibility by ignoring unknown columns.
            pass
        records = []
        for row in reader:
            records.append(
                SlideRecord(**{name: row.get(name, "") for name in MANIFEST_FIELDS})
            )
    validate_manifest(records)
    return records


def validate_manifest(records: Iterable[SlideRecord]) -> None:
    seen_slide_keys: set[tuple[str, str, str]] = set()
    seen_series: set[tuple[str, str]] = set()
    for row in records:
        if not row.slide_id or not row.case_id or not row.source:
            raise ValueError("slide_id, case_id and source are required")
        key = (row.source, row.subset, row.slide_id)
        if key in seen_slide_keys:
            raise ValueError(f"Duplicate slide key: {key}")
        seen_slide_keys.add(key)
        if row.series_uid:
            series_key = (row.source, row.series_uid)
            if series_key in seen_series:
                raise ValueError(f"Duplicate series UID: {series_key}")
            seen_series.add(series_key)
        row.metadata()


def assert_dataset_disjoint(
    pretraining: Iterable[SlideRecord], downstream: Iterable[SlideRecord]
) -> None:
    """Fail on exact identifiers/paths shared by strict pretraining and benchmarks."""

    downstream_ids = {
        (row.source, row.subset, row.slide_id) for row in downstream
    }
    downstream_paths = {
        str(Path(row.raw_path).expanduser().resolve())
        for row in downstream
        if row.raw_path
    }
    downstream_series = {row.series_uid for row in downstream if row.series_uid}

    collisions: list[str] = []
    for row in pretraining:
        key = (row.source, row.subset, row.slide_id)
        if key in downstream_ids:
            collisions.append(f"id:{key}")
        if row.raw_path:
            path = str(Path(row.raw_path).expanduser().resolve())
            if path in downstream_paths:
                collisions.append(f"path:{path}")
        if row.series_uid and row.series_uid in downstream_series:
            collisions.append(f"series:{row.series_uid}")
    if collisions:
        preview = ", ".join(collisions[:10])
        raise ValueError(f"Pretraining/downstream overlap detected: {preview}")


def assert_no_path_under(
    records: Iterable[SlideRecord], forbidden_root: str | Path | None, *, label: str = "excluded root"
) -> None:
    """Leakage guard: fail if any ``raw_path`` resolves inside ``forbidden_root``.

    Used to guarantee a preserved corpus (e.g. TCGA) is never silently pulled into
    the strict EAF pretraining union, without ever touching the preserved data itself.
    """

    if forbidden_root is None:
        return
    forbidden = Path(forbidden_root).expanduser().resolve()
    for row in records:
        if not row.raw_path:
            continue
        path = Path(row.raw_path).expanduser().resolve()
        try:
            path.relative_to(forbidden)
        except ValueError:
            continue
        raise ValueError(
            f"Leakage guard: {label} contains {row.source}:{row.slide_id} -> {path}"
        )


def group_disjoint_split(
    records: Iterable[SlideRecord],
    *,
    seed: int = 17,
    val_frac: float = 0.05,
    holdout_frac: float = 0.05,
) -> list[SlideRecord]:
    """Assign train/val/holdout deterministically, grouped by ``(source, case_id)``.

    Every slide belonging to the same case lands in the same split, so downstream
    evaluation never leaks a case across splits. Returns new records with ``split``
    populated; input records are not mutated.
    """

    rows = list(records)
    by_group: dict[str, list[SlideRecord]] = defaultdict(list)
    for row in rows:
        by_group[f"{row.source}::{row.case_id}"].append(row)

    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_val = max(1, round(n * val_frac)) if n >= 20 else max(1, n // 10)
    n_holdout = max(1, round(n * holdout_frac)) if n >= 20 else max(1, n // 10)
    n_train = max(0, n - n_val - n_holdout)
    assignment = {
        **{g: "train" for g in groups[:n_train]},
        **{g: "val" for g in groups[n_train : n_train + n_val]},
        **{g: "holdout" for g in groups[n_train + n_val :]},
    }

    out: list[SlideRecord] = []
    for group, items in by_group.items():
        split = assignment[group]
        out.extend(replace(row, split=split) for row in items)
    return out
