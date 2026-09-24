#!/usr/bin/env python
"""Join Patho-Bench cptac_* split TSVs (case_id,label) with the locally
downloaded CPTAC WSI inventory (custom_list_of_wsis.csv: cohort/case_id/.../<uuid>.dcm)
and emit <labels-root>/<cohort>/labels/<task>.csv (slide_id,label) in the format
evaluate_wsi_downstream.py's discover_tasks() expects. slide_id = the Trident
patch h5 stem (== the DICOM instance uuid), matching what cache_tile/cache_wsi_teacher
will use once CPTAC caching is built.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/data2/home/vcivale/data/WSI/datasets/downstream/wsi_level")
CPTAC_ROOT = ROOT / "cptac_v1"
PATHOBENCH_ROOT = ROOT / "pathobench_v1" / "splits"

COHORT_MAP = {
    "cptac_brca": "CPTAC-BRCA",
    "cptac_coad": "CPTAC-COAD",
    "cptac_lscc": "CPTAC-LSCC",
    "cptac_luad": "CPTAC-LUAD",
}


def load_case_to_slides(cohort_key: str) -> dict[str, list[str]]:
    path = CPTAC_ROOT / "manifests" / f"{cohort_key}_custom_list_of_wsis.csv"
    mapping: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            wsi = row["wsi"]
            parts = wsi.split("/")
            case_id = parts[1]
            slide_id = Path(parts[-1]).stem
            mapping.setdefault(case_id, []).append(slide_id)
    return mapping


def main() -> int:
    total_tasks = 0
    total_rows = 0
    for cohort_key, cohort_name in COHORT_MAP.items():
        case_to_slides = load_case_to_slides(cohort_key)
        cohort_dir = PATHOBENCH_ROOT / cohort_key
        if not cohort_dir.is_dir():
            continue
        out_labels_dir = ROOT / cohort_name / "labels"
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        for task_dir in sorted(p for p in cohort_dir.iterdir() if p.is_dir()):
            tsv_path = task_dir / "k=all.tsv"
            if not tsv_path.exists():
                continue
            task_col = task_dir.name
            rows: list[tuple[str, str]] = []
            with open(tsv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    case_id = row.get("case_id", "").strip()
                    label = row.get(task_col, "").strip()
                    if not case_id or label in ("", "nan", "NA"):
                        continue
                    for slide_id in case_to_slides.get(case_id, []):
                        rows.append((slide_id, label))
            if not rows:
                continue
            out_path = out_labels_dir / f"{task_col}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["slide_id", "label"])
                writer.writerows(rows)
            total_tasks += 1
            total_rows += len(rows)
            print(f"{cohort_name}/{task_col}: {len(rows)} slide,label rows -> {out_path}")
    print(f"done: {total_tasks} task files, {total_rows} total rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
