from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


WSI_ROOT = Path(
    "/data2/home/vcivale/projects/imaging/data/WSI"
)

DATASET_ROOT = (
    WSI_ROOT
    / "datasets/pretraining/tcga_eaf_multicohort_v1"
)

SLIDES_MANIFEST = DATASET_ROOT / "manifests/slides.csv"

BLOCKED_CASES_PATH = (
    WSI_ROOT
    / "catalog/thunder_overlap/all_thunder_tcga_cases.txt"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "manifests/leakage_firewall"
)

AUDIT_PATH = OUTPUT_ROOT / "thunder_overlap_audit.csv"
CLEAN_PATH = OUTPUT_ROOT / "pretrain_thunder_clean.csv"
EXCLUDED_PATH = (
    OUTPUT_ROOT
    / "excluded_thunder_and_wsi_benchmarks.csv"
)

TCGA_CASE_RE = re.compile(
    r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
    re.IGNORECASE,
)

# WSI benchmark che devono restare completamente unseen.
RESERVED_COHORTS = {
    "TCGA-LUAD",
    "TCGA-LUSC",
}


def normalize_case(row: dict[str, str]) -> str:
    existing = row.get("case_id", "").strip().upper()

    match = TCGA_CASE_RE.search(existing)
    if match:
        return match.group(1).upper()

    for column in ("slide_id", "file_name", "raw_path"):
        match = TCGA_CASE_RE.search(
            row.get(column, "")
        )

        if match:
            return match.group(1).upper()

    return ""


def main() -> None:
    if not BLOCKED_CASES_PATH.is_file():
        raise RuntimeError(
            f"Lista THUNDER assente: {BLOCKED_CASES_PATH}"
        )

    blocked_cases = {
        line.strip().upper()
        for line in BLOCKED_CASES_PATH.read_text().splitlines()
        if line.strip()
    }

    if not blocked_cases:
        raise RuntimeError(
            "La lista dei casi THUNDER è vuota."
        )

    with SLIDES_MANIFEST.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])

    audit_rows = []

    for row in rows:
        result = dict(row)

        case_id = normalize_case(row)
        cohort = row.get("cohort", "").strip().upper()

        reasons = []

        if case_id in blocked_cases:
            reasons.append("thunder_tcga_case_overlap")

        if cohort in RESERVED_COHORTS:
            reasons.append("reserved_wsi_benchmark_cohort")

        result["normalized_case_id"] = case_id
        result["leakage_status"] = (
            "exclude" if reasons else "eligible"
        )
        result["leakage_reasons"] = "|".join(reasons)

        audit_rows.append(result)

    output_fields = list(original_fields)

    for field in (
        "normalized_case_id",
        "leakage_status",
        "leakage_reasons",
    ):
        if field not in output_fields:
            output_fields.append(field)

    eligible = [
        row
        for row in audit_rows
        if row["leakage_status"] == "eligible"
    ]

    excluded = [
        row
        for row in audit_rows
        if row["leakage_status"] == "exclude"
    ]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for path, data in (
        (AUDIT_PATH, audit_rows),
        (CLEAN_PATH, eligible),
        (EXCLUDED_PATH, excluded),
    ):
        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=output_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(data)

    reason_counts = Counter()

    for row in excluded:
        reason_counts.update(
            row["leakage_reasons"].split("|")
        )

    eligible_cases = {
        row["normalized_case_id"]
        for row in eligible
        if row["normalized_case_id"]
    }

    excluded_cases = {
        row["normalized_case_id"]
        for row in excluded
        if row["normalized_case_id"]
    }

    print("=== THUNDER LEAKAGE FIREWALL ===")
    print(f"Slide totali:       {len(audit_rows):,}")
    print(f"Slide eligible:     {len(eligible):,}")
    print(f"Slide escluse:      {len(excluded):,}")
    print(f"Casi eligible:      {len(eligible_cases):,}")
    print(f"Casi esclusi:       {len(excluded_cases):,}")

    print("\n=== MOTIVI ===")
    for reason, count in sorted(reason_counts.items()):
        print(f"{reason:40s} {count:,}")

    print("\nAudit:", AUDIT_PATH)
    print("Clean:", CLEAN_PATH)
    print("Excluded:", EXCLUDED_PATH)


if __name__ == "__main__":
    main()
