#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path

import requests


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def build_filters(project: str):
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_category",
                    "value": ["Biospecimen"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Slide Image"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_format",
                    "value": ["SVS"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.access",
                    "value": ["open"],
                },
            },
        ],
    }


def get_nested_case_submitter_id(hit):
    cases = hit.get("cases") or []
    if not cases:
        return ""
    return cases[0].get("submitter_id", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="TCGA-COAD")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer-dx", action="store_true")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-slides-csv", required=True)
    parser.add_argument("--query-size", type=int, default=5000)
    args = parser.parse_args()

    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "state",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.sample_type",
    ]

    payload = {
        "filters": build_filters(args.project),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": args.query_size,
        "sort": "file_size:asc",
    }

    response = requests.post(GDC_FILES_ENDPOINT, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    hits = data.get("data", {}).get("hits", [])
    if not hits:
        raise SystemExit(
            f"No SVS slide images found for project={args.project}. "
            "Try another project, e.g. TCGA-BRCA, TCGA-LUAD, TCGA-LUSC, TCGA-READ."
        )

    # Prefer diagnostic slides when filename convention exposes DX.
    # TCGA filenames often include DX for diagnostic slides.
    if args.prefer_dx:
        dx_hits = [h for h in hits if "DX" in h.get("file_name", "").upper()]
        if dx_hits:
            hits = dx_hits

    rng = random.Random(args.seed)
    rng.shuffle(hits)
    selected = hits[: args.n]

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    output_slides_csv = Path(args.output_slides_csv)
    output_slides_csv.parent.mkdir(parents=True, exist_ok=True)

    # GDC manifest format commonly used by gdc-client:
    # id, filename, md5, size, state
    with output_manifest.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "filename", "md5", "size", "state"])
        for h in selected:
            writer.writerow(
                [
                    h["file_id"],
                    h["file_name"],
                    h.get("md5sum", ""),
                    h.get("file_size", ""),
                    h.get("state", "released"),
                ]
            )

    with output_slides_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "slide_id",
                "file_id",
                "file_name",
                "case_submitter_id",
                "file_size",
                "project",
            ]
        )
        for h in selected:
            file_name = h["file_name"]
            slide_id = Path(file_name).stem
            writer.writerow(
                [
                    slide_id,
                    h["file_id"],
                    file_name,
                    get_nested_case_submitter_id(h),
                    h.get("file_size", ""),
                    args.project,
                ]
            )

    total_gb = sum(int(h.get("file_size", 0) or 0) for h in selected) / 1e9

    print(f"Project: {args.project}")
    print(f"Available hits after filtering: {len(hits)}")
    print(f"Selected slides: {len(selected)}")
    print(f"Estimated download size: {total_gb:.2f} GB")
    print(f"Wrote manifest: {output_manifest}")
    print(f"Wrote slides CSV: {output_slides_csv}")
    print()
    print("First selected files:")
    for h in selected[:5]:
        print(f"  {h['file_name']}  {int(h.get('file_size', 0) or 0) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
