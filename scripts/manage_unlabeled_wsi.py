#!/usr/bin/env python3
"""Discover, estimate, plan, and download public unlabeled WSI cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wsi_pipeline.acquisition import (
    AcquisitionError,
    build_plan,
    discover_gdc,
    discover_idc,
    discover_idc_pathology_collections,
    download_gdc,
    download_idc,
    human_bytes,
    load_catalog,
    load_plan,
    print_plan_summary,
    save_plan,
    write_provider_manifests,
)

DEFAULT_CATALOG = Path("configs/wsi/unlabeled_wsi_sources.json")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-items", type=int, help="Maximum number of slides/series in the plan.")
    parser.add_argument("--max-gib", type=_positive_float, help="Maximum estimated raw download size.")
    parser.add_argument(
        "--per-cohort", type=int, help="Maximum items per project/collection before global limits."
    )
    parser.add_argument("--seed", type=int, default=0)


def _catalog(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    if args.json:
        print(json.dumps(catalog, indent=2, sort_keys=True))
        return 0
    print("Public WSI acquisition sources\n")
    for source in catalog["sources"]:
        estimate = source.get("literature_estimate", {})
        slide_text = estimate.get("slides", "unknown")
        storage = estimate.get("raw_storage", "query live before download")
        print(f"- {source['id']}: {source['name']}")
        print(f"  provider={source['provider']} automation={source['automation']}")
        print(f"  estimated slides={slide_text}; storage={storage}")
        print(f"  {source['notes']}")
    return 0


def _discover_gdc(args: argparse.Namespace) -> int:
    items = discover_gdc(
        projects=args.projects,
        all_tcga=args.all_tcga,
        diagnostic_only=not args.include_tissue_slides,
        page_size=args.page_size,
        timeout=args.timeout,
    )
    plan = build_plan(
        provider="gdc",
        source_id="tcga_gdc" if args.all_tcga else "gdc_custom",
        all_items=items,
        filters={
            "projects": args.projects,
            "all_tcga": args.all_tcga,
            "diagnostic_only": not args.include_tissue_slides,
        },
        max_items=args.max_items,
        max_gib=args.max_gib,
        per_cohort=args.per_cohort,
        seed=args.seed,
    )
    path = save_plan(plan, args.output_dir)
    write_provider_manifests(plan, args.output_dir)
    print(print_plan_summary(plan))
    print(f"Plan: {path}")
    return 0


def _list_idc(_: argparse.Namespace) -> int:
    rows = discover_idc_pathology_collections()
    print("collection_id\tn_series\tapproximate_size")
    for row in rows:
        size_bytes = int(round(float(row.get("approximate_size_MB") or 0) * 1_000_000))
        print(f"{row['collection_id']}\t{int(row['n_series']):,}\t{human_bytes(size_bytes)}")
    return 0


def _discover_idc(args: argparse.Namespace) -> int:
    items = discover_idc(collections=args.collections)
    plan = build_plan(
        provider="idc",
        source_id="idc_pathology",
        all_items=items,
        filters={"collections": args.collections, "modality": "SM"},
        max_items=args.max_items,
        max_gib=args.max_gib,
        per_cohort=args.per_cohort,
        seed=args.seed,
    )
    path = save_plan(plan, args.output_dir)
    write_provider_manifests(plan, args.output_dir)
    print(print_plan_summary(plan))
    print(f"Plan: {path}")
    return 0


def _estimate(args: argparse.Namespace) -> int:
    print(print_plan_summary(load_plan(args.plan)))
    return 0


def _download(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    print(print_plan_summary(plan))
    if not args.yes:
        raise AcquisitionError(
            "Download not started. Review the plan and repeat with --yes. "
            "Large WSI cohorts can consume multiple TiB."
        )
    if args.require_free_gib is not None:
        import shutil

        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(args.output_dir.parent).free
        required = int(args.require_free_gib * 2**30)
        if free_bytes < required:
            raise AcquisitionError(
                f"Only {human_bytes(free_bytes)} free; --require-free-gib requests "
                f"{human_bytes(required)}."
            )
    if plan.provider == "gdc":
        download_gdc(
            plan,
            plan_path=args.plan,
            output_dir=args.output_dir,
            processes=args.processes,
            gdc_client=args.gdc_client,
        )
    elif plan.provider == "idc":
        download_idc(plan, output_dir=args.output_dir)
    else:
        raise AcquisitionError(f"Unsupported plan provider: {plan.provider}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reviewable acquisition plans for public unlabeled pathology WSIs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="Show supported and candidate data sources.")
    catalog.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    catalog.add_argument("--json", action="store_true")
    catalog.set_defaults(func=_catalog)

    gdc = subparsers.add_parser("discover-gdc", help="Query GDC and write an exact TCGA/GDC plan.")
    project_group = gdc.add_mutually_exclusive_group(required=True)
    project_group.add_argument("--projects", nargs="+", help="Explicit GDC project IDs.")
    project_group.add_argument("--all-tcga", action="store_true")
    gdc.add_argument("--include-tissue-slides", action="store_true")
    gdc.add_argument("--page-size", type=int, default=1000)
    gdc.add_argument("--timeout", type=int, default=120)
    gdc.add_argument("--output-dir", type=Path, required=True)
    _add_budget_arguments(gdc)
    gdc.set_defaults(func=_discover_gdc)

    idc_list = subparsers.add_parser(
        "list-idc", help="Live-list every IDC collection containing DICOM WSI series."
    )
    idc_list.set_defaults(func=_list_idc)

    idc = subparsers.add_parser("discover-idc", help="Build a plan from IDC pathology collections.")
    idc.add_argument("--collections", nargs="+", required=True)
    idc.add_argument("--output-dir", type=Path, required=True)
    _add_budget_arguments(idc)
    idc.set_defaults(func=_discover_idc)

    estimate = subparsers.add_parser("estimate", help="Print the exact size stored in a plan.")
    estimate.add_argument("--plan", type=Path, required=True)
    estimate.set_defaults(func=_estimate)

    download = subparsers.add_parser("download", help="Download an already-reviewed plan.")
    download.add_argument("--plan", type=Path, required=True)
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--yes", action="store_true")
    download.add_argument("--require-free-gib", type=_positive_float)
    download.add_argument("--processes", type=int, default=4)
    download.add_argument("--gdc-client", default="gdc-client")
    download.set_defaults(func=_download)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AcquisitionError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
