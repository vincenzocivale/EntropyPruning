#!/usr/bin/env python3
"""Unified operational CLI for the refactored EAF WSI data/cache pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi.corpora import (  # noqa: E402
    HISTAI_SUBSETS,
    build_strict_corpus,
    download_gtex,
    download_histai,
    plan_gtex,
    plan_histai,
    register_hest,
)
from src.data.wsi.layout import StoreLayout  # noqa: E402
from src.wsi_pipeline.archive import (  # noqa: E402
    LifecycleStage,
    SlideLifecycleEvidence,
    classify_stage,
    release_preconditions,
    verify_pixel_archive,
)
from src.wsi_pipeline.cache_io import validate_cache  # noqa: E402


def _root(args: argparse.Namespace) -> Path:
    return StoreLayout.from_root(args.data_root).root


def cmd_layout(args: argparse.Namespace) -> None:
    layout = StoreLayout.from_root(args.data_root)
    layout.ensure_base_dirs()
    print(json.dumps({
        "root": str(layout.root),
        "sources": str(layout.sources),
        "datasets": str(layout.datasets),
        "caches": str(layout.caches),
        "archives": str(layout.archives),
        "checkpoints": str(layout.checkpoints),
        "results": str(layout.results),
        "logs": str(layout.logs),
    }, indent=2))


def cmd_plan_histai(args: argparse.Namespace) -> None:
    path = plan_histai(
        _root(args), token=args.token, subsets=args.subset, force=args.force
    )
    print(path)


def cmd_download_histai(args: argparse.Namespace) -> None:
    path = download_histai(
        _root(args), subsets=args.subset, workers=args.workers, token=args.token
    )
    print(path)


def cmd_plan_gtex(args: argparse.Namespace) -> None:
    print(plan_gtex(_root(args), max_series=args.max_series))


def cmd_download_gtex(args: argparse.Namespace) -> None:
    print(download_gtex(_root(args), workers=args.workers, limit=args.limit))


def cmd_scan_hest(args: argparse.Namespace) -> None:
    print(register_hest(_root(args), args.hest_root, dataset_name=args.dataset_name))


def cmd_build_strict(args: argparse.Namespace) -> None:
    print(
        build_strict_corpus(
            _root(args),
            tcga_root=args.tcga_root,
            sources=args.source or ("histai", "gtex", "hest"),
            seed=args.seed,
        )
    )


def cmd_validate_cache(args: argparse.Namespace) -> None:
    print(json.dumps(validate_cache(args.path, expected_kind=args.kind), indent=2))


def cmd_verify_archive(args: argparse.Namespace) -> None:
    print(json.dumps(verify_pixel_archive(args.path), indent=2))


def cmd_lifecycle(args: argparse.Namespace) -> None:
    evidence = SlideLifecycleEvidence(
        slide_id=args.slide_id,
        raw_path=args.raw_path,
        coords_path=args.coords_path,
        tile_cache_paths=tuple(args.tile_cache or ()),
        wsi_cache_paths=tuple(args.wsi_cache or ()),
        pixel_archive_path=args.archive_path,
        remote_recoverable=args.remote_recoverable,
    )
    stage = classify_stage(evidence)
    checks = release_preconditions(evidence)
    print(
        json.dumps(
            {"slide_id": args.slide_id, "stage": stage.value, "release_preconditions": checks},
            indent=2,
            default=str,
        )
    )
    if stage != LifecycleStage.RAW_RELEASABLE:
        print(
            "[eaf-archive] read-only classification only; there is no raw-deletion "
            "command in this repository.",
            file=sys.stderr,
        )


def add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Canonical $EAF_WSI_ROOT. Defaults to the EAF_WSI_ROOT environment variable.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    layout = sub.add_parser("layout", help="Create/print the canonical runtime layout")
    add_root(layout)
    layout.set_defaults(func=cmd_layout)

    data = sub.add_parser("data", help="Plan/download pretraining corpora")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    p = data_sub.add_parser("plan-histai")
    add_root(p)
    p.add_argument("--subset", action="append", choices=HISTAI_SUBSETS)
    p.add_argument("--token")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-list every requested subset even if already present in plan.csv.",
    )
    p.set_defaults(func=cmd_plan_histai)

    p = data_sub.add_parser("download-histai")
    add_root(p)
    p.add_argument("--subset", action="append", choices=HISTAI_SUBSETS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--token")
    p.set_defaults(func=cmd_download_histai)

    p = data_sub.add_parser("plan-gtex")
    add_root(p)
    p.add_argument("--max-series", type=int)
    p.set_defaults(func=cmd_plan_gtex)

    p = data_sub.add_parser("download-gtex")
    add_root(p)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_download_gtex)

    p = data_sub.add_parser(
        "scan-hest", help="Register an existing HEST raw-WSI tree without copying it"
    )
    add_root(p)
    p.add_argument("--hest-root", type=Path, required=True)
    p.add_argument("--dataset-name", default="hest_eaf_wsi_v1")
    p.set_defaults(func=cmd_scan_hest)

    p = data_sub.add_parser(
        "build-strict",
        help="Union downloaded/registered pretraining sources into eaf_wsi_pretrain_strict_v1",
    )
    add_root(p)
    p.add_argument(
        "--source",
        action="append",
        choices=("histai", "hest", "gtex"),
        help="Repeatable. Defaults to histai + gtex + hest (TCGA is excluded by construction).",
    )
    p.add_argument(
        "--tcga-root",
        type=Path,
        help="Preserved TCGA root to guard against; defaults to <data-root>/sources/gdc/tcga.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(func=cmd_build_strict)

    cache = sub.add_parser("cache", help="Inspect versioned offline teacher caches")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    p = cache_sub.add_parser("validate")
    p.add_argument("path", type=Path)
    p.add_argument("--kind", choices=("tile_eaf", "wsi_eaf"))
    p.set_defaults(func=cmd_validate_cache)

    archive = sub.add_parser("archive", help="Operate on cold tissue-pixel archives")
    archive_sub = archive.add_subparsers(dest="archive_command", required=True)
    p = archive_sub.add_parser("verify")
    p.add_argument("path", type=Path)
    p.set_defaults(func=cmd_verify_archive)

    p = archive_sub.add_parser(
        "lifecycle",
        help=(
            "Read-only: classify a slide's RAW..RAW_RELEASABLE stage and report "
            "release preconditions. Never deletes anything."
        ),
    )
    p.add_argument("--slide-id", required=True)
    p.add_argument("--raw-path", type=Path)
    p.add_argument("--coords-path", type=Path)
    p.add_argument("--tile-cache", type=Path, action="append")
    p.add_argument("--wsi-cache", type=Path, action="append")
    p.add_argument("--archive-path", type=Path)
    p.add_argument(
        "--remote-recoverable",
        action="store_true",
        help="Mark the source as safely remotely recoverable (e.g. public GTEx/IDC).",
    )
    p.set_defaults(func=cmd_lifecycle)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
