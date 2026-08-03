#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.wsi_pipeline.preprocess import TridentConfig, run_trident_preprocessing


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable TRIDENT segmentation and coordinate extraction.")
    parser.add_argument("--trident-repo", type=Path, required=True)
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=("seg", "coords"), default=("seg", "coords"))
    parser.add_argument("--gpus", nargs="+", type=int, default=(0,))
    parser.add_argument("--segmenter", default="hest")
    parser.add_argument("--mag", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--custom-list-of-wsis", type=Path)
    parser.add_argument("--no-search-nested", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remove-artifacts", action="store_true")
    group.add_argument("--remove-penmarks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = TridentConfig(
        trident_repo=args.trident_repo,
        wsi_dir=args.wsi_dir,
        job_dir=args.job_dir,
        gpus=tuple(args.gpus),
        segmenter=args.segmenter,
        mag=args.mag,
        patch_size=args.patch_size,
        overlap=args.overlap,
        custom_list_of_wsis=args.custom_list_of_wsis,
        search_nested=not args.no_search_nested,
        remove_artifacts=args.remove_artifacts,
        remove_penmarks=args.remove_penmarks,
    )
    run_trident_preprocessing(config, args.stages, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
