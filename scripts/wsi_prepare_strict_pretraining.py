#!/usr/bin/env python3
"""DEPRECATED compatibility wrapper for the strict EAF-WSI pretraining CLI.

All logic now lives in ``src/data/wsi/{layout,manifest,corpora}.py`` and is
exposed through the unified ``scripts/eaf.py`` CLI (see AGENTS.md
"Operational entry point"). This wrapper keeps the old subcommand names
(``init``, ``plan-histai``, ``download-histai``, ``scan-hest``, ``build``)
working for anyone with saved commands/notebooks, but every one of them is a
thin call into the same core functions ``scripts/eaf.py`` uses — there is only
one HISTAI/GTEx/HEST planner implementation.

Prefer the new entry points going forward::

    python scripts/eaf.py layout --data-root "$EAF_WSI_ROOT"
    python scripts/eaf.py data plan-histai --data-root "$EAF_WSI_ROOT"
    python scripts/eaf.py data download-histai --data-root "$EAF_WSI_ROOT" --workers 8
    python scripts/eaf.py data scan-hest --data-root "$EAF_WSI_ROOT" --hest-root ...
    python scripts/eaf.py data build-strict --data-root "$EAF_WSI_ROOT"

One behavior change from the pre-refactor version of this script: there is no
longer a separate ``registry.json`` written by ``init``. The TCGA leakage
guard now defaults to ``<data-root>/sources/gdc/tcga`` automatically (matching
the old default) and can be overridden with ``--tcga-root`` directly on
``build``/``build-strict``; ``init`` is now just a directory-creation no-op
kept for script-compatibility.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi.corpora import (  # noqa: E402
    HISTAI_SUBSETS,
    build_strict_corpus,
    download_histai,
    plan_histai,
    register_hest,
)
from src.data.wsi.layout import StoreLayout  # noqa: E402

_DEPRECATION_NOTICE = (
    "[eaf-wsi-data] scripts/wsi_prepare_strict_pretraining.py is a compatibility "
    "wrapper; prefer `python scripts/eaf.py data ...` (see AGENTS.md)."
)


def _add_data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["EAF_WSI_ROOT"]) if os.environ.get("EAF_WSI_ROOT") else None,
        required="EAF_WSI_ROOT" not in os.environ,
        help="Canonical WSI data root (see docs/data_layout.md). Defaults to $EAF_WSI_ROOT.",
    )


def cmd_init(args: argparse.Namespace) -> None:
    layout = StoreLayout.from_root(args.data_root)
    layout.ensure_base_dirs()
    tcga = args.tcga_root or (layout.sources / "gdc" / "tcga")
    hest_note = f"; pass --hest-root to `scan-hest` next: {args.hest_root}" if args.hest_root else ""
    print(f"[eaf-wsi-data] initialized layout under: {layout.root}")
    if Path(tcga).exists():
        print(f"[eaf-wsi-data] TCGA preserved in place and excluded from strict-v1: {tcga}")
    else:
        print("[eaf-wsi-data] no TCGA root found; leakage guard will be a no-op at build time")
    print("[eaf-wsi-data] registry.json is no longer written; `build` takes --tcga-root directly" + hest_note)


def cmd_plan_histai(args: argparse.Namespace) -> None:
    print(plan_histai(args.data_root, token=args.token, subsets=args.subset, force=args.force))


def cmd_download_histai(args: argparse.Namespace) -> None:
    print(
        download_histai(
            args.data_root, subsets=args.subset, workers=args.workers, token=args.token
        )
    )


def cmd_scan_hest(args: argparse.Namespace) -> None:
    print(register_hest(args.data_root, args.hest_root))


def cmd_build(args: argparse.Namespace) -> None:
    print(
        build_strict_corpus(
            args.data_root,
            tcga_root=args.tcga_root,
            sources=args.source or ("histai", "gtex", "hest"),
            seed=args.seed,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="[compat] Ensure layout dirs exist; no registry.json anymore.")
    _add_data_root(p)
    p.add_argument("--tcga-root", type=Path)
    p.add_argument("--hest-root", type=Path)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("plan-histai", help="[compat] see: eaf.py data plan-histai")
    _add_data_root(p)
    p.add_argument("--subset", action="append", choices=HISTAI_SUBSETS)
    p.add_argument("--token")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_plan_histai)

    p = sub.add_parser("download-histai", help="[compat] see: eaf.py data download-histai")
    _add_data_root(p)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--subset", action="append", choices=HISTAI_SUBSETS)
    p.add_argument("--token")
    p.set_defaults(func=cmd_download_histai)

    p = sub.add_parser("scan-hest", help="[compat] see: eaf.py data scan-hest")
    _add_data_root(p)
    p.add_argument("--hest-root", type=Path, required=True)
    p.set_defaults(func=cmd_scan_hest)

    p = sub.add_parser("build", help="[compat] see: eaf.py data build-strict")
    _add_data_root(p)
    p.add_argument("--tcga-root", type=Path)
    p.add_argument("--source", action="append", choices=("histai", "hest", "gtex"))
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(func=cmd_build)

    return parser


def main() -> int:
    print(_DEPRECATION_NOTICE, file=sys.stderr)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
