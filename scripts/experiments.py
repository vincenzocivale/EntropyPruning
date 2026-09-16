#!/usr/bin/env python3
"""Inspect declared EAF experiments and deterministic paths."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wsi_pipeline.experiment_registry import load_registry


def cmd_list(args) -> int:
    registry = load_registry(args.registry)[0]
    for experiment_id, entry in registry.get("experiments", {}).items():
        variants = ", ".join(sorted(entry.get("variants", {}))) or "-"
        blockers = "; ".join(entry.get("blockers", []))
        print(
            f"{experiment_id}\t{entry.get('status')}\t{entry.get('family')}/{entry.get('stage')}\tvariants={variants}"
            + (f"\tblockers={blockers}" if blockers else "")
        )
    return 0


def cmd_show(args) -> int:
    registry = load_registry(args.registry)[0]
    entry = registry.get("experiments", {}).get(args.experiment_id)
    if entry is None:
        raise SystemExit(f"Unknown experiment: {args.experiment_id}")
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


def cmd_paths(args) -> int:
    registry, registry_path, digest = load_registry(args.registry)
    entry = registry.get("experiments", {}).get(args.experiment_id)
    if entry is None:
        raise SystemExit(f"Unknown experiment: {args.experiment_id}")
    if args.variant_id not in entry.get("variants", {}):
        raise SystemExit(f"Unknown variant {args.variant_id!r}; declared={sorted(entry.get('variants', {}))}")
    root_value = args.data_root or os.environ.get("EAF_WSI_ROOT")
    if not root_value:
        raise SystemExit("Set EAF_WSI_ROOT or pass --data-root")
    root = Path(root_value).expanduser().resolve()
    suffix = Path(args.experiment_id) / args.variant_id / f"seed_{args.seed}"
    family, stage = entry["family"], entry["stage"]
    paths = {
        "registry": str(registry_path),
        "registry_sha256": digest,
        "run_key": str(suffix),
        "checkpoint_dir": str(root / "checkpoints" / family / stage / suffix),
        "result_dir": str(root / "results" / family / stage / suffix),
        "log_dir": str(root / "logs" / family / stage / suffix),
        "derived_cache_dir": str(root / "caches" / "experiments" / suffix),
        "tile_pruned_cache_dir": str(root / "caches" / "experiments" / suffix / "tile_pruned"),
        "wsi_source_cache_dir": str(root / "caches" / "experiments" / suffix / "wsi_source"),
    }
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("show")
    p.add_argument("experiment_id")
    p.set_defaults(func=cmd_show)
    p = sub.add_parser("paths")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--variant-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.set_defaults(func=cmd_paths)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
