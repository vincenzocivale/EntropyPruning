#!/usr/bin/env python
"""Validate a paired input/target WSI HDF5 feature store combination.

Checks that an input feature store (e.g. early-layer tile features) and a
target feature store (e.g. tile importance / attention targets) can be
joined via ``load_paired_wsi_bag`` for every slide id common to both stores,
without silently dropping or misaligning tiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import H5WSIFeatureStore, load_paired_wsi_bag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate paired input/target WSI HDF5 feature stores."
    )

    parser.add_argument("--input-feature-store", type=Path, required=True)
    parser.add_argument("--target-feature-store", type=Path, required=True)
    parser.add_argument("--input-feature-dim", type=int, default=None)
    parser.add_argument(
        "--alignment-mode",
        type=str,
        choices=("index", "coords"),
        default="index",
    )
    parser.add_argument("--require-coords", action="store_true")
    parser.add_argument(
        "--require-attention",
        action="store_true",
        help=(
            "Documents intent explicitly. Pairing already requires the "
            "target store to provide an attention/importance value for "
            "every common slide; this flag additionally fails validation "
            "if any common slide is missing target attention."
        ),
    )
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--output-json", type=Path, default=None)

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.input_feature_dim is not None and args.input_feature_dim <= 0:
        raise ValueError("--input-feature-dim must be positive when provided.")
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be positive.")


def _validate_slide(
    input_store: H5WSIFeatureStore,
    target_store: H5WSIFeatureStore,
    slide_id: str,
    *,
    expected_feature_dim: int | None,
    alignment_mode: str,
    require_coords: bool,
) -> tuple[list[str], dict[str, int | str | bool | None]]:
    errors: list[str] = []
    info: dict[str, int | str | bool | None] = {
        "slide_id": slide_id,
        "n_tiles": None,
        "input_feature_dim": None,
        "has_target_attention": None,
        "has_input_coords": None,
        "has_target_coords": None,
    }

    try:
        input_bag = input_store.read(slide_id)
    except Exception as exc:  # noqa: BLE001 - report validation failure, continue.
        return [f"failed to read input slide: {type(exc).__name__}: {exc}"], info

    try:
        target_bag = target_store.read(slide_id)
    except Exception as exc:  # noqa: BLE001 - report validation failure, continue.
        return [f"failed to read target slide: {type(exc).__name__}: {exc}"], info

    info["input_feature_dim"] = input_bag.feature_dim
    info["has_target_attention"] = target_bag.attention is not None
    info["has_input_coords"] = input_bag.coords is not None
    info["has_target_coords"] = target_bag.coords is not None

    if expected_feature_dim is not None and input_bag.feature_dim != expected_feature_dim:
        errors.append(
            "input feature_dim mismatch: expected "
            f"{expected_feature_dim}, got {input_bag.feature_dim}"
        )

    try:
        paired = load_paired_wsi_bag(
            input_store,
            target_store,
            slide_id,
            alignment_mode=alignment_mode,
            require_coords=require_coords,
        )
    except (ValueError, KeyError) as exc:
        errors.append(f"pairing failed: {exc}")
        return errors, info

    info["n_tiles"] = paired.n_tiles
    return errors, info


def main() -> int:
    args = parse_args()

    input_store = H5WSIFeatureStore(args.input_feature_store)
    target_store = H5WSIFeatureStore(args.target_feature_store)

    input_slide_ids = set(input_store.slide_ids())
    target_slide_ids = set(target_store.slide_ids())
    common_slide_ids = sorted(input_slide_ids & target_slide_ids)
    only_in_input = sorted(input_slide_ids - target_slide_ids)
    only_in_target = sorted(target_slide_ids - input_slide_ids)

    errors: list[dict[str, str]] = []

    for slide_id in only_in_input:
        errors.append(
            {
                "slide_id": slide_id,
                "error": "present in input store but missing from target store",
            }
        )
    for slide_id in only_in_target:
        errors.append(
            {
                "slide_id": slide_id,
                "error": "present in target store but missing from input store",
            }
        )

    if not common_slide_ids:
        errors.append(
            {"slide_id": "<store>", "error": "no slide ids common to both feature stores"}
        )

    n_tiles_values: list[int] = []
    feature_dims: set[int] = set()
    n_with_target_attention = 0
    n_with_both_coords = 0
    n_paired_ok = 0

    for slide_id in common_slide_ids:
        slide_errors, info = _validate_slide(
            input_store,
            target_store,
            slide_id,
            expected_feature_dim=args.input_feature_dim,
            alignment_mode=args.alignment_mode,
            require_coords=args.require_coords,
        )

        if isinstance(info["input_feature_dim"], int):
            feature_dims.add(info["input_feature_dim"])
        if info["has_target_attention"] is True:
            n_with_target_attention += 1
        if info["has_input_coords"] is True and info["has_target_coords"] is True:
            n_with_both_coords += 1
        if isinstance(info["n_tiles"], int):
            n_tiles_values.append(info["n_tiles"])
            n_paired_ok += 1

        for error in slide_errors:
            if len(errors) < args.max_errors:
                errors.append({"slide_id": slide_id, "error": error})

    if (
        args.require_attention
        and common_slide_ids
        and n_with_target_attention < len(common_slide_ids)
        and len(errors) < args.max_errors
    ):
        errors.append(
            {
                "slide_id": "<store>",
                "error": (
                    "--require-attention set but only "
                    f"{n_with_target_attention}/{len(common_slide_ids)} common "
                    "slides have target attention"
                ),
            }
        )

    valid = len(errors) == 0

    summary = {
        "valid": valid,
        "input_feature_store": str(args.input_feature_store),
        "target_feature_store": str(args.target_feature_store),
        "alignment_mode": args.alignment_mode,
        "require_coords": args.require_coords,
        "require_attention": args.require_attention,
        "expected_input_feature_dim": args.input_feature_dim,
        "n_slides_input": len(input_slide_ids),
        "n_slides_target": len(target_slide_ids),
        "n_slides_common": len(common_slide_ids),
        "n_slides_only_in_input": len(only_in_input),
        "n_slides_only_in_target": len(only_in_target),
        "n_slides": len(common_slide_ids),
        "n_tiles_total": sum(n_tiles_values),
        "n_tiles_min": min(n_tiles_values) if n_tiles_values else None,
        "n_tiles_mean": (
            sum(n_tiles_values) / len(n_tiles_values) if n_tiles_values else None
        ),
        "n_tiles_max": max(n_tiles_values) if n_tiles_values else None,
        "input_feature_dims": sorted(feature_dims),
        "target_coverage": (
            n_with_target_attention / len(common_slide_ids) if common_slide_ids else None
        ),
        "coords_coverage": (
            n_with_both_coords / len(common_slide_ids) if common_slide_ids else None
        ),
        "n_slides_paired_ok": n_paired_ok,
        "n_errors": len(errors),
        "mismatch_examples": errors,
    }

    output_text = json.dumps(summary, indent=2)
    print(output_text, flush=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output_text)

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
