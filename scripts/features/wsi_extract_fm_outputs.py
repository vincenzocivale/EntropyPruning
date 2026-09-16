#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.wsi_pipeline.registry import write_manifest
from src.wsi_pipeline.wsi_extraction import WSIExtractionConfig, extract_many_wsi_outputs
from src.wsi_pipeline.wsi_models import create_wsi_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract WSI embeddings and native attention when the model exposes it.")
    parser.add_argument("--feature-manifest", type=Path, required=True, help="CSV with path and optionally slide_id")
    parser.add_argument(
        "--path-root",
        type=Path,
        help="Root for relative feature paths; by default also tries the manifest's parent and parent directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-type", default="tile_features")
    parser.add_argument("--feature-set-id")
    parser.add_argument("--model", choices=("feather", "titan", "gigapath"), required=True)
    parser.add_argument("--feature-key", default="final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--patch-size-level0", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--feather-model", default="abmil.base.conch_v15.pc108-24k")
    parser.add_argument(
        "--titan-attention-mode",
        action="append",
        choices=("global_to_tokens", "received", "rollout", "full"),
        help="Repeat to select TITAN outputs. Default: global_to_tokens, received, rollout.",
    )
    parser.add_argument("--titan-global-token-index", type=int, default=0)
    parser.add_argument(
        "--titan-full-layer",
        action="append",
        type=int,
        help="Layer index to save as a full HxTxT matrix; negative indices are supported.",
    )
    parser.add_argument("--titan-max-full-attention-tokens", type=int, default=2048)
    parser.add_argument("--titan-max-rollout-tokens", type=int, default=4096)
    parser.add_argument("--titan-revision")
    args = parser.parse_args()

    table = pd.read_csv(args.feature_manifest)
    if "artifact_type" in table.columns and args.artifact_type:
        table = table[table["artifact_type"] == args.artifact_type]
    if args.feature_set_id is not None:
        if "feature_set_id" not in table.columns:
            raise ValueError("--feature-set-id requested but manifest lacks feature_set_id")
        table = table[table["feature_set_id"] == args.feature_set_id]
    if "status" in table.columns:
        table = table[table["status"].isin(["complete", "available", "valid", "skipped"])]
    if "path" not in table:
        raise ValueError("feature manifest requires path")
    paths = []
    for value in table["path"].tolist():
        path = Path(str(value))
        if not path.is_absolute():
            roots = [args.feature_manifest.parent]
            if args.path_root is not None:
                roots.insert(0, args.path_root)
            # Dataset artifact registries commonly live in ``<dataset>/registry``
            # while their paths are rooted at ``<dataset>``.
            roots.append(args.feature_manifest.parent.parent)
            candidates = [(root / path).resolve() for root in roots]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        paths.append(path)
    if args.max_slides is not None:
        paths = paths[: args.max_slides]

    if args.model == "feather":
        model = create_wsi_model(
            "feather",
            model_id=args.feather_model,
            token=args.hf_token,
        )
    elif args.model == "titan":
        modes = tuple(args.titan_attention_mode or ("global_to_tokens", "received", "rollout"))
        full_layers = tuple(args.titan_full_layer or (-1,))
        model = create_wsi_model(
            "titan",
            token=args.hf_token,
            attention_modes=modes,
            global_token_index=args.titan_global_token_index,
            full_attention_layers=full_layers,
            max_full_attention_tokens=args.titan_max_full_attention_tokens,
            max_rollout_tokens=args.titan_max_rollout_tokens,
            revision=args.titan_revision,
        )
    else:
        model = create_wsi_model("gigapath")
    config = WSIExtractionConfig(
        output_dir=args.output_dir,
        device=args.device,
        storage_dtype=args.storage_dtype,
        compression=None if args.compression == "none" else args.compression,
        patch_size_level0=args.patch_size_level0,
        overwrite=args.overwrite,
        max_full_attention_tiles=args.titan_max_full_attention_tokens,
    )
    rows = extract_many_wsi_outputs(paths, model=model, config=config, feature_key=args.feature_key)
    write_manifest(rows, args.output_dir / "manifest.csv")
    errors = [row for row in rows if row.get("status") == "error"]
    missing_attention = sum(not row.get("attention_available", False) for row in rows if row.get("status") == "complete")
    print(
        f"complete={len(rows) - len(errors)} errors={len(errors)} "
        f"without_native_attention={missing_attention} manifest={args.output_dir / 'manifest.csv'}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
