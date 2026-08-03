#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.wsi_pipeline.registry import load_slides, write_manifest
from src.wsi_pipeline.tile_encoders import (
    ConchV15MultiLayerEncoder,
    TimmPreprocess,
    TimmViTMultiLayerEncoder,
)
from src.wsi_pipeline.tile_extraction import TileExtractionConfig, extract_many_slides


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract second-block and final tile embeddings in one forward per patch batch."
    )
    parser.add_argument("--slides", type=Path, required=True, help="CSV with slide_id,wsi_path")
    parser.add_argument("--coords-registry", type=Path, required=True, help="CSV with slide_id,path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coords-artifact-type", default="coords")
    parser.add_argument("--encoder", choices=("conch_v15", "timm"), default="conch_v15")
    parser.add_argument("--model-name", help="timm model name when --encoder=timm")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-slides", type=int)
    args = parser.parse_args()

    slides = {entry.slide_id: entry for entry in load_slides(args.slides)}
    coords = pd.read_csv(args.coords_registry)
    if "artifact_type" in coords.columns and args.coords_artifact_type:
        coords = coords[coords["artifact_type"] == args.coords_artifact_type]
    if not {"slide_id", "path"}.issubset(coords.columns):
        raise ValueError("coords registry requires slide_id,path")
    items = []
    for row in coords.to_dict("records"):
        slide_id = str(row["slide_id"])
        if slide_id not in slides:
            continue
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = (args.coords_registry.parent / path).resolve()
        items.append((slides[slide_id], path))
    if args.max_slides is not None:
        items = items[: args.max_slides]

    if args.encoder == "conch_v15":
        encoder = ConchV15MultiLayerEncoder(token=args.hf_token)
    else:
        if not args.model_name:
            parser.error("--model-name is required with --encoder=timm")
        encoder = TimmViTMultiLayerEncoder(
            args.model_name,
            preprocess=TimmPreprocess(input_size=args.input_size),
        )
    config = TileExtractionConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        storage_dtype=args.storage_dtype,
        compression=None if args.compression == "none" else args.compression,
        device=args.device,
        overwrite=args.overwrite,
    )
    rows = extract_many_slides(items, encoder=encoder, config=config)
    write_manifest(rows, args.output_dir / "manifest.csv")
    errors = [row for row in rows if row.get("status") == "error"]
    print(f"complete={len(rows) - len(errors)} errors={len(errors)} manifest={args.output_dir / 'manifest.csv'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
