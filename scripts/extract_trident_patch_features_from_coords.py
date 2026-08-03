#!/usr/bin/env python
"""Extract TRIDENT-style patch features from an existing coords file.

This is a small single-slide helper intended for low-concurrency environments:
it bypasses TRIDENT's batch wrapper and calls `WSI.extract_patch_features`
directly on one slide.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIDENT_REPO = Path("/data2/home/vcivale/repos/TRIDENT")

if str(TRIDENT_REPO) not in sys.path:
    sys.path.insert(0, str(TRIDENT_REPO))

from trident import load_wsi
from trident.patch_encoder_models import encoder_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TRIDENT-style patch features for one slide from an existing coords file."
    )
    parser.add_argument("--slide-path", type=Path, required=True)
    parser.add_argument("--coords-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-encoder", type=str, default="uni_v1")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-limit", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--feature-precision", type=str, choices=("float32", "native"), default="float32")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.batch_limit <= 0:
        raise ValueError("--batch-limit must be positive.")
    if args.max_workers < 0:
        raise ValueError("--max-workers must be non-negative.")
    if not args.slide_path.exists():
        raise FileNotFoundError(args.slide_path)
    if not args.coords_path.exists():
        raise FileNotFoundError(args.coords_path)
    return args


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(
        json.dumps(
            {
                "event": "start",
                "slide_path": str(args.slide_path),
                "coords_path": str(args.coords_path),
                "output_dir": str(args.output_dir),
                "patch_encoder": args.patch_encoder,
                "device": args.device,
                "batch_limit": args.batch_limit,
                "max_workers": args.max_workers,
            }
        ),
        flush=True,
    )

    encoder_kwargs = {}
    if args.weights_path is not None:
        encoder_kwargs["weights_path"] = str(args.weights_path)

    encoder = encoder_factory(args.patch_encoder, **encoder_kwargs)
    if args.feature_precision == "float32":
        encoder.precision = torch.float32
    encoder.eval()

    with load_wsi(
        slide_path=str(args.slide_path),
        lazy_init=False,
        max_workers=args.max_workers,
    ) as slide:
        output_path = slide.extract_patch_features(
            patch_encoder=encoder,
            coords_path=str(args.coords_path),
            save_features=str(args.output_dir),
            device=args.device,
            saveas="h5",
            batch_limit=args.batch_limit,
            verbose=args.verbose,
        )

    print(
        json.dumps(
            {
                "event": "done",
                "output_path": str(output_path),
                "elapsed_s": round(time.time() - started, 2),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
