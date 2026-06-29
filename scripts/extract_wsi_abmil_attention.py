#!/usr/bin/env python
"""Extract ABMIL tile attention targets into a WSI HDF5 feature store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import (
    FeatureStoreWSIBagDataset,
    H5WSIFeatureStore,
    WSIBag,
    collate_padded_wsi_bags,
)
from src.models.wsi import load_abmil_classifier_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tile attention from an ABMIL checkpoint into an HDF5 WSI store."
    )

    parser.add_argument("--input-feature-store", type=Path, required=True)
    parser.add_argument("--output-feature-store", type=Path, required=True)
    parser.add_argument("--abmil-checkpoint", type=Path, required=True)

    parser.add_argument("--slide-ids-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    input_path = args.input_feature_store.resolve()
    output_path = args.output_feature_store.resolve()
    if input_path == output_path:
        raise ValueError(
            "--output-feature-store must be different from --input-feature-store."
        )

    if args.output_feature_store.exists() and not args.overwrite:
        raise FileExistsError(
            f"output feature store already exists: {args.output_feature_store}"
        )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return resolved


def _read_slide_ids(path: Path) -> tuple[str, ...]:
    slide_ids = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        slide_ids.append(line)

    if not slide_ids:
        raise ValueError(f"slide id file is empty: {path}")

    return tuple(slide_ids)


def _make_loader(
    dataset: FeatureStoreWSIBagDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_padded_wsi_bags,
    )


def main() -> int:
    args = parse_args()

    if args.output_feature_store.exists() and args.overwrite:
        args.output_feature_store.unlink()

    device = _resolve_device(args.device)

    input_store = H5WSIFeatureStore(args.input_feature_store)
    all_slide_ids = input_store.slide_ids()
    if not all_slide_ids:
        raise ValueError(f"input feature store contains no slides: {args.input_feature_store}")

    if args.slide_ids_file is None:
        slide_ids = all_slide_ids
    else:
        slide_ids = _read_slide_ids(args.slide_ids_file)

    dataset = FeatureStoreWSIBagDataset(input_store, slide_ids=slide_ids)
    loader = _make_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint = load_abmil_classifier_checkpoint(
        args.abmil_checkpoint,
        map_location=device,
    )
    model = checkpoint.model.to(device)
    model.eval()

    output_store = H5WSIFeatureStore(args.output_feature_store)

    n_written = 0
    n_tiles_total = 0

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "input_feature_store": str(args.input_feature_store),
                "output_feature_store": str(args.output_feature_store),
                "abmil_checkpoint": str(args.abmil_checkpoint),
                "n_slides": len(slide_ids),
            }
        ),
        flush=True,
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device)
            output = model(batch.tile_features, mask=batch.mask)

            for row, slide_id in enumerate(batch.slide_ids):
                valid_mask = batch.mask[row]
                n_tiles = int(valid_mask.sum().item())

                tile_features = batch.tile_features[row, valid_mask].detach().cpu()
                attention = output.attention[row, valid_mask].detach().cpu()

                coords = None
                if batch.coords is not None:
                    coords = batch.coords[row, valid_mask].detach().cpu()

                label = batch.labels[row] if batch.labels else None
                if isinstance(label, torch.Tensor):
                    label = label.detach().cpu()

                metadata = batch.metadata[row] if batch.metadata else None
                metadata = dict(metadata) if metadata is not None else None
                if metadata is not None:
                    metadata["attention_source"] = "ABMILClassifier"
                    metadata["abmil_checkpoint"] = str(args.abmil_checkpoint)
                else:
                    metadata = {
                        "attention_source": "ABMILClassifier",
                        "abmil_checkpoint": str(args.abmil_checkpoint),
                    }

                output_store.write(
                    WSIBag(
                        slide_id=slide_id,
                        tile_features=tile_features,
                        coords=coords,
                        label=label,
                        attention=attention,
                        metadata=metadata,
                    )
                )

                n_written += 1
                n_tiles_total += n_tiles

    summary = {
        "event": "done",
        "input_feature_store": str(args.input_feature_store),
        "output_feature_store": str(args.output_feature_store),
        "abmil_checkpoint": str(args.abmil_checkpoint),
        "n_slides": n_written,
        "n_tiles_total": n_tiles_total,
    }

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
