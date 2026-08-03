#!/usr/bin/env python
"""Rank WSI tiles with an EAF forecaster without materializing pruned features."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import (
    FeatureStoreWSIBagDataset,
    H5WSIFeatureStore,
    WSIRankingStore,
    WSITileRanking,
    collate_padded_wsi_bags,
)
from src.models.wsi import load_wsi_tile_importance_forecaster_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Produce per-slide tile rankings from a WSI feature store using an "
            "EAF importance forecaster."
        )
    )
    parser.add_argument("--input-feature-store", type=Path, required=True)
    parser.add_argument("--forecaster-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slide-ids-file", type=Path, default=None)
    parser.add_argument("--keep-ratios", type=float, nargs="+", required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-format",
        type=str,
        choices=("npz", "parquet"),
        default="npz",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if not args.keep_ratios:
        raise ValueError("--keep-ratios must not be empty.")
    normalized = []
    for keep_ratio in args.keep_ratios:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("--keep-ratios values must be in (0, 1].")
        normalized.append(float(keep_ratio))
    args.keep_ratios = tuple(dict.fromkeys(normalized))

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory already exists and is not empty: {args.output_dir}"
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


def _ratio_key(value: float) -> str:
    return format(float(value), ".12g")


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


def _rank_scores(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D; got shape {tuple(scores.shape)}.")
    if scores.numel() == 0:
        raise ValueError("scores must contain at least one tile.")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf.")

    order = torch.argsort(scores, descending=True, stable=True)
    ranks = torch.empty(scores.shape[0], dtype=torch.int64)
    ranks[order] = torch.arange(1, scores.shape[0] + 1, dtype=torch.int64)
    return order, ranks


def _selected_indices(
    scores: torch.Tensor,
    *,
    keep_ratios: tuple[float, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    score_order, _ = _rank_scores(scores)
    by_score: dict[str, torch.Tensor] = {}
    by_original_order: dict[str, torch.Tensor] = {}
    n_tiles = int(scores.numel())

    for keep_ratio in keep_ratios:
        n_keep = max(1, math.ceil(n_tiles * keep_ratio))
        n_keep = min(n_keep, n_tiles)
        key = _ratio_key(keep_ratio)
        selected = score_order[:n_keep].to(dtype=torch.int64)
        by_score[key] = selected
        by_original_order[key] = torch.sort(selected).values

    return by_score, by_original_order


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)

    store = H5WSIFeatureStore(args.input_feature_store)
    all_slide_ids = store.slide_ids()
    if not all_slide_ids:
        raise ValueError(f"input feature store contains no slides: {args.input_feature_store}")

    slide_ids = all_slide_ids if args.slide_ids_file is None else _read_slide_ids(args.slide_ids_file)
    dataset = FeatureStoreWSIBagDataset(store, slide_ids=slide_ids)
    loader = _make_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint = load_wsi_tile_importance_forecaster_checkpoint(
        args.forecaster_checkpoint,
        map_location=device,
    )
    model = checkpoint.model.to(device)
    model.eval()

    if args.output_dir.exists() and args.overwrite:
        for child in args.output_dir.glob(f"*.{args.output_format}"):
            child.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_store = WSIRankingStore(args.output_dir, file_format=args.output_format)

    summary = {
        "event": "start",
        "device": str(device),
        "input_feature_store": str(args.input_feature_store),
        "forecaster_checkpoint": str(args.forecaster_checkpoint),
        "output_dir": str(args.output_dir),
        "output_format": args.output_format,
        "keep_ratios": list(args.keep_ratios),
        "n_slides": len(slide_ids),
    }
    print(json.dumps(summary), flush=True)

    n_slides_written = 0
    n_tiles_total = 0
    feature_dim: int | None = None

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device)
            batch_scores = model(batch.tile_features, mask=batch.mask)

            for row_index, slide_id in enumerate(batch.slide_ids):
                valid_mask = batch.mask[row_index]
                scores = batch_scores[row_index, valid_mask].detach().cpu()
                _, ranks = _rank_scores(scores)
                selected_indices, original_order_indices = _selected_indices(
                    scores,
                    keep_ratios=args.keep_ratios,
                )
                feature_dim = int(batch.tile_features.shape[-1])

                coords = None
                if batch.coords is not None:
                    coords = batch.coords[row_index, valid_mask].detach().cpu()

                source_metadata = (
                    dict(batch.metadata[row_index])
                    if batch.metadata is not None and batch.metadata[row_index] is not None
                    else {}
                )
                source_metadata.update(
                    {
                        "ranking_created_at": datetime.now(timezone.utc).isoformat(),
                        "ranking_checkpoint": str(args.forecaster_checkpoint),
                        "ranking_input_feature_store": str(args.input_feature_store),
                        "ranking_keep_ratios": list(args.keep_ratios),
                        "ranking_feature_dim": feature_dim,
                        "ranking_n_tiles": int(scores.numel()),
                        "ranking_scores_descending": True,
                        "ranking_ranks_are_1_based": True,
                        "ranking_output_format": args.output_format,
                        "ranking_preserved_original_tile_order": True,
                        "ranking_model_type": checkpoint.metadata.get(
                            "model_type",
                            "WSITileImportanceForecaster",
                        ),
                    }
                )
                if "target_source" in checkpoint.metadata:
                    source_metadata["ranking_target_source"] = checkpoint.metadata["target_source"]

                ranking_store.write(
                    WSITileRanking(
                        slide_id=slide_id,
                        coords=coords,
                        scores=scores,
                        ranks=ranks,
                        selected_indices=selected_indices,
                        original_order_indices=original_order_indices,
                        metadata=source_metadata,
                    )
                )

                n_slides_written += 1
                n_tiles_total += int(scores.numel())

    done_summary = {
        "event": "done",
        "input_feature_store": str(args.input_feature_store),
        "forecaster_checkpoint": str(args.forecaster_checkpoint),
        "output_dir": str(args.output_dir),
        "output_format": args.output_format,
        "keep_ratios": list(args.keep_ratios),
        "n_slides": n_slides_written,
        "n_tiles_total": n_tiles_total,
        "feature_dim": feature_dim,
    }
    print(json.dumps(done_summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
