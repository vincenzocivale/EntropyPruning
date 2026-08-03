#!/usr/bin/env python
"""Evaluate feature-level WSI tile pruning driven by an attention forecaster."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import (
    FeatureStoreWSIBagDataset,
    H5WSIFeatureStore,
    collate_padded_wsi_bags,
)
from src.evaluation.wsi_attention_metrics import wsi_attention_spearmanr
from src.models.wsi import load_wsi_tile_attention_forecaster_checkpoint


@dataclass(frozen=True)
class SlidePruningMetrics:
    slide_id: str
    keep_ratio: float
    n_tiles: int
    n_tiles_kept: int
    effective_keep_ratio: float
    spearmanr: float
    topk_overlap: float
    ndcg_at_k: float
    attention_mass_retained: float
    oracle_attention_mass_at_k: float
    relative_attention_mass_retained: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WSI tile pruning from forecaster scores."
    )

    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--forecaster-checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--keep-ratios", type=float, nargs="+", required=True)

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
    if not args.keep_ratios:
        raise ValueError("--keep-ratios must contain at least one value.")

    for ratio in args.keep_ratios:
        if not 0.0 < ratio <= 1.0:
            raise ValueError("--keep-ratios values must be in (0, 1].")

    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"output CSV already exists: {args.output_csv}")


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


def _topk_overlap(pred_indices: torch.Tensor, target_indices: torch.Tensor) -> float:
    pred_set = set(pred_indices.detach().cpu().tolist())
    target_set = set(target_indices.detach().cpu().tolist())
    return len(pred_set.intersection(target_set)) / len(pred_set)


def _ndcg_at_k(target_attention: torch.Tensor, pred_indices: torch.Tensor, ideal_indices: torch.Tensor) -> float:
    k = int(pred_indices.numel())
    discounts = 1.0 / torch.log2(
        torch.arange(k, device=target_attention.device, dtype=torch.float32) + 2.0
    )

    dcg = (target_attention[pred_indices].to(torch.float32) * discounts).sum()
    ideal_dcg = (target_attention[ideal_indices].to(torch.float32) * discounts).sum()

    if ideal_dcg <= 0:
        raise ValueError("ideal DCG must be positive.")

    return float((dcg / ideal_dcg).detach().cpu())


def _evaluate_slide_ratio(
    *,
    slide_id: str,
    scores: torch.Tensor,
    target_attention: torch.Tensor,
    keep_ratio: float,
) -> SlidePruningMetrics:
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D; got {tuple(scores.shape)}.")
    if target_attention.ndim != 1:
        raise ValueError(
            f"target_attention must be 1D; got {tuple(target_attention.shape)}."
        )
    if scores.shape != target_attention.shape:
        raise ValueError(
            "scores and target_attention must have the same shape; "
            f"got {tuple(scores.shape)} and {tuple(target_attention.shape)}."
        )
    if scores.numel() == 0:
        raise ValueError("each slide must contain at least one tile.")
    if not torch.isfinite(scores).all():
        raise ValueError(f"scores contain NaN or Inf for slide {slide_id}.")
    if not torch.isfinite(target_attention).all():
        raise ValueError(f"attention contains NaN or Inf for slide {slide_id}.")
    if (target_attention < 0).any():
        raise ValueError(f"attention contains negative values for slide {slide_id}.")

    attention_mass = target_attention.sum()
    if attention_mass <= 0:
        raise ValueError(f"attention mass must be positive for slide {slide_id}.")

    n_tiles = int(scores.numel())
    n_tiles_kept = max(1, math.ceil(n_tiles * keep_ratio))
    n_tiles_kept = min(n_tiles_kept, n_tiles)

    pred_indices = torch.topk(scores, k=n_tiles_kept).indices
    ideal_indices = torch.topk(target_attention, k=n_tiles_kept).indices

    retained_mass = (target_attention[pred_indices].sum() / attention_mass).clamp(0.0, 1.0)
    oracle_mass = (target_attention[ideal_indices].sum() / attention_mass).clamp(0.0, 1.0)
    relative_mass = (retained_mass / oracle_mass).clamp(0.0, 1.0)

    spearmanr = wsi_attention_spearmanr(scores, target_attention)
    topk_overlap = _topk_overlap(pred_indices, ideal_indices)
    ndcg_at_k = _ndcg_at_k(target_attention, pred_indices, ideal_indices)

    return SlidePruningMetrics(
        slide_id=slide_id,
        keep_ratio=keep_ratio,
        n_tiles=n_tiles,
        n_tiles_kept=n_tiles_kept,
        effective_keep_ratio=n_tiles_kept / n_tiles,
        spearmanr=float(spearmanr.detach().cpu()),
        topk_overlap=topk_overlap,
        ndcg_at_k=ndcg_at_k,
        attention_mass_retained=float(retained_mass.detach().cpu()),
        oracle_attention_mass_at_k=float(oracle_mass.detach().cpu()),
        relative_attention_mass_retained=float(relative_mass.detach().cpu()),
    )


def _aggregate(rows: list[SlidePruningMetrics]) -> list[dict[str, float | int]]:
    by_ratio: dict[float, list[SlidePruningMetrics]] = {}
    for row in rows:
        by_ratio.setdefault(row.keep_ratio, []).append(row)

    aggregated = []
    for keep_ratio in sorted(by_ratio):
        group = by_ratio[keep_ratio]
        n_slides = len(group)

        def mean(name: str) -> float:
            return sum(float(getattr(row, name)) for row in group) / n_slides

        aggregated.append(
            {
                "keep_ratio": keep_ratio,
                "n_slides": n_slides,
                "n_tiles_total": sum(row.n_tiles for row in group),
                "n_tiles_kept_total": sum(row.n_tiles_kept for row in group),
                "mean_n_tiles": mean("n_tiles"),
                "mean_n_tiles_kept": mean("n_tiles_kept"),
                "mean_effective_keep_ratio": mean("effective_keep_ratio"),
                "mean_spearmanr": mean("spearmanr"),
                "mean_topk_overlap": mean("topk_overlap"),
                "mean_ndcg_at_k": mean("ndcg_at_k"),
                "mean_attention_mass_retained": mean("attention_mass_retained"),
                "mean_oracle_attention_mass_at_k": mean("oracle_attention_mass_at_k"),
                "mean_relative_attention_mass_retained": mean(
                    "relative_attention_mass_retained"
                ),
            }
        )

    return aggregated


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "keep_ratio",
        "n_slides",
        "n_tiles_total",
        "n_tiles_kept_total",
        "mean_n_tiles",
        "mean_n_tiles_kept",
        "mean_effective_keep_ratio",
        "mean_spearmanr",
        "mean_topk_overlap",
        "mean_ndcg_at_k",
        "mean_attention_mass_retained",
        "mean_oracle_attention_mass_at_k",
        "mean_relative_attention_mass_retained",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    device = _resolve_device(args.device)

    store = H5WSIFeatureStore(args.feature_store)
    all_slide_ids = store.slide_ids()
    if not all_slide_ids:
        raise ValueError(f"feature store contains no slides: {args.feature_store}")

    if args.slide_ids_file is None:
        slide_ids = all_slide_ids
    else:
        slide_ids = _read_slide_ids(args.slide_ids_file)

    dataset = FeatureStoreWSIBagDataset(store, slide_ids=slide_ids)
    loader = _make_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint = load_wsi_tile_attention_forecaster_checkpoint(
        args.forecaster_checkpoint,
        map_location=device,
    )
    model = checkpoint.model.to(device)
    model.eval()

    rows: list[SlidePruningMetrics] = []

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "feature_store": str(args.feature_store),
                "forecaster_checkpoint": str(args.forecaster_checkpoint),
                "n_slides": len(slide_ids),
                "keep_ratios": args.keep_ratios,
                "output_csv": str(args.output_csv),
            }
        ),
        flush=True,
    )

    with torch.no_grad():
        for batch in loader:
            if batch.attention is None:
                raise ValueError(
                    "feature store must contain attention targets; "
                    "run extract_wsi_abmil_attention.py first."
                )

            batch = batch.to(device=device)
            scores = model(batch.tile_features, mask=batch.mask)

            for row_index, slide_id in enumerate(batch.slide_ids):
                valid_mask = batch.mask[row_index]
                slide_scores = scores[row_index, valid_mask].detach().cpu()
                slide_attention = batch.attention[row_index, valid_mask].detach().cpu()

                for keep_ratio in args.keep_ratios:
                    rows.append(
                        _evaluate_slide_ratio(
                            slide_id=slide_id,
                            scores=slide_scores,
                            target_attention=slide_attention,
                            keep_ratio=keep_ratio,
                        )
                    )

    aggregated = _aggregate(rows)
    _write_csv(args.output_csv, aggregated)

    print(
        json.dumps(
            {
                "event": "done",
                "n_slides": len(slide_ids),
                "n_rows": len(aggregated),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        ),
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
