#!/usr/bin/env python
"""Evaluate ABMIL full-vs-pruned agreement using forecaster-selected tiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import (
    FeatureStoreWSIBagDataset,
    H5WSIFeatureStore,
    collate_padded_wsi_bags,
)
from src.models.wsi import (
    load_abmil_classifier_checkpoint,
    load_wsi_tile_attention_forecaster_checkpoint,
)


@dataclass(frozen=True)
class SlideAgreementMetrics:
    slide_id: str
    keep_ratio: float
    n_tiles: int
    n_tiles_kept: int
    effective_keep_ratio: float
    label: int
    full_prediction: int
    pruned_prediction: int
    full_correct: float
    pruned_correct: float
    prediction_agreement: float
    logit_cosine_similarity: float
    prob_kl_full_to_pruned: float
    attention_mass_retained: float
    oracle_attention_mass_at_k: float
    relative_attention_mass_retained: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether forecaster-selected tiles preserve ABMIL teacher "
            "predictions."
        )
    )

    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--abmil-checkpoint", type=Path, required=True)
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


def _label_to_int(label: int | float | torch.Tensor | None, *, slide_id: str) -> int:
    if label is None:
        raise ValueError(f"label is required for slide {slide_id}.")

    if isinstance(label, bool):
        raise TypeError(f"boolean label is not supported for slide {slide_id}.")

    if isinstance(label, int):
        value = label
    elif isinstance(label, float):
        if not label.is_integer():
            raise TypeError(f"label must be integer-valued for slide {slide_id}.")
        value = int(label)
    elif isinstance(label, torch.Tensor):
        if label.numel() != 1:
            raise ValueError(f"tensor label must be scalar for slide {slide_id}.")
        raw_value = label.detach().cpu().item()
        if isinstance(raw_value, bool):
            raise TypeError(f"boolean tensor label is not supported for slide {slide_id}.")
        if isinstance(raw_value, float) and not float(raw_value).is_integer():
            raise TypeError(f"tensor label must be integer-valued for slide {slide_id}.")
        value = int(raw_value)
    else:
        raise TypeError(
            f"label for slide {slide_id} must be int, integer float, or scalar tensor; "
            f"got {type(label).__name__}."
        )

    if value < 0:
        raise ValueError(f"label must be non-negative for slide {slide_id}.")

    return value


def _attention_mass_metrics(
    *,
    target_attention: torch.Tensor,
    selected_indices: torch.Tensor,
    k: int,
    slide_id: str,
) -> tuple[float, float, float]:
    if target_attention.ndim != 1:
        raise ValueError(
            f"target_attention must be 1D for slide {slide_id}; "
            f"got {tuple(target_attention.shape)}."
        )
    if not torch.isfinite(target_attention).all():
        raise ValueError(f"attention contains NaN or Inf for slide {slide_id}.")
    if (target_attention < 0).any():
        raise ValueError(f"attention contains negative values for slide {slide_id}.")

    attention_mass = target_attention.sum()
    if attention_mass <= 0:
        raise ValueError(f"attention mass must be positive for slide {slide_id}.")

    ideal_indices = torch.topk(target_attention, k=k).indices

    retained = (target_attention[selected_indices].sum() / attention_mass).clamp(0.0, 1.0)
    oracle = (target_attention[ideal_indices].sum() / attention_mass).clamp(0.0, 1.0)
    relative = (retained / oracle).clamp(0.0, 1.0)

    return (
        float(retained.detach().cpu()),
        float(oracle.detach().cpu()),
        float(relative.detach().cpu()),
    )


def _evaluate_slide_ratio(
    *,
    slide_id: str,
    label: int,
    tile_features: torch.Tensor,
    target_attention: torch.Tensor,
    forecaster_scores: torch.Tensor,
    full_logits: torch.Tensor,
    abmil_model: torch.nn.Module,
    keep_ratio: float,
) -> SlideAgreementMetrics:
    if tile_features.ndim != 2:
        raise ValueError(
            f"tile_features must be 2D for slide {slide_id}; "
            f"got {tuple(tile_features.shape)}."
        )
    if forecaster_scores.ndim != 1:
        raise ValueError(
            f"forecaster_scores must be 1D for slide {slide_id}; "
            f"got {tuple(forecaster_scores.shape)}."
        )
    if full_logits.ndim != 1:
        raise ValueError(
            f"full_logits must be 1D for slide {slide_id}; "
            f"got {tuple(full_logits.shape)}."
        )

    n_tiles = int(tile_features.shape[0])
    if n_tiles == 0:
        raise ValueError(f"slide {slide_id} contains no tiles.")
    if forecaster_scores.shape[0] != n_tiles:
        raise ValueError(
            f"forecaster score length does not match tile count for slide {slide_id}."
        )

    n_tiles_kept = max(1, math.ceil(n_tiles * keep_ratio))
    n_tiles_kept = min(n_tiles_kept, n_tiles)

    selected_indices = torch.topk(forecaster_scores, k=n_tiles_kept).indices
    pruned_features = tile_features[selected_indices]

    pruned_output = abmil_model(pruned_features)
    pruned_logits = pruned_output.logits

    full_probs = torch.softmax(full_logits, dim=0)
    pruned_log_probs = torch.log_softmax(pruned_logits, dim=0)

    full_prediction = int(full_logits.argmax(dim=0).detach().cpu())
    pruned_prediction = int(pruned_logits.argmax(dim=0).detach().cpu())

    cosine = F.cosine_similarity(
        full_logits.unsqueeze(0),
        pruned_logits.unsqueeze(0),
        dim=1,
    ).squeeze(0)
    kl = F.kl_div(pruned_log_probs, full_probs, reduction="sum")

    retained, oracle, relative = _attention_mass_metrics(
        target_attention=target_attention,
        selected_indices=selected_indices,
        k=n_tiles_kept,
        slide_id=slide_id,
    )

    return SlideAgreementMetrics(
        slide_id=slide_id,
        keep_ratio=keep_ratio,
        n_tiles=n_tiles,
        n_tiles_kept=n_tiles_kept,
        effective_keep_ratio=n_tiles_kept / n_tiles,
        label=label,
        full_prediction=full_prediction,
        pruned_prediction=pruned_prediction,
        full_correct=float(full_prediction == label),
        pruned_correct=float(pruned_prediction == label),
        prediction_agreement=float(full_prediction == pruned_prediction),
        logit_cosine_similarity=float(cosine.detach().cpu()),
        prob_kl_full_to_pruned=float(kl.detach().cpu()),
        attention_mass_retained=retained,
        oracle_attention_mass_at_k=oracle,
        relative_attention_mass_retained=relative,
    )


def _aggregate(rows: list[SlideAgreementMetrics]) -> list[dict[str, float | int]]:
    by_ratio: dict[float, list[SlideAgreementMetrics]] = {}
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
                "full_accuracy": mean("full_correct"),
                "pruned_accuracy": mean("pruned_correct"),
                "prediction_agreement": mean("prediction_agreement"),
                "mean_logit_cosine_similarity": mean("logit_cosine_similarity"),
                "mean_prob_kl_full_to_pruned": mean("prob_kl_full_to_pruned"),
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
        "full_accuracy",
        "pruned_accuracy",
        "prediction_agreement",
        "mean_logit_cosine_similarity",
        "mean_prob_kl_full_to_pruned",
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

    abmil_checkpoint = load_abmil_classifier_checkpoint(
        args.abmil_checkpoint,
        map_location=device,
    )
    forecaster_checkpoint = load_wsi_tile_attention_forecaster_checkpoint(
        args.forecaster_checkpoint,
        map_location=device,
    )

    if abmil_checkpoint.config.feature_dim != forecaster_checkpoint.config.feature_dim:
        raise ValueError(
            "ABMIL and forecaster checkpoints use different feature dimensions: "
            f"{abmil_checkpoint.config.feature_dim} vs "
            f"{forecaster_checkpoint.config.feature_dim}."
        )

    abmil_model = abmil_checkpoint.model.to(device)
    forecaster_model = forecaster_checkpoint.model.to(device)
    abmil_model.eval()
    forecaster_model.eval()

    rows: list[SlideAgreementMetrics] = []

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "feature_store": str(args.feature_store),
                "abmil_checkpoint": str(args.abmil_checkpoint),
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

            full_abmil_output = abmil_model(batch.tile_features, mask=batch.mask)
            forecaster_scores = forecaster_model(batch.tile_features, mask=batch.mask)

            for row_index, slide_id in enumerate(batch.slide_ids):
                valid_mask = batch.mask[row_index]
                label = _label_to_int(batch.labels[row_index], slide_id=slide_id)

                slide_features = batch.tile_features[row_index, valid_mask]
                slide_attention = batch.attention[row_index, valid_mask]
                slide_scores = forecaster_scores[row_index, valid_mask]
                full_logits = full_abmil_output.logits[row_index]

                for keep_ratio in args.keep_ratios:
                    rows.append(
                        _evaluate_slide_ratio(
                            slide_id=slide_id,
                            label=label,
                            tile_features=slide_features,
                            target_attention=slide_attention,
                            forecaster_scores=slide_scores,
                            full_logits=full_logits,
                            abmil_model=abmil_model,
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
