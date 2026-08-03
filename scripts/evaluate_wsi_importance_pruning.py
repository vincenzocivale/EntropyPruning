#!/usr/bin/env python
"""Evaluate WSI tile pruning driven by a tile-importance forecaster.

Unlike ``evaluate_wsi_forecaster_pruning.py`` (kept unchanged, single-store
only), this script accepts an input/selection feature store and a target
feature store separately (``--input-feature-store``/``--target-feature-store``),
so a forecaster trained on early-layer tile features can be evaluated against
a tile-importance target that was produced independently (ABMIL attention, a
WSI foundation model tile score, or any other precomputed target), optionally
aligned by tile coordinates rather than array index. ``--feature-store``
remains as a single-store convenience alias for the legacy case.

If ``--abmil-checkpoint`` is provided, this script additionally evaluates
full-vs-pruned ABMIL prediction agreement (accuracy, agreement, logit
similarity) using the same selection features the forecaster scored. This is
optional: without an ABMIL checkpoint, only target-importance recovery
metrics are reported.
"""

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
    PairedFeatureStoreWSIBagDataset,
    H5WSIFeatureStore,
    WSIFeatureStore,
    collate_padded_wsi_bags,
)
from src.evaluation.wsi_attention_metrics import wsi_attention_spearmanr
from src.models.wsi import (
    load_abmil_classifier_checkpoint,
    load_wsi_tile_importance_forecaster_checkpoint,
)

_ALIGNMENT_MODES = ("index", "coords")


@dataclass(frozen=True)
class SlideImportancePruningMetrics:
    slide_id: str
    keep_ratio: float
    n_tiles: int
    n_tiles_kept: int
    effective_keep_ratio: float
    spearmanr: float
    topk_overlap: float
    ndcg_at_k: float
    target_importance_mass_retained: float
    oracle_importance_mass_retained: float
    relative_importance_mass_retained: float
    label: int | None = None
    full_prediction: int | None = None
    pruned_prediction: int | None = None
    full_correct: float | None = None
    pruned_correct: float | None = None
    prediction_agreement: float | None = None
    logit_cosine_similarity: float | None = None
    prob_kl_full_to_pruned: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate WSI tile pruning from a tile-importance forecaster, "
            "using paired input/target HDF5 feature stores (or a single "
            "legacy fused store)."
        )
    )

    parser.add_argument(
        "--feature-store",
        type=Path,
        default=None,
        help=(
            "Legacy single-store convenience alias: use the same store as "
            "both --input-feature-store and --target-feature-store. "
            "Mutually exclusive with the paired-store flags."
        ),
    )
    parser.add_argument("--input-feature-store", type=Path, default=None)
    parser.add_argument("--target-feature-store", type=Path, default=None)

    parser.add_argument("--forecaster-checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--keep-ratios", type=float, nargs="+", required=True)

    parser.add_argument(
        "--alignment-mode",
        type=str,
        choices=_ALIGNMENT_MODES,
        default="index",
        help=(
            "'index' requires identical slide id, tile count, and (if both "
            "sides have coords) identical tile order. 'coords' aligns tiles "
            "by exact coordinate match and requires coords on both sides."
        ),
    )
    parser.add_argument("--require-coords", action="store_true")

    parser.add_argument(
        "--abmil-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional ABMIL classifier checkpoint. When provided, also "
            "evaluates full-vs-pruned prediction agreement using the "
            "selection-store tile features. Not required: without it, only "
            "target-importance recovery metrics are reported."
        ),
    )

    parser.add_argument("--slide-ids-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    has_legacy_store = args.feature_store is not None
    has_paired_stores = (
        args.input_feature_store is not None or args.target_feature_store is not None
    )

    if has_legacy_store and has_paired_stores:
        raise ValueError(
            "--feature-store is mutually exclusive with "
            "--input-feature-store/--target-feature-store."
        )

    if not has_legacy_store:
        if args.input_feature_store is None or args.target_feature_store is None:
            raise ValueError(
                "either --feature-store, or both --input-feature-store and "
                "--target-feature-store, must be provided."
            )

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
    dataset: PairedFeatureStoreWSIBagDataset,
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


def _infer_target_source(
    target_store: WSIFeatureStore,
    slide_ids: tuple[str, ...],
) -> str | None:
    for slide_id in slide_ids[:5]:
        bag = target_store.read(slide_id)
        if not bag.metadata:
            continue
        for key in ("target_source", "attention_source", "source"):
            if key in bag.metadata:
                return str(bag.metadata[key])

    return None


def _topk_overlap(pred_indices: torch.Tensor, target_indices: torch.Tensor) -> float:
    pred_set = set(pred_indices.detach().cpu().tolist())
    target_set = set(target_indices.detach().cpu().tolist())
    return len(pred_set.intersection(target_set)) / len(pred_set)


def _ndcg_at_k(target_importance: torch.Tensor, pred_indices: torch.Tensor, ideal_indices: torch.Tensor) -> float:
    k = int(pred_indices.numel())
    discounts = 1.0 / torch.log2(
        torch.arange(k, device=target_importance.device, dtype=torch.float32) + 2.0
    )

    dcg = (target_importance[pred_indices].to(torch.float32) * discounts).sum()
    ideal_dcg = (target_importance[ideal_indices].to(torch.float32) * discounts).sum()

    if ideal_dcg <= 0:
        raise ValueError("ideal DCG must be positive.")

    return float((dcg / ideal_dcg).detach().cpu())


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


def _evaluate_slide_ratio(
    *,
    slide_id: str,
    scores: torch.Tensor,
    target_importance: torch.Tensor,
    keep_ratio: float,
    tile_features: torch.Tensor | None,
    label: int | None,
    full_logits: torch.Tensor | None,
    abmil_model: torch.nn.Module | None,
) -> SlideImportancePruningMetrics:
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D; got {tuple(scores.shape)}.")
    if target_importance.ndim != 1:
        raise ValueError(
            f"target_importance must be 1D; got {tuple(target_importance.shape)}."
        )
    if scores.shape != target_importance.shape:
        raise ValueError(
            "scores and target_importance must have the same shape; "
            f"got {tuple(scores.shape)} and {tuple(target_importance.shape)}."
        )
    if scores.numel() == 0:
        raise ValueError("each slide must contain at least one tile.")
    if not torch.isfinite(scores).all():
        raise ValueError(f"scores contain NaN or Inf for slide {slide_id}.")
    if not torch.isfinite(target_importance).all():
        raise ValueError(f"target_importance contains NaN or Inf for slide {slide_id}.")
    if (target_importance < 0).any():
        raise ValueError(f"target_importance contains negative values for slide {slide_id}.")

    importance_mass = target_importance.sum()
    if importance_mass <= 0:
        raise ValueError(f"target_importance mass must be positive for slide {slide_id}.")

    n_tiles = int(scores.numel())
    n_tiles_kept = max(1, math.ceil(n_tiles * keep_ratio))
    n_tiles_kept = min(n_tiles_kept, n_tiles)

    pred_indices = torch.topk(scores, k=n_tiles_kept).indices
    ideal_indices = torch.topk(target_importance, k=n_tiles_kept).indices

    retained_mass = (target_importance[pred_indices].sum() / importance_mass).clamp(0.0, 1.0)
    oracle_mass = (target_importance[ideal_indices].sum() / importance_mass).clamp(0.0, 1.0)
    relative_mass = (retained_mass / oracle_mass).clamp(0.0, 1.0)

    spearmanr = wsi_attention_spearmanr(scores, target_importance)
    topk_overlap = _topk_overlap(pred_indices, ideal_indices)
    ndcg_at_k = _ndcg_at_k(target_importance, pred_indices, ideal_indices)

    agreement_fields: dict[str, float | int] = {}
    if abmil_model is not None:
        if tile_features is None or full_logits is None or label is None:
            raise ValueError(
                f"slide {slide_id}: ABMIL evaluation requires tile_features, "
                "full_logits, and label."
            )

        pruned_features = tile_features[pred_indices.to(tile_features.device)]
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

        agreement_fields = {
            "label": label,
            "full_prediction": full_prediction,
            "pruned_prediction": pruned_prediction,
            "full_correct": float(full_prediction == label),
            "pruned_correct": float(pruned_prediction == label),
            "prediction_agreement": float(full_prediction == pruned_prediction),
            "logit_cosine_similarity": float(cosine.detach().cpu()),
            "prob_kl_full_to_pruned": float(kl.detach().cpu()),
        }

    return SlideImportancePruningMetrics(
        slide_id=slide_id,
        keep_ratio=keep_ratio,
        n_tiles=n_tiles,
        n_tiles_kept=n_tiles_kept,
        effective_keep_ratio=n_tiles_kept / n_tiles,
        spearmanr=float(spearmanr.detach().cpu()),
        topk_overlap=topk_overlap,
        ndcg_at_k=ndcg_at_k,
        target_importance_mass_retained=float(retained_mass.detach().cpu()),
        oracle_importance_mass_retained=float(oracle_mass.detach().cpu()),
        relative_importance_mass_retained=float(relative_mass.detach().cpu()),
        **agreement_fields,
    )


def _aggregate(
    rows: list[SlideImportancePruningMetrics], *, with_agreement: bool
) -> list[dict[str, float | int]]:
    by_ratio: dict[float, list[SlideImportancePruningMetrics]] = {}
    for row in rows:
        by_ratio.setdefault(row.keep_ratio, []).append(row)

    aggregated = []
    for keep_ratio in sorted(by_ratio):
        group = by_ratio[keep_ratio]
        n_slides = len(group)

        def mean(name: str) -> float:
            return sum(float(getattr(row, name)) for row in group) / n_slides

        record = {
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
            "mean_target_importance_mass_retained": mean("target_importance_mass_retained"),
            "mean_oracle_importance_mass_retained": mean("oracle_importance_mass_retained"),
            "mean_relative_importance_mass_retained": mean("relative_importance_mass_retained"),
        }

        if with_agreement:
            record.update(
                {
                    "full_accuracy": mean("full_correct"),
                    "pruned_accuracy": mean("pruned_correct"),
                    "prediction_agreement": mean("prediction_agreement"),
                    "mean_logit_cosine_similarity": mean("logit_cosine_similarity"),
                    "mean_prob_kl_full_to_pruned": mean("prob_kl_full_to_pruned"),
                }
            )

        aggregated.append(record)

    return aggregated


def _write_csv(path: Path, rows: list[dict[str, float | int]], *, with_agreement: bool) -> None:
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
        "mean_target_importance_mass_retained",
        "mean_oracle_importance_mass_retained",
        "mean_relative_importance_mass_retained",
    ]
    if with_agreement:
        fieldnames += [
            "full_accuracy",
            "pruned_accuracy",
            "prediction_agreement",
            "mean_logit_cosine_similarity",
            "mean_prob_kl_full_to_pruned",
        ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    device = _resolve_device(args.device)

    if args.feature_store is not None:
        input_path = args.feature_store
        target_path = args.feature_store
    else:
        input_path = args.input_feature_store
        target_path = args.target_feature_store

    input_store = H5WSIFeatureStore(input_path)
    target_store = H5WSIFeatureStore(target_path)

    input_slide_ids = set(input_store.slide_ids())
    target_slide_ids = set(target_store.slide_ids())
    common_slide_ids = tuple(sorted(input_slide_ids & target_slide_ids))

    if not common_slide_ids:
        raise ValueError(
            "no slide ids common to --input-feature-store and "
            "--target-feature-store."
        )

    if args.slide_ids_file is None:
        slide_ids = common_slide_ids
    else:
        slide_ids = _read_slide_ids(args.slide_ids_file)

    dataset = PairedFeatureStoreWSIBagDataset(
        input_store,
        target_store,
        slide_ids=slide_ids,
        alignment_mode=args.alignment_mode,
        require_coords=args.require_coords,
    )
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

    abmil_model = None
    if args.abmil_checkpoint is not None:
        abmil_checkpoint = load_abmil_classifier_checkpoint(
            args.abmil_checkpoint,
            map_location=device,
        )
        if abmil_checkpoint.config.feature_dim != checkpoint.config.feature_dim:
            raise ValueError(
                "ABMIL and forecaster checkpoints use different feature dimensions: "
                f"{abmil_checkpoint.config.feature_dim} vs {checkpoint.config.feature_dim}."
            )
        abmil_model = abmil_checkpoint.model.to(device)
        abmil_model.eval()

    with_agreement = abmil_model is not None
    target_source = checkpoint.metadata.get("target_source") or _infer_target_source(
        target_store, slide_ids
    )

    rows: list[SlideImportancePruningMetrics] = []

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "input_feature_store": str(input_path),
                "target_feature_store": str(target_path),
                "forecaster_checkpoint": str(args.forecaster_checkpoint),
                "abmil_checkpoint": str(args.abmil_checkpoint) if args.abmil_checkpoint else None,
                "alignment_mode": args.alignment_mode,
                "target_source": target_source,
                "n_slides": len(slide_ids),
                "keep_ratios": args.keep_ratios,
                "output_csv": str(args.output_csv),
            }
        ),
        flush=True,
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device)
            scores = model(batch.tile_features, mask=batch.mask)

            full_abmil_output = None
            if abmil_model is not None:
                full_abmil_output = abmil_model(batch.tile_features, mask=batch.mask)

            for row_index, slide_id in enumerate(batch.slide_ids):
                valid_mask = batch.mask[row_index]
                slide_scores = scores[row_index, valid_mask].detach().cpu()
                slide_target = batch.attention[row_index, valid_mask].detach().cpu()

                slide_features = None
                slide_label = None
                slide_full_logits = None
                if abmil_model is not None:
                    slide_features = batch.tile_features[row_index, valid_mask]
                    slide_label = _label_to_int(batch.labels[row_index], slide_id=slide_id)
                    slide_full_logits = full_abmil_output.logits[row_index]

                for keep_ratio in args.keep_ratios:
                    rows.append(
                        _evaluate_slide_ratio(
                            slide_id=slide_id,
                            scores=slide_scores,
                            target_importance=slide_target,
                            keep_ratio=keep_ratio,
                            tile_features=slide_features,
                            label=slide_label,
                            full_logits=slide_full_logits,
                            abmil_model=abmil_model,
                        )
                    )

    aggregated = _aggregate(rows, with_agreement=with_agreement)
    _write_csv(args.output_csv, aggregated, with_agreement=with_agreement)

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
