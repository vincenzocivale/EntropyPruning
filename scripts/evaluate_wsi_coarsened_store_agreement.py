#!/usr/bin/env python
"""Evaluate a materialized coarsened store against the full WSI store.

The same frozen ABMIL checkpoint is applied to both stores. This isolates the
effect of tile coarsening and complements downstream retraining experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi.h5_feature_store import H5WSIFeatureStore
from src.models.wsi import load_abmil_classifier_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen ABMIL full-vs-coarsened store agreement."
    )
    parser.add_argument("--full-feature-store", type=Path, required=True)
    parser.add_argument("--coarsened-feature-store", type=Path, required=True)
    parser.add_argument("--abmil-checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-slide-csv", type=Path, default=None)
    parser.add_argument("--slide-ids-file", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (
        args.full_feature_store,
        args.coarsened_feature_store,
        args.abmil_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.output_csv, args.per_slide_csv):
        if path is not None and path.exists() and not args.overwrite:
            raise FileExistsError(path)
    return args


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _read_slide_ids(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values:
        raise ValueError(f"slide id file is empty: {path}")
    return values


def _label_to_int(label: int | float | torch.Tensor | None) -> int | None:
    if label is None:
        return None
    if isinstance(label, torch.Tensor):
        if label.numel() != 1:
            return None
        label = label.detach().cpu().item()
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)) and float(label).is_integer():
        return int(label)
    return None


def _selected_full_indices(
    full_coords: torch.Tensor | None,
    coarsened_coords: torch.Tensor | None,
) -> torch.Tensor | None:
    if full_coords is None or coarsened_coords is None:
        return None
    full_keys = [tuple(row.tolist()) for row in full_coords]
    selected_keys = [tuple(row.tolist()) for row in coarsened_coords]
    if len(set(full_keys)) != len(full_keys):
        raise ValueError("full store contains duplicate coordinates.")
    if len(set(selected_keys)) != len(selected_keys):
        raise ValueError("coarsened store contains duplicate coordinates.")
    index_by_key = {key: index for index, key in enumerate(full_keys)}
    missing = [key for key in selected_keys if key not in index_by_key]
    if missing:
        raise ValueError(
            f"coarsened store contains coordinates absent from full store, e.g. {missing[:5]}"
        )
    return torch.tensor([index_by_key[key] for key in selected_keys], dtype=torch.long)


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=1)
        .squeeze(0)
        .detach()
        .cpu()
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    full_store = H5WSIFeatureStore(args.full_feature_store)
    coarsened_store = H5WSIFeatureStore(args.coarsened_feature_store)

    full_ids = full_store.slide_ids()
    coarsened_ids = set(coarsened_store.slide_ids())
    slide_ids = (
        full_ids if args.slide_ids_file is None else _read_slide_ids(args.slide_ids_file)
    )
    missing = [slide_id for slide_id in slide_ids if slide_id not in coarsened_ids]
    if missing:
        raise KeyError(
            f"coarsened store is missing {len(missing)} requested slide(s), e.g. {missing[:10]}"
        )

    checkpoint = load_abmil_classifier_checkpoint(
        args.abmil_checkpoint,
        map_location=device,
    )
    model = checkpoint.model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for slide_id in slide_ids:
            full = full_store.read(slide_id)
            coarsened = coarsened_store.read(slide_id)
            if full.feature_dim != coarsened.feature_dim:
                raise ValueError(
                    f"feature dimension mismatch for {slide_id}: "
                    f"{full.feature_dim} vs {coarsened.feature_dim}"
                )
            if full.feature_dim != checkpoint.config.feature_dim:
                raise ValueError(
                    f"checkpoint expects feature_dim={checkpoint.config.feature_dim}; "
                    f"store has {full.feature_dim}."
                )

            full_output = model(full.tile_features.to(device))
            coarsened_output = model(coarsened.tile_features.to(device))
            full_logits = full_output.logits
            coarsened_logits = coarsened_output.logits
            full_probs = torch.softmax(full_logits, dim=0)
            coarsened_log_probs = torch.log_softmax(coarsened_logits, dim=0)
            label = _label_to_int(full.label)
            coarsened_label = _label_to_int(coarsened.label)
            if label is not None and coarsened_label is not None and label != coarsened_label:
                raise ValueError(f"label mismatch for {slide_id}.")

            full_prediction = int(full_logits.argmax())
            coarsened_prediction = int(coarsened_logits.argmax())
            selected_indices = _selected_full_indices(full.coords, coarsened.coords)
            attention_mass = None
            oracle_mass = None
            relative_mass = None
            if (
                full.attention is not None
                and selected_indices is not None
                and float(full.attention.sum()) > 0
            ):
                target = full.attention.to(torch.float32)
                total = target.sum()
                attention_mass = float(target[selected_indices].sum() / total)
                oracle_indices = torch.topk(target, k=coarsened.n_tiles).indices
                oracle_mass = float(target[oracle_indices].sum() / total)
                relative_mass = attention_mass / max(oracle_mass, 1e-12)

            row = {
                "slide_id": slide_id,
                "n_tiles_full": full.n_tiles,
                "n_tiles_coarsened": coarsened.n_tiles,
                "effective_keep_ratio": coarsened.n_tiles / full.n_tiles,
                "label": label,
                "full_prediction": full_prediction,
                "coarsened_prediction": coarsened_prediction,
                "prediction_agreement": float(full_prediction == coarsened_prediction),
                "full_correct": (
                    float(full_prediction == label) if label is not None else None
                ),
                "coarsened_correct": (
                    float(coarsened_prediction == label) if label is not None else None
                ),
                "logit_cosine_similarity": _safe_cosine(full_logits, coarsened_logits),
                "bag_embedding_cosine_similarity": _safe_cosine(
                    full_output.bag_embedding,
                    coarsened_output.bag_embedding,
                ),
                "prob_kl_full_to_coarsened": float(
                    F.kl_div(coarsened_log_probs, full_probs, reduction="sum")
                    .detach()
                    .cpu()
                ),
                "attention_mass_retained": attention_mass,
                "oracle_attention_mass_at_k": oracle_mass,
                "relative_attention_mass_retained": relative_mass,
            }
            rows.append(row)

    aggregate = {
        "n_slides": len(rows),
        "n_tiles_full_total": sum(int(row["n_tiles_full"]) for row in rows),
        "n_tiles_coarsened_total": sum(
            int(row["n_tiles_coarsened"]) for row in rows
        ),
        "mean_effective_keep_ratio": _mean(rows, "effective_keep_ratio"),
        "full_accuracy": _mean(rows, "full_correct"),
        "coarsened_accuracy": _mean(rows, "coarsened_correct"),
        "prediction_agreement": _mean(rows, "prediction_agreement"),
        "mean_logit_cosine_similarity": _mean(rows, "logit_cosine_similarity"),
        "mean_bag_embedding_cosine_similarity": _mean(
            rows, "bag_embedding_cosine_similarity"
        ),
        "mean_prob_kl_full_to_coarsened": _mean(
            rows, "prob_kl_full_to_coarsened"
        ),
        "mean_attention_mass_retained": _mean(rows, "attention_mass_retained"),
        "mean_oracle_attention_mass_at_k": _mean(
            rows, "oracle_attention_mass_at_k"
        ),
        "mean_relative_attention_mass_retained": _mean(
            rows, "relative_attention_mass_retained"
        ),
    }
    _write_csv(args.output_csv, [aggregate])
    if args.per_slide_csv is not None:
        _write_csv(args.per_slide_csv, rows)
    print(json.dumps({"event": "done", **aggregate}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
