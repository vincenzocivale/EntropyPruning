#!/usr/bin/env python
"""Create a pruned WSI HDF5 feature store using forecaster-selected tiles.

Two usage modes are supported:

- Legacy single-store mode (``--input-feature-store``): tiles are selected
  and materialized from the same store, exactly as before. This mode's
  behavior and output metadata are unchanged.
- Selection/materialize mode (``--selection-feature-store``, optionally with
  a different ``--materialize-feature-store``): tiles are *selected* using
  early-layer (or otherwise separate) features, and the corresponding late
  features are *materialized* into the output store. When the materialize
  store differs from the selection store, tiles are aligned either by array
  index (``--alignment-mode index``, the default) or by exact tile
  coordinate match (``--alignment-mode coords``).
"""

from __future__ import annotations

import argparse
import json
import math
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
    WSIFeatureStore,
    align_by_coords,
    collate_padded_wsi_bags,
)
from src.models.wsi import (
    load_wsi_tile_attention_forecaster_checkpoint,
    load_wsi_tile_importance_forecaster_checkpoint,
)

_ALIGNMENT_MODES = ("index", "coords")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pruned HDF5 WSI feature store from forecaster scores."
    )

    parser.add_argument(
        "--input-feature-store",
        type=Path,
        default=None,
        help=(
            "Legacy single-store mode: tiles are selected and materialized "
            "from this same store. Mutually exclusive with "
            "--selection-feature-store/--materialize-feature-store."
        ),
    )
    parser.add_argument(
        "--selection-feature-store",
        type=Path,
        default=None,
        help=(
            "Feature store used to score tiles (e.g. early-layer features). "
            "Mutually exclusive with --input-feature-store."
        ),
    )
    parser.add_argument(
        "--materialize-feature-store",
        type=Path,
        default=None,
        help=(
            "Feature store providing the tile features written to the "
            "output store (e.g. late-layer features). Defaults to "
            "--selection-feature-store when omitted. Requires "
            "--selection-feature-store."
        ),
    )

    parser.add_argument("--output-feature-store", type=Path, required=True)
    parser.add_argument("--forecaster-checkpoint", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)

    parser.add_argument(
        "--alignment-mode",
        type=str,
        choices=_ALIGNMENT_MODES,
        default="index",
        help=(
            "Used only when --materialize-feature-store differs from "
            "--selection-feature-store. 'index' requires identical tile "
            "counts (and, if both sides have coords, identical tile order). "
            "'coords' aligns tiles by exact coordinate match and requires "
            "coords on both sides."
        ),
    )
    parser.add_argument("--require-coords", action="store_true")

    parser.add_argument("--slide-ids-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.keep_ratio <= 1.0:
        raise ValueError("--keep-ratio must be in (0, 1].")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    has_legacy = args.input_feature_store is not None
    has_selection = args.selection_feature_store is not None

    if has_legacy and has_selection:
        raise ValueError(
            "--input-feature-store is mutually exclusive with "
            "--selection-feature-store/--materialize-feature-store."
        )
    if not has_legacy and not has_selection:
        raise ValueError(
            "either --input-feature-store, or --selection-feature-store, "
            "must be provided."
        )
    if has_legacy and args.materialize_feature_store is not None:
        raise ValueError(
            "--materialize-feature-store requires --selection-feature-store, "
            "not --input-feature-store."
        )

    selection_path = args.input_feature_store if has_legacy else args.selection_feature_store
    materialize_path = (
        args.materialize_feature_store
        if args.materialize_feature_store is not None
        else selection_path
    )

    output_path = args.output_feature_store.resolve()
    if selection_path.resolve() == output_path:
        raise ValueError(
            "--output-feature-store must be different from the selection "
            "feature store."
        )
    if materialize_path.resolve() == output_path:
        raise ValueError(
            "--output-feature-store must be different from the materialize "
            "feature store."
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


def _selected_indices_for_ratio(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D; got {tuple(scores.shape)}.")
    if scores.numel() == 0:
        raise ValueError("cannot prune an empty slide.")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf.")

    n_tiles = int(scores.numel())
    n_keep = max(1, math.ceil(n_tiles * keep_ratio))
    n_keep = min(n_keep, n_tiles)

    # Select by score, then sort indices to preserve original tile order.
    selected = torch.topk(scores, k=n_keep).indices
    return torch.sort(selected).values


def _run_legacy(args: argparse.Namespace, device: torch.device) -> dict:
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

    checkpoint = load_wsi_tile_attention_forecaster_checkpoint(
        args.forecaster_checkpoint,
        map_location=device,
    )
    model = checkpoint.model.to(device)
    model.eval()

    output_store = H5WSIFeatureStore(args.output_feature_store)

    n_slides_written = 0
    n_tiles_input_total = 0
    n_tiles_output_total = 0
    attention_mass_retained_values: list[float] = []

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "input_feature_store": str(args.input_feature_store),
                "output_feature_store": str(args.output_feature_store),
                "forecaster_checkpoint": str(args.forecaster_checkpoint),
                "keep_ratio": args.keep_ratio,
                "n_slides": len(slide_ids),
            }
        ),
        flush=True,
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device)
            scores = model(batch.tile_features, mask=batch.mask)

            for row_index, slide_id in enumerate(batch.slide_ids):
                valid_mask = batch.mask[row_index]
                slide_scores = scores[row_index, valid_mask]
                selected = _selected_indices_for_ratio(slide_scores, args.keep_ratio)

                slide_features = batch.tile_features[row_index, valid_mask]
                pruned_features = slide_features[selected].detach().cpu()

                pruned_coords = None
                if batch.coords is not None:
                    slide_coords = batch.coords[row_index, valid_mask]
                    pruned_coords = slide_coords[selected].detach().cpu()

                pruned_attention = None
                if batch.attention is not None:
                    slide_attention = batch.attention[row_index, valid_mask]
                    pruned_attention = slide_attention[selected].detach().cpu()

                    attention_mass = slide_attention.sum()
                    if attention_mass > 0:
                        retained = pruned_attention.sum().to(slide_attention.device) / attention_mass
                        attention_mass_retained_values.append(
                            float(retained.clamp(0.0, 1.0).detach().cpu())
                        )

                label = batch.labels[row_index] if batch.labels else None
                if isinstance(label, torch.Tensor):
                    label = label.detach().cpu()

                metadata = batch.metadata[row_index] if batch.metadata else None
                metadata = dict(metadata) if metadata is not None else {}
                metadata.update(
                    {
                        "pruned_by": "WSITileAttentionForecaster",
                        "forecaster_checkpoint": str(args.forecaster_checkpoint),
                        "pruning_keep_ratio": args.keep_ratio,
                        "pruning_input_n_tiles": int(slide_scores.numel()),
                        "pruning_output_n_tiles": int(selected.numel()),
                        "pruning_preserved_original_tile_order": True,
                    }
                )

                output_store.write(
                    WSIBag(
                        slide_id=slide_id,
                        tile_features=pruned_features,
                        coords=pruned_coords,
                        label=label,
                        attention=pruned_attention,
                        metadata=metadata,
                    )
                )

                n_slides_written += 1
                n_tiles_input_total += int(slide_scores.numel())
                n_tiles_output_total += int(selected.numel())

    return {
        "event": "done",
        "input_feature_store": str(args.input_feature_store),
        "output_feature_store": str(args.output_feature_store),
        "forecaster_checkpoint": str(args.forecaster_checkpoint),
        "keep_ratio": args.keep_ratio,
        "n_slides": n_slides_written,
        "n_tiles_input_total": n_tiles_input_total,
        "n_tiles_output_total": n_tiles_output_total,
        "effective_keep_ratio": (
            n_tiles_output_total / n_tiles_input_total
            if n_tiles_input_total > 0
            else None
        ),
        "n_slides_with_attention_mass": len(attention_mass_retained_values),
        "mean_attention_mass_retained": (
            sum(attention_mass_retained_values) / len(attention_mass_retained_values)
            if attention_mass_retained_values
            else None
        ),
    }


def _materialize_indices(
    *,
    slide_id: str,
    selection_coords: torch.Tensor | None,
    materialize_bag: WSIBag,
    n_selection_tiles: int,
    selected: torch.Tensor,
    alignment_mode: str,
) -> torch.Tensor:
    """Map ``selected`` (indices into the selection tile order) onto indices
    into ``materialize_bag``'s own tile order.
    """

    if alignment_mode == "index":
        if materialize_bag.n_tiles != n_selection_tiles:
            raise ValueError(
                f"slide {slide_id}: index alignment requires equal tile counts; "
                f"got selection={n_selection_tiles}, materialize={materialize_bag.n_tiles}."
            )
        if (
            selection_coords is not None
            and materialize_bag.coords is not None
            and not torch.equal(selection_coords, materialize_bag.coords)
        ):
            raise ValueError(
                f"slide {slide_id}: index alignment requires identical tile "
                "order, but selection and materialize coords differ. Use "
                "--alignment-mode coords instead."
            )
        return selected

    # alignment_mode == "coords"
    if selection_coords is None or materialize_bag.coords is None:
        raise ValueError(
            f"slide {slide_id}: --alignment-mode coords requires coords in "
            "both the selection and materialize feature stores."
        )

    _, mapping = align_by_coords(selection_coords, materialize_bag.coords)
    return mapping[selected]


def _run_selection_materialize(args: argparse.Namespace, device: torch.device) -> dict:
    selection_path = args.selection_feature_store
    materialize_path = (
        args.materialize_feature_store
        if args.materialize_feature_store is not None
        else selection_path
    )
    same_store = materialize_path.resolve() == selection_path.resolve()

    selection_store = H5WSIFeatureStore(selection_path)
    materialize_store: WSIFeatureStore = (
        selection_store if same_store else H5WSIFeatureStore(materialize_path)
    )

    all_slide_ids = selection_store.slide_ids()
    if not all_slide_ids:
        raise ValueError(f"selection feature store contains no slides: {selection_path}")

    if args.slide_ids_file is None:
        slide_ids = all_slide_ids
    else:
        slide_ids = _read_slide_ids(args.slide_ids_file)

    dataset = FeatureStoreWSIBagDataset(selection_store, slide_ids=slide_ids)
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

    model_type = checkpoint.metadata.get("model_type", "WSITileImportanceForecaster")
    target_source = checkpoint.metadata.get("target_source")

    output_store = H5WSIFeatureStore(args.output_feature_store)

    n_slides_written = 0
    n_tiles_input_total = 0
    n_tiles_output_total = 0

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "selection_feature_store": str(selection_path),
                "materialize_feature_store": str(materialize_path),
                "output_feature_store": str(args.output_feature_store),
                "forecaster_checkpoint": str(args.forecaster_checkpoint),
                "keep_ratio": args.keep_ratio,
                "alignment_mode": args.alignment_mode,
                "n_slides": len(slide_ids),
            }
        ),
        flush=True,
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device)
            scores = model(batch.tile_features, mask=batch.mask)

            for row_index, slide_id in enumerate(batch.slide_ids):
                valid_mask = batch.mask[row_index]
                slide_scores = scores[row_index, valid_mask]
                n_selection_tiles = int(slide_scores.numel())
                selected = _selected_indices_for_ratio(slide_scores, args.keep_ratio)

                if same_store:
                    slide_features = batch.tile_features[row_index, valid_mask]
                    materialize_selected = selected
                    pruned_features = slide_features[materialize_selected].detach().cpu()

                    pruned_coords = None
                    if batch.coords is not None:
                        pruned_coords = (
                            batch.coords[row_index, valid_mask][materialize_selected]
                            .detach()
                            .cpu()
                        )

                    pruned_attention = None
                    if batch.attention is not None:
                        pruned_attention = (
                            batch.attention[row_index, valid_mask][materialize_selected]
                            .detach()
                            .cpu()
                        )

                    label = batch.labels[row_index] if batch.labels else None
                    metadata = dict(batch.metadata[row_index]) if batch.metadata and batch.metadata[row_index] else {}
                else:
                    selection_coords = (
                        batch.coords[row_index, valid_mask].detach().cpu()
                        if batch.coords is not None
                        else None
                    )
                    materialize_bag = materialize_store.read(slide_id)

                    materialize_selected = _materialize_indices(
                        slide_id=slide_id,
                        selection_coords=selection_coords,
                        materialize_bag=materialize_bag,
                        n_selection_tiles=n_selection_tiles,
                        selected=selected.detach().cpu(),
                        alignment_mode=args.alignment_mode,
                    )

                    pruned_features = materialize_bag.tile_features[materialize_selected].detach().cpu()

                    pruned_coords = None
                    if materialize_bag.coords is not None:
                        pruned_coords = materialize_bag.coords[materialize_selected].detach().cpu()

                    pruned_attention = None
                    if materialize_bag.attention is not None:
                        pruned_attention = (
                            materialize_bag.attention[materialize_selected].detach().cpu()
                        )

                    label = materialize_bag.label
                    metadata = dict(materialize_bag.metadata) if materialize_bag.metadata else {}

                if isinstance(label, torch.Tensor):
                    label = label.detach().cpu()

                metadata.update(
                    {
                        "pruned_by": model_type,
                        "forecaster_checkpoint": str(args.forecaster_checkpoint),
                        "keep_ratio": args.keep_ratio,
                        "n_tiles_original": n_selection_tiles,
                        "n_tiles_kept": int(selected.numel()),
                        "selection_feature_store": str(selection_path),
                        "materialize_feature_store": str(materialize_path),
                        "alignment_mode": args.alignment_mode,
                        "pruning_preserved_original_tile_order": True,
                    }
                )
                if target_source is not None:
                    metadata["target_source"] = target_source

                output_store.write(
                    WSIBag(
                        slide_id=slide_id,
                        tile_features=pruned_features,
                        coords=pruned_coords,
                        label=label,
                        attention=pruned_attention,
                        metadata=metadata,
                    )
                )

                n_slides_written += 1
                n_tiles_input_total += n_selection_tiles
                n_tiles_output_total += int(selected.numel())

    return {
        "event": "done",
        "selection_feature_store": str(selection_path),
        "materialize_feature_store": str(materialize_path),
        "output_feature_store": str(args.output_feature_store),
        "forecaster_checkpoint": str(args.forecaster_checkpoint),
        "keep_ratio": args.keep_ratio,
        "alignment_mode": args.alignment_mode,
        "n_slides": n_slides_written,
        "n_tiles_input_total": n_tiles_input_total,
        "n_tiles_output_total": n_tiles_output_total,
        "effective_keep_ratio": (
            n_tiles_output_total / n_tiles_input_total
            if n_tiles_input_total > 0
            else None
        ),
    }


def main() -> int:
    args = parse_args()

    if args.output_feature_store.exists() and args.overwrite:
        args.output_feature_store.unlink()

    args.output_feature_store.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)

    if args.input_feature_store is not None:
        summary = _run_legacy(args, device)
    else:
        summary = _run_selection_materialize(args, device)

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
