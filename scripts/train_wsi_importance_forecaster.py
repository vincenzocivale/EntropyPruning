#!/usr/bin/env python
"""Train a WSI-level tile importance forecaster from HDF5 feature stores.

Unlike ``scripts/train_wsi_attention_forecaster.py`` (kept as a KL-only,
single-store legacy CLI), this script:

- accepts an input feature store and a target feature store separately
  (``--input-feature-store``/``--target-feature-store``), so early-layer
  tile features and precomputed importance targets (ABMIL attention, a WSI
  foundation model tile score, or any other precomputed target) can live in
  independently produced stores. ``--feature-store`` remains as a
  single-store convenience alias.
- supports multiple loss functions via ``--loss``: ``kl``, ``mse``,
  ``topk_bce``, ``kl+rank``.
- writes checkpoint/summary metadata sufficient for downstream
  evaluation/pruning (loss, target provenance, alignment mode, feature
  stores used).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import (
    H5WSIFeatureStore,
    PairedFeatureStoreWSIBagDataset,
    WSIFeatureStore,
    collate_padded_wsi_bags,
)
from src.models.wsi import (
    WSI_TILE_IMPORTANCE_LOSS_TYPES,
    WSITileImportanceForecasterConfig,
    save_wsi_tile_importance_forecaster_checkpoint,
)
from src.training.wsi import (
    evaluate_wsi_tile_importance_forecasting_epoch,
    train_wsi_tile_importance_forecasting_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a WSI tile importance forecaster from paired input/target "
            "HDF5 feature stores (or a single legacy fused store)."
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

    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--input-feature-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--loss",
        type=str,
        choices=WSI_TILE_IMPORTANCE_LOSS_TYPES,
        default="kl",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.1,
        help="Weight of the rank component in the 'kl+rank' loss.",
    )
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=1.0,
        help="Margin used by the rank component in the 'kl+rank' loss.",
    )
    parser.add_argument(
        "--target-smoothing",
        type=float,
        default=0.0,
        help=(
            "Non-negative value added to the target importance before "
            "normalization. 0.0 (default) means an all-zero target on a "
            "slide raises an explicit error; a positive value turns an "
            "all-zero target into a uniform distribution instead."
        ),
    )

    parser.add_argument(
        "--alignment-mode",
        type=str,
        choices=("index", "coords"),
        default="index",
        help=(
            "'index' requires identical slide id, tile count, and (if both "
            "sides have coords) identical tile order. 'coords' aligns tiles "
            "by exact coordinate match and requires coords on both sides."
        ),
    )
    parser.add_argument("--require-coords", action="store_true")
    parser.add_argument(
        "--target-source",
        type=str,
        default=None,
        help=(
            "Explicit provenance label for the importance target (e.g. "
            "'abmil', 'gigapath_wsi_fm'). If omitted, inferred from target "
            "store metadata when possible."
        ),
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio used when explicit split files are not provided.",
    )
    parser.add_argument("--train-slide-ids-file", type=Path, default=None)
    parser.add_argument("--val-slide-ids-file", type=Path, default=None)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing train.txt and val.txt. "
            "Mutually exclusive with --train-slide-ids-file/--val-slide-ids-file."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)

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

    if args.input_feature_dim <= 0:
        raise ValueError("--input-feature-dim must be positive.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive.")
    if args.n_heads <= 0:
        raise ValueError("--n-heads must be positive.")
    if args.n_layers <= 0:
        raise ValueError("--n-layers must be positive.")
    if args.hidden_dim % args.n_heads != 0:
        raise ValueError("--hidden-dim must be divisible by --n-heads.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.rank_weight < 0:
        raise ValueError("--rank-weight must be non-negative.")
    if args.rank_margin <= 0:
        raise ValueError("--rank-margin must be positive.")
    if args.target_smoothing < 0:
        raise ValueError("--target-smoothing must be non-negative.")

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0:
        raise ValueError("--grad-clip-norm must be positive when provided.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    has_train_file = args.train_slide_ids_file is not None
    has_val_file = args.val_slide_ids_file is not None
    has_split_dir = args.split_dir is not None

    if has_split_dir and (has_train_file or has_val_file):
        raise ValueError(
            "--split-dir is mutually exclusive with "
            "--train-slide-ids-file/--val-slide-ids-file."
        )

    if has_split_dir:
        if not args.split_dir.exists():
            raise FileNotFoundError(f"split directory not found: {args.split_dir}")
        if not args.split_dir.is_dir():
            raise NotADirectoryError(f"split path is not a directory: {args.split_dir}")

    if has_train_file != has_val_file:
        raise ValueError(
            "--train-slide-ids-file and --val-slide-ids-file must be provided together."
        )

    if not has_split_dir and not has_train_file and not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1) when split files are not used.")


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


def _make_random_split(
    slide_ids: tuple[str, ...],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(slide_ids) < 2:
        raise ValueError("at least two slides are required for a train/val split.")

    rng = random.Random(seed)
    shuffled = list(slide_ids)
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_ratio))
    n_val = min(n_val, len(shuffled) - 1)

    val_ids = tuple(shuffled[:n_val])
    train_ids = tuple(shuffled[n_val:])

    return train_ids, val_ids


def _resolve_splits(
    all_slide_ids: tuple[str, ...],
    *,
    val_ratio: float,
    seed: int,
    train_slide_ids_file: Path | None,
    val_slide_ids_file: Path | None,
    split_dir: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str | None]]:
    if split_dir is not None:
        train_path = split_dir / "train.txt"
        val_path = split_dir / "val.txt"
        train_ids = _read_slide_ids(train_path)
        val_ids = _read_slide_ids(val_path)
        split_files = {"train": str(train_path), "val": str(val_path)}
    elif train_slide_ids_file is not None and val_slide_ids_file is not None:
        train_ids = _read_slide_ids(train_slide_ids_file)
        val_ids = _read_slide_ids(val_slide_ids_file)
        split_files = {
            "train": str(train_slide_ids_file),
            "val": str(val_slide_ids_file),
        }
    else:
        train_ids, val_ids = _make_random_split(
            all_slide_ids,
            val_ratio=val_ratio,
            seed=seed,
        )
        split_files = {"train": None, "val": None}

    overlap = set(train_ids).intersection(val_ids)
    if overlap:
        raise ValueError(
            "train and validation slide ids overlap: "
            + ", ".join(sorted(overlap)[:10])
        )

    return train_ids, val_ids, split_files


def _make_loader(
    dataset: PairedFeatureStoreWSIBagDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_padded_wsi_bags,
    )


def _infer_target_source(
    target_store: WSIFeatureStore,
    slide_ids: tuple[str, ...],
    *,
    override: str | None,
) -> str | None:
    if override:
        return override

    for slide_id in slide_ids[:5]:
        bag = target_store.read(slide_id)
        if not bag.metadata:
            continue
        for key in ("target_source", "attention_source", "source"):
            if key in bag.metadata:
                return str(bag.metadata[key])

    return None


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
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

    train_ids, val_ids, split_files = _resolve_splits(
        common_slide_ids,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_slide_ids_file=args.train_slide_ids_file,
        val_slide_ids_file=args.val_slide_ids_file,
        split_dir=args.split_dir,
    )

    train_dataset = PairedFeatureStoreWSIBagDataset(
        input_store,
        target_store,
        slide_ids=train_ids,
        alignment_mode=args.alignment_mode,
        require_coords=args.require_coords,
    )
    val_dataset = PairedFeatureStoreWSIBagDataset(
        input_store,
        target_store,
        slide_ids=val_ids,
        alignment_mode=args.alignment_mode,
        require_coords=args.require_coords,
    )

    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    target_source = _infer_target_source(
        target_store, common_slide_ids, override=args.target_source
    )

    config = WSITileImportanceForecasterConfig(
        feature_dim=args.input_feature_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = config.build().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_wsi_tile_importance_forecaster.pt"
    summary_path = args.output_dir / "training_summary.json"

    best_val_loss = float("inf")
    best_epoch = None
    history = []

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "loss": args.loss,
                "n_slides_input": len(input_slide_ids),
                "n_slides_target": len(target_slide_ids),
                "n_slides_common": len(common_slide_ids),
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "input_feature_store": str(input_path),
                "target_feature_store": str(target_path),
                "alignment_mode": args.alignment_mode,
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_output = train_wsi_tile_importance_forecasting_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss=args.loss,
            top_k=args.top_k,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            target_smoothing=args.target_smoothing,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_output = evaluate_wsi_tile_importance_forecasting_epoch(
            model,
            val_loader,
            device=device,
            loss=args.loss,
            top_k=args.top_k,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            target_smoothing=args.target_smoothing,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_output.loss,
            "val_loss": val_output.loss,
            "train_metrics": train_output.metrics,
            "val_metrics": val_output.metrics,
        }
        history.append(record)

        improved = val_output.loss < best_val_loss
        if improved:
            best_val_loss = val_output.loss
            best_epoch = epoch

            checkpoint_metrics = {
                "train_loss": train_output.loss,
                "val_loss": val_output.loss,
                **{f"train_{key}": value for key, value in train_output.metrics.items()},
                **{f"val_{key}": value for key, value in val_output.metrics.items()},
            }
            save_wsi_tile_importance_forecaster_checkpoint(
                best_path,
                model=model,
                config=config,
                loss=args.loss,
                target_type="tile_importance",
                target_source=target_source,
                input_feature_store=str(input_path),
                target_feature_store=str(target_path),
                alignment_mode=args.alignment_mode,
                epoch=epoch,
                metrics=checkpoint_metrics,
                metadata={
                    "train_slide_ids": list(train_ids),
                    "val_slide_ids": list(val_ids),
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "split_files": split_files,
                },
            )

        print(
            json.dumps(
                {
                    "event": "epoch",
                    **record,
                    "best_epoch": best_epoch,
                    "is_best": improved,
                }
            ),
            flush=True,
        )

    best_val_metrics = next(
        (record["val_metrics"] for record in history if record["epoch"] == best_epoch),
        {},
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "best_checkpoint": str(best_path),
        "target_entropy_mean": best_val_metrics.get("target_entropy"),
        "target_source": target_source,
        "seed": args.seed,
        "history": history,
        "config": {
            "input_feature_dim": args.input_feature_dim,
            "hidden_dim": args.hidden_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "loss": args.loss,
            "top_k": args.top_k,
            "rank_weight": args.rank_weight,
            "rank_margin": args.rank_margin,
            "target_smoothing": args.target_smoothing,
            "alignment_mode": args.alignment_mode,
        },
        "input_feature_store": str(input_path),
        "target_feature_store": str(target_path),
        "split": {
            "train_slide_ids": list(train_ids),
            "val_slide_ids": list(val_ids),
            "split_files": split_files,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        json.dumps(
            {
                "event": "done",
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_checkpoint": str(best_path),
                "summary": str(summary_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
