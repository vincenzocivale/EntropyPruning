#!/usr/bin/env python
"""Train a WSI-level tile attention forecaster from an HDF5 feature store.

The input HDF5 file must be readable by ``H5WSIFeatureStore`` and each training
bag must contain ``attention`` targets. This script does not extract features
and does not train a MIL classifier; it only trains the forecasting module on
precomputed WSI bags.
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
    FeatureStoreWSIBagDataset,
    H5WSIFeatureStore,
    collate_padded_wsi_bags,
)
from src.models.wsi import (
    WSITileAttentionForecasterConfig,
    save_wsi_tile_attention_forecaster_checkpoint,
)
from src.training.wsi import (
    evaluate_wsi_attention_forecasting_epoch,
    train_wsi_attention_forecasting_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a WSI tile attention forecaster from an HDF5 feature store."
    )

    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio used when explicit split files are not provided.",
    )
    parser.add_argument(
        "--train-slide-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited train slide ids.",
    )
    parser.add_argument(
        "--val-slide-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited validation slide ids.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.feature_dim <= 0:
        raise ValueError("--feature-dim must be positive.")
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
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    has_train_file = args.train_slide_ids_file is not None
    has_val_file = args.val_slide_ids_file is not None
    if has_train_file != has_val_file:
        raise ValueError(
            "--train-slide-ids-file and --val-slide-ids-file must be provided together."
        )

    if not has_train_file and not 0.0 < args.val_ratio < 1.0:
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
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if train_slide_ids_file is not None and val_slide_ids_file is not None:
        train_ids = _read_slide_ids(train_slide_ids_file)
        val_ids = _read_slide_ids(val_slide_ids_file)
    else:
        train_ids, val_ids = _make_random_split(
            all_slide_ids,
            val_ratio=val_ratio,
            seed=seed,
        )

    overlap = set(train_ids).intersection(val_ids)
    if overlap:
        raise ValueError(
            "train and validation slide ids overlap: "
            + ", ".join(sorted(overlap)[:10])
        )

    return train_ids, val_ids


def _make_loader(
    dataset: FeatureStoreWSIBagDataset,
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


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    store = H5WSIFeatureStore(args.feature_store)
    all_slide_ids = store.slide_ids()
    if not all_slide_ids:
        raise ValueError(f"feature store contains no slides: {args.feature_store}")

    train_ids, val_ids = _resolve_splits(
        all_slide_ids,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_slide_ids_file=args.train_slide_ids_file,
        val_slide_ids_file=args.val_slide_ids_file,
    )

    train_dataset = FeatureStoreWSIBagDataset(store, slide_ids=train_ids)
    val_dataset = FeatureStoreWSIBagDataset(store, slide_ids=val_ids)

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

    config = WSITileAttentionForecasterConfig(
        feature_dim=args.feature_dim,
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
    best_path = args.output_dir / "best_wsi_tile_attention_forecaster.pt"
    summary_path = args.output_dir / "training_summary.json"

    best_val_loss = float("inf")
    best_epoch = None
    history = []

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "n_slides": len(all_slide_ids),
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "feature_store": str(args.feature_store),
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_output = train_wsi_attention_forecasting_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            top_k=args.top_k,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_output = evaluate_wsi_attention_forecasting_epoch(
            model,
            val_loader,
            device=device,
            top_k=args.top_k,
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
            save_wsi_tile_attention_forecaster_checkpoint(
                best_path,
                model=model,
                config=config,
                epoch=epoch,
                metrics=checkpoint_metrics,
                metadata={
                    "feature_store": str(args.feature_store),
                    "train_slide_ids": list(train_ids),
                    "val_slide_ids": list(val_ids),
                    "top_k": args.top_k,
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

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_path),
        "history": history,
        "config": {
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        },
        "split": {
            "train_slide_ids": list(train_ids),
            "val_slide_ids": list(val_ids),
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
