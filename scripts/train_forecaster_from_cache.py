#!/usr/bin/env python
"""Train AttentionForecaster from a pre-extracted (source, target) HDF5 cache.

The backbone forward pass is already paid once during caching (see
scripts/build_wsi_tile_eaf_cache.py / scripts/build_thunder_online_forecaster_cache.py),
so every epoch here is a full, cheap pass over the *entire* cached training
pool -- no --max-steps-per-epoch cap, matching the recipe behind the historical
UNI run (uni_multi_phase2_src02_tgt23, rho=0.83: full epochs, batch 128, 50
epochs, hidden=1024) instead of the online trainer's step-capped epochs.

Usage:
    python scripts/train_forecaster_from_cache.py \\
        --cache-dir /path/to/titan_src02_tgtlast_wsi \\
        --output-dir results/wsi_tile_eaf_cached \\
        --experiment-name titan_wsi_cached \\
        --epochs 50 --hidden 1024 \\
        --wandb-project eaf-thunder-online
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cached_forecast_dataset import CachedForecastDataset
from src.models import AttentionForecaster
from src.utils import get_device, save_results, set_seed

TOPK_FRACTIONS = (0.10, 0.25, 0.50)


def _spearman(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rank = predicted.argsort(-1).argsort(-1).float()
    target_rank = target.argsort(-1).argsort(-1).float()
    pred_rank -= pred_rank.mean(-1, keepdim=True)
    target_rank -= target_rank.mean(-1, keepdim=True)
    denominator = torch.sqrt(
        pred_rank.square().sum(-1) * target_rank.square().sum(-1)
    ).clamp_min(1e-8)
    return (pred_rank * target_rank).sum(-1) / denominator


def _topk_recall(predicted: torch.Tensor, target: torch.Tensor, fraction: float) -> torch.Tensor:
    k = max(1, int(round(predicted.shape[-1] * fraction)))
    pred_idx = predicted.topk(k, dim=-1).indices
    target_idx = target.topk(k, dim=-1).indices
    matches = (pred_idx.unsqueeze(-1) == target_idx.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean(dim=-1)


def _find_split_file(cache_dir: Path, split: str) -> Path | None:
    """Resolve a split to a cache file, supporting either on-disk layout."""
    per_split = cache_dir / f"{split}.h5"
    if per_split.exists():
        return per_split
    # legacy layout: one file per dataset, groups per split -- caller passes
    # the dataset .h5 directly via --cache-dir in that case.
    if cache_dir.suffix == ".h5" and cache_dir.exists():
        return cache_dir
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True, help="Directory from build_wsi_tile_eaf_cache.py (contains train.h5/val.h5/cache_meta.json), or a single dataset .h5 with per-split groups.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="forecaster_from_cache")

    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="0 disables early stopping (run all --epochs).")

    parser.add_argument("--wandb-project", default=None)
    return parser


def _evaluate(forecaster, loader, device, amp) -> dict[str, float]:
    forecaster.eval()
    total = 0
    sums = {"kl": 0.0, "rho": 0.0, **{f"recall_{f:.2f}": 0.0 for f in TOPK_FRACTIONS}}
    with torch.inference_mode():
        for emb, target in tqdm(loader, desc="val", leave=False):
            emb, target = emb.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and amp)):
                logits = forecaster(emb)
            logits_f, target_f = logits.float(), target.float()
            kl = F.kl_div(logits_f.log_softmax(-1), target_f, reduction="none").sum(-1)
            rho = _spearman(logits_f, target_f)
            batch = emb.shape[0]
            total += batch
            sums["kl"] += float(kl.sum())
            sums["rho"] += float(rho.sum())
            for fraction in TOPK_FRACTIONS:
                sums[f"recall_{fraction:.2f}"] += float(_topk_recall(logits_f, target_f, fraction).sum())
    return {key: value / max(total, 1) for key, value in sums.items()} | {"n": total}


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device()

    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "cache_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    source_layer = meta.get("source_layer", 2)
    target_layer = meta.get("target_layer")
    embed_dim = meta.get("embed_dim", 1024)
    if target_layer is None:
        raise ValueError(f"Could not resolve target_layer from {meta_path}; pass a cache built with build_wsi_tile_eaf_cache.py")

    train_path = _find_split_file(cache_dir, "train")
    val_path = _find_split_file(cache_dir, "val")
    if train_path is None or val_path is None:
        raise FileNotFoundError(f"Expected train.h5/val.h5 (or a dataset .h5 with /train,/val groups) under {cache_dir}")
    train_split_group = "train" if train_path == val_path else None
    val_split_group = "val" if train_path == val_path else None

    train_ds = CachedForecastDataset(train_path, source_layer, target_layer, split=train_split_group)
    val_ds = CachedForecastDataset(val_path, source_layer, target_layer, split=val_split_group)
    print(f"[train] cache: {cache_dir}")
    print(f"[train] source_layer={source_layer} target_layer={target_layer} embed_dim={embed_dim}")
    print(f"[train] train tiles={len(train_ds):,}  val tiles={len(val_ds):,}")

    kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)

    forecaster = AttentionForecaster(
        embed_dim=embed_dim, hidden=args.hidden, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(forecaster.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_forecaster.pt"

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.experiment_name,
            config={**vars(args), "source_layer": source_layer, "target_layer": target_layer,
                    "embed_dim": embed_dim, "n_train": len(train_ds), "n_val": len(val_ds)},
            tags=["eaf", "cached", "full_epoch"],
        )

    best_val_kl = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        forecaster.train()
        loss_sum, n_seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d} train", leave=False)
        for emb, target in progress:
            emb, target = emb.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp)):
                logits = forecaster(emb)
                loss = F.kl_div(logits.log_softmax(-1), target, reduction="batchmean")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch = emb.shape[0]
            loss_sum += float(loss.detach()) * batch
            n_seen += batch
            progress.set_postfix(kl=f"{loss_sum / max(n_seen, 1):.4f}")
        train_kl = loss_sum / max(n_seen, 1)

        val_metrics = _evaluate(forecaster, val_loader, device, args.amp)
        scheduler.step()

        row = {"epoch": epoch, "train_kl": train_kl, **{f"val_{k}": v for k, v in val_metrics.items()}, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(row)
        improved = val_metrics["kl"] < best_val_kl
        if improved:
            best_val_kl = val_metrics["kl"]
            epochs_without_improvement = 0
            torch.save({
                "forecaster_state_dict": forecaster.state_dict(),
                "forecaster": {"embed_dim": embed_dim, "hidden": args.hidden, "n_heads": args.n_heads, "n_layers": args.n_layers, "dropout": args.dropout},
                "cache": {"source_layer": source_layer, "target_layer": target_layer, "cache_dir": str(cache_dir)},
                "training": {"epoch": epoch, "best_val_kl": best_val_kl},
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:03d} train_kl={train_kl:.6f} val_kl={val_metrics['kl']:.6f} "
            f"val_rho={val_metrics['rho']:.4f} val_recall_0.10={val_metrics['recall_0.10']:.4f} "
            f"best_val_kl={best_val_kl:.6f}"
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    results = {
        "experiment_name": args.experiment_name,
        "cache_dir": str(cache_dir),
        "source_layer": source_layer,
        "target_layer": target_layer,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "checkpoint": str(checkpoint_path),
        "best_val_kl": best_val_kl,
        "history": history,
    }
    results_path = save_results(output_dir / "results.json", results)
    print(json.dumps({"results": str(results_path), "best_val_kl": best_val_kl, "checkpoint": str(checkpoint_path)}, indent=2))

    if wandb_run is not None:
        wandb_run.summary["best_val_kl"] = best_val_kl
        wandb_run.finish()


if __name__ == "__main__":
    main()
