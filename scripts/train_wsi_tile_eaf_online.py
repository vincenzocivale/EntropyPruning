#!/usr/bin/env python
"""Train tile-level EAF directly from WSI pixels without feature caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name
from src.data.wsi_tile_stream import build_online_tile_loaders, load_wsi_manifest
from src.wsi_pipeline.cache_index import read_tile_cache_index
from src.wsi_pipeline.cache_io import validate_cache
from src.wsi_pipeline.compact_cache_dataset import build_compact_cache_tile_loaders
from src.models import AttentionForecaster, ThunderBackboneAdapter
from src.models.online_tile_eaf import OnlineAttentionTeacher, load_checkpoint_flexibly
from src.utils import set_seed


def _autocast(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _spearman(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rank = prediction.argsort(-1).argsort(-1).float()
    target_rank = target.argsort(-1).argsort(-1).float()
    pred_rank -= pred_rank.mean(-1, keepdim=True)
    target_rank -= target_rank.mean(-1, keepdim=True)
    denominator = torch.sqrt(
        pred_rank.square().sum(-1) * target_rank.square().sum(-1)
    ).clamp_min(1e-8)
    return (pred_rank * target_rank).sum(-1) / denominator


def _topk_recall(prediction: torch.Tensor, target: torch.Tensor, ratio: float) -> torch.Tensor:
    count = max(1, int(round(prediction.shape[-1] * ratio)))
    pred_indices = prediction.topk(count, dim=-1).indices
    target_indices = target.topk(count, dim=-1).indices
    pred_mask = torch.zeros_like(prediction, dtype=torch.bool)
    pred_mask.scatter_(1, pred_indices, True)
    return pred_mask.gather(1, target_indices).float().mean(dim=-1)


def _rank_alignment_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_score = target.clamp_min(1e-8).log()
    logits_centered = logits - logits.mean(dim=-1, keepdim=True)
    target_centered = target_score - target_score.mean(dim=-1, keepdim=True)
    return 1.0 - F.cosine_similarity(logits_centered, target_centered, dim=-1).mean()


def _run_epoch(
    *,
    forecaster: AttentionForecaster,
    teacher: OnlineAttentionTeacher,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    amp_dtype: str,
    rank_loss_weight: float,
    keep_ratio_metric: float,
    grad_accum: int,
    train: bool,
    log_every: int,
    global_step: int,
    use_wandb: bool,
    cached_targets: bool,
) -> tuple[dict[str, float], int]:
    forecaster.train(train)
    total = 0
    sums = {"loss": 0.0, "kl": 0.0, "rank": 0.0, "rho": 0.0, "topk_recall": 0.0}
    start = time.perf_counter()
    if train:
        optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, leave=False, desc="train" if train else "val")
    for batch_index, (images, batch_target) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), _autocast(device, amp_dtype):
            if cached_targets:
                source_tokens = teacher.extract_early(images)
                target_attention = batch_target.to(device, non_blocking=True)
            else:
                source_tokens, target_attention = teacher.extract(images)
        with torch.set_grad_enabled(train), _autocast(device, amp_dtype):
            logits = forecaster(source_tokens)
            kl = F.kl_div(
                logits.log_softmax(dim=-1),
                target_attention,
                reduction="batchmean",
            )
            rank_loss = _rank_alignment_loss(logits, target_attention)
            loss = kl + rank_loss_weight * rank_loss

        if train:
            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if use_wandb and global_step % log_every == 0:
                    elapsed = max(time.perf_counter() - start, 1e-6)
                    wandb.log(
                        {
                            "step/loss": float(loss.detach()),
                            "step/kl": float(kl.detach()),
                            "step/rank": float(rank_loss.detach()),
                            "step/grad_norm": float(grad_norm),
                            "step/tiles_per_second": total / elapsed,
                            "trainer/global_step": global_step,
                        },
                        step=global_step,
                    )

        batch_size = images.shape[0]
        with torch.no_grad():
            rho = _spearman(logits.float(), target_attention.float()).mean()
            recall = _topk_recall(
                logits.float(), target_attention.float(), keep_ratio_metric
            ).mean()
        total += batch_size
        sums["loss"] += float(loss.detach()) * batch_size
        sums["kl"] += float(kl.detach()) * batch_size
        sums["rank"] += float(rank_loss.detach()) * batch_size
        sums["rho"] += float(rho) * batch_size
        sums["topk_recall"] += float(recall) * batch_size

    elapsed = max(time.perf_counter() - start, 1e-6)
    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    metrics["tiles"] = float(total)
    metrics["tiles_per_second"] = total / elapsed
    return metrics, global_step


def _checkpoint_payload(model: AttentionForecaster, args: argparse.Namespace, adapter) -> dict[str, Any]:
    return {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": vars(args),
        "encoder": {
            "embed_dim": adapter.embed_dim,
            "n_blocks": adapter.n_blocks,
            "n_patches": adapter.n_patches,
            "num_prefix_tokens": adapter.num_prefix_tokens,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task-agnostic tile EAF training from WSI tiles sampled on the fly"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument(
        "--target-cache-index",
        default=None,
        help="Validated index from `eaf.py cache index-tile`; enables compact-cache training",
    )
    parser.add_argument("--source-layer", type=int, default=2)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--split-column", default="split")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--slide-group", nargs="+", default=["diagnostic"])
    parser.add_argument("--exclude-cohort", nargs="*", default=[])
    parser.add_argument("--default-patch-size", type=int, default=512)
    parser.add_argument(
        "--tile-size-at-target-mag", type=int, default=None,
        help=(
            "Optional encoder-specific physical crop size at the TRIDENT target "
            "magnification; smaller crops are jittered inside each coordinate window"
        ),
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--slides-per-batch", type=int, default=4)
    parser.add_argument(
        "--train-wsis-per-epoch", type=int, default=0,
        help="Exact WSI count per epoch; 0 resolves from --train-wsi-fraction",
    )
    parser.add_argument("--train-wsi-fraction", type=float, default=0.5)
    parser.add_argument("--tiles-per-wsi", type=int, default=24)
    parser.add_argument("--val-wsis", type=int, default=128)
    parser.add_argument("--val-tiles-per-wsi", type=int, default=16)
    parser.add_argument("--cohort-balance-power", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--slide-cache-size", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--rank-loss-weight", type=float, default=0.1)
    parser.add_argument("--keep-ratio-metric", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Disable cuDNN autotuning for stricter reproducibility",
    )

    parser.add_argument(
        "--output-dir", default=None,
        help="Defaults to checkpoints/tile_eaf/<model-name> (tile-encoder-dependent)",
    )
    parser.add_argument("--wandb-project", default="eaf-tile-online")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if (
        args.target_cache_index
        and args.tile_size_at_target_mag is not None
        and args.tile_size_at_target_mag != args.default_patch_size
    ):
        raise ValueError(
            "Compact-cache training must reread the exact cache-time field of view; "
            "omit --tile-size-at-target-mag or set it equal to --default-patch-size"
        )

    set_seed(args.seed)
    if not args.deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Online WSI EAF training requires a CUDA device")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        args.amp_dtype = "fp16"

    backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    backbone = backbone.to(device).eval()
    adapter = ThunderBackboneAdapter(backbone)
    target_layer = args.target_layer if args.target_layer is not None else adapter.n_blocks - 1
    if args.teacher_checkpoint:
        missing, unexpected = load_checkpoint_flexibly(backbone, args.teacher_checkpoint)
        print(f"Teacher checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    teacher = OnlineAttentionTeacher(backbone, adapter, args.source_layer, target_layer)

    split_records = load_wsi_manifest(
        args.manifest,
        args.data_root,
        split_column=args.split_column,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        include_slide_groups=args.slide_group,
        exclude_cohorts=args.exclude_cohort,
        default_patch_size=args.default_patch_size,
        crop_size_at_target_mag=args.tile_size_at_target_mag,
    )
    if not 0.0 < args.train_wsi_fraction <= 1.0:
        raise ValueError("--train-wsi-fraction must be in (0, 1]")
    available_train_wsis = len(split_records["train"])
    resolved_train_wsis = args.train_wsis_per_epoch
    if resolved_train_wsis <= 0:
        resolved_train_wsis = min(
            available_train_wsis,
            max(512, math.ceil(available_train_wsis * args.train_wsi_fraction)),
        )
    resolved_train_wsis = min(resolved_train_wsis, available_train_wsis)
    if available_train_wsis < args.slides_per_batch:
        raise ValueError(
            "Training split has fewer WSI than --slides-per-batch: "
            f"{available_train_wsis} < {args.slides_per_batch}"
        )
    resolved_train_wsis -= resolved_train_wsis % args.slides_per_batch
    resolved_train_wsis = max(args.slides_per_batch, resolved_train_wsis)
    loader_builder = build_online_tile_loaders
    loader_args: tuple[Any, ...] = (split_records, transform)
    if args.target_cache_index:
        cache_paths = read_tile_cache_index(args.target_cache_index)
        first_cache = next(iter(cache_paths.values()))
        cache_info = validate_cache(first_cache, expected_kind="tile_eaf")
        cache_spec = cache_info["spec"]
        cache_encoder = str(cache_spec.get("tile_encoder", ""))
        compatible_names = {cache_encoder}
        if cache_encoder == "conch_v15":
            compatible_names.add("titan")
        if args.model_name not in compatible_names:
            raise ValueError(
                f"Cache encoder={cache_encoder!r} is incompatible with "
                f"--model-name={args.model_name!r}"
            )
        if int(cache_spec.get("early_layer", args.source_layer)) != args.source_layer:
            raise ValueError(
                f"Cache documents early_layer={cache_spec.get('early_layer')}, "
                f"but training requested --source-layer={args.source_layer}"
            )
        attention_shape = cache_info["datasets"]["final_attention"]
        if len(attention_shape) != 2 or int(attention_shape[1]) != adapter.n_patches:
            raise ValueError(
                f"Cache attention geometry {attention_shape} does not match "
                f"encoder n_patches={adapter.n_patches}"
            )
        loader_builder = build_compact_cache_tile_loaders
        loader_args = (split_records, cache_paths, transform)
    train_loader, val_loader, train_sampler, val_sampler = loader_builder(
        *loader_args,
        batch_size=args.batch_size,
        slides_per_batch=args.slides_per_batch,
        train_slides_per_epoch=resolved_train_wsis,
        train_tiles_per_slide=args.tiles_per_wsi,
        val_slides_per_epoch=args.val_wsis,
        val_tiles_per_slide=args.val_tiles_per_wsi,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        slide_cache_size=args.slide_cache_size,
        cohort_balance_power=args.cohort_balance_power,
        seed=args.seed,
    )
    print(
        f"WSI split: train={len(split_records['train'])} val={len(split_records['val'])}; "
        f"epoch={len(train_sampler)} batches/{resolved_train_wsis} WSI/"
        f"~{resolved_train_wsis * args.tiles_per_wsi} tiles"
    )

    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        forecaster.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp_dtype == "fp16"
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(f"checkpoints/tile_eaf/{args.model_name}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (
        f"{args.model_name}_src{args.source_layer:02d}_tgt{target_layer:02d}_online"
    )
    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=run_name,
            job_type="tile_eaf_online",
            config={
                **vars(args),
                "target_layer_resolved": target_layer,
                "train_wsi_count": len(split_records["train"]),
                "val_wsi_count": len(split_records["val"]),
                "embed_dim": adapter.embed_dim,
                "n_patches": adapter.n_patches,
                "resolved_train_wsis_per_epoch": resolved_train_wsis,
                "resolved_train_tiles_per_epoch": resolved_train_wsis * args.tiles_per_wsi,
            },
            tags=[
                args.model_name,
                "tile_eaf",
                "compact_cache" if args.target_cache_index else "online",
                "task_agnostic",
            ],
        )

    best_val = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    checkpoint_path = output_dir / f"best_{run_name}.pt"

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(0)
        train_metrics, global_step = _run_epoch(
            forecaster=forecaster,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=args.amp_dtype,
            rank_loss_weight=args.rank_loss_weight,
            keep_ratio_metric=args.keep_ratio_metric,
            grad_accum=args.grad_accum,
            train=True,
            log_every=args.log_every,
            global_step=global_step,
            use_wandb=use_wandb,
            cached_targets=bool(args.target_cache_index),
        )
        val_metrics, global_step = _run_epoch(
            forecaster=forecaster,
            teacher=teacher,
            loader=val_loader,
            optimizer=None,
            scaler=scaler,
            device=device,
            amp_dtype=args.amp_dtype,
            rank_loss_weight=args.rank_loss_weight,
            keep_ratio_metric=args.keep_ratio_metric,
            grad_accum=1,
            train=False,
            log_every=args.log_every,
            global_step=global_step,
            use_wandb=use_wandb,
            cached_targets=bool(args.target_cache_index),
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": scheduler.get_last_lr()[0],
            "sampling": train_sampler.last_summary,
        }
        history.append(row)
        improved = val_metrics["kl"] < best_val - args.early_stopping_min_delta
        if improved:
            best_val = val_metrics["kl"]
            epochs_without_improvement = 0
            torch.save(_checkpoint_payload(forecaster, args, adapter), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if use_wandb:
            log = {
                "epoch": epoch + 1,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                "optimizer/lr": scheduler.get_last_lr()[0],
                "sampling/unique_wsi": train_sampler.last_summary.get("unique_slides", 0),
                "sampling/scheduled_tiles": train_sampler.last_summary.get("scheduled_tiles", 0),
                "early_stopping/best_val_kl": best_val,
                "early_stopping/epochs_without_improvement": epochs_without_improvement,
            }
            for cohort, count in train_sampler.last_summary.get(
                "cohort_counts", {}
            ).items():
                log[f"sampling/cohort/{cohort}"] = count
            if torch.cuda.is_available():
                log["system/max_cuda_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
            wandb.log(log, step=global_step)
        print(
            f"Epoch {epoch + 1:02d} | train_kl={train_metrics['kl']:.5f} "
            f"val_kl={val_metrics['kl']:.5f} val_rho={val_metrics['rho']:.4f} "
            f"top{args.keep_ratio_metric:.0%}_recall={val_metrics['topk_recall']:.4f} "
            f"best={best_val:.5f}"
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    summary = {
        "run_name": run_name,
        "best_val_kl": best_val,
        "checkpoint": str(checkpoint_path),
        "epochs_completed": len(history),
        "history": history,
        "storage_policy": (
            "best checkpoint + JSON; cached final attention + online early exit"
            if args.target_cache_index
            else "best checkpoint + JSON only; no tile/embedding/attention cache"
        ),
    }
    (output_dir / f"summary_{run_name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if use_wandb:
        wandb.summary["best_val_kl"] = best_val
        wandb.summary["checkpoint"] = str(checkpoint_path)
        wandb.finish()


if __name__ == "__main__":
    main()
