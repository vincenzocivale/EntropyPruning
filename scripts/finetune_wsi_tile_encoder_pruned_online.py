#!/usr/bin/env python
"""Task-agnostic pruning-aware adaptation of a tile encoder from WSI pixels."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name
from src.data.wsi_tile_stream import build_online_tile_loaders, load_wsi_manifest
from src.models import AttentionForecaster, ThunderBackboneAdapter
from src.models.online_tile_eaf import (
    PrunedLoRAEncoder,
    embedding_distillation_loss,
    load_checkpoint_flexibly,
    unwrap_checkpoint_state,
)
from src.utils import set_seed


def _autocast(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _run_epoch(
    *,
    student: PrunedLoRAEncoder,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    amp_dtype: str,
    cosine_weight: float,
    mse_weight: float,
    pairwise_weight: float,
    grad_accum: int,
    train: bool,
    global_step: int,
    log_every: int,
    use_wandb: bool,
) -> tuple[dict[str, float], int]:
    student.train(train)
    sums = {"loss": 0.0, "cosine": 0.0, "mse": 0.0, "pairwise": 0.0}
    total = 0
    start = time.perf_counter()
    if train:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, (images, _) in enumerate(
        tqdm(loader, leave=False, desc="train" if train else "val")
    ):
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), _autocast(device, amp_dtype):
            teacher_embedding = student.full_teacher_embedding(images)
        with torch.set_grad_enabled(train), _autocast(device, amp_dtype):
            student_embedding = student(images)
            loss, components = embedding_distillation_loss(
                student_embedding,
                teacher_embedding,
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
                pairwise_weight=pairwise_weight,
            )

        if train:
            scaler.scale(loss / grad_accum).backward()
            should_step = (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if use_wandb and global_step % log_every == 0:
                    elapsed = max(time.perf_counter() - start, 1e-6)
                    wandb.log(
                        {
                            "step/loss": float(loss.detach()),
                            "step/cosine": float(components["cosine"].detach()),
                            "step/grad_norm": float(grad_norm),
                            "step/tiles_per_second": total / elapsed,
                            "trainer/global_step": global_step,
                        },
                        step=global_step,
                    )

        batch_size = images.shape[0]
        total += batch_size
        sums["loss"] += float(loss.detach()) * batch_size
        for key in ("cosine", "mse", "pairwise"):
            sums[key] += float(components[key].detach()) * batch_size

    elapsed = max(time.perf_counter() - start, 1e-6)
    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    metrics["tiles"] = float(total)
    metrics["tiles_per_second"] = total / elapsed
    return metrics, global_step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task-agnostic online distillation of a forecaster-pruned tile encoder"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--forecaster-ckpt", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

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

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--slides-per-batch", type=int, default=4)
    parser.add_argument(
        "--train-wsis-per-epoch", type=int, default=0,
        help="Exact WSI count per epoch; 0 resolves from --train-wsi-fraction",
    )
    parser.add_argument("--train-wsi-fraction", type=float, default=0.5)
    parser.add_argument("--tiles-per-wsi", type=int, default=16)
    parser.add_argument("--val-wsis", type=int, default=128)
    parser.add_argument("--val-tiles-per-wsi", type=int, default=8)
    parser.add_argument("--cohort-balance-power", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--slide-cache-size", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Disable cuDNN autotuning for stricter reproducibility",
    )

    parser.add_argument("--output-dir", default="checkpoints/wsi_tile_pruned_online")
    parser.add_argument("--wandb-project", default="eaf-pruned-tile-online")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)
    if not args.deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Pruned tile-encoder distillation requires a CUDA device")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        args.amp_dtype = "fp16"

    student_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    if args.teacher_checkpoint:
        missing, unexpected = load_checkpoint_flexibly(
            student_backbone, args.teacher_checkpoint
        )
        print(
            "Base checkpoint loaded: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    student_backbone = student_backbone.to(device)
    adapter = ThunderBackboneAdapter(student_backbone)
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    forecaster_payload = torch.load(args.forecaster_ckpt, map_location="cpu")
    forecaster.load_state_dict(unwrap_checkpoint_state(forecaster_payload), strict=True)
    student = PrunedLoRAEncoder(
        student_backbone,
        adapter,
        forecaster,
        prune_layer=args.prune_layer,
        keep_ratio=args.keep_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    if args.gradient_checkpointing and hasattr(student.raw_backbone, "set_grad_checkpointing"):
        student.raw_backbone.set_grad_checkpointing(True)

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in student.parameters())
    print(f"Trainable parameters: {trainable_count:,}/{total_count:,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")

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
    train_loader, val_loader, train_sampler, val_sampler = build_online_tile_loaders(
        split_records,
        transform,
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

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (
        f"{args.model_name}_prune{args.prune_layer}_keep{int(args.keep_ratio * 100)}_online"
    )
    checkpoint_path = output_dir / f"best_{run_name}_adapter.pt"
    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=run_name,
            job_type="pruned_tile_distillation_online",
            config={
                **vars(args),
                "train_wsi_count": len(split_records["train"]),
                "val_wsi_count": len(split_records["val"]),
                "trainable_parameters": trainable_count,
                "total_parameters": total_count,
                "resolved_train_wsis_per_epoch": resolved_train_wsis,
                "resolved_train_tiles_per_epoch": resolved_train_wsis * args.tiles_per_wsi,
                "single_backbone_teacher_student": True,
            },
            tags=[args.model_name, "pruned_tile_encoder", "online", "task_agnostic"],
        )

    best_val = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(0)
        train_metrics, global_step = _run_epoch(
            student=student,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=args.amp_dtype,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            pairwise_weight=args.pairwise_weight,
            grad_accum=args.grad_accum,
            train=True,
            global_step=global_step,
            log_every=args.log_every,
            use_wandb=use_wandb,
        )
        val_metrics, global_step = _run_epoch(
            student=student,
            loader=val_loader,
            optimizer=None,
            scaler=scaler,
            device=device,
            amp_dtype=args.amp_dtype,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            pairwise_weight=args.pairwise_weight,
            grad_accum=1,
            train=False,
            global_step=global_step,
            log_every=args.log_every,
            use_wandb=use_wandb,
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
        improved = val_metrics["loss"] < best_val - args.early_stopping_min_delta
        if improved:
            best_val = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "trainable_state_dict": student.trainable_state_dict(),
                    "config": vars(args),
                    "base_model": args.model_name,
                    "forecaster_checkpoint": str(Path(args.forecaster_ckpt).resolve()),
                    "trainable_parameters": trainable_count,
                    "storage_policy": "LoRA tensors only; single shared backbone; no full checkpoint or tile cache",
                },
                checkpoint_path,
            )
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
                "early_stopping/best_val_loss": best_val,
                "early_stopping/epochs_without_improvement": epochs_without_improvement,
                "system/max_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            for cohort, count in train_sampler.last_summary.get(
                "cohort_counts", {}
            ).items():
                log[f"sampling/cohort/{cohort}"] = count
            wandb.log(log, step=global_step)
        print(
            f"Epoch {epoch + 1:02d} | train={train_metrics['loss']:.5f} "
            f"val={val_metrics['loss']:.5f} val_cos={val_metrics['cosine']:.5f} "
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
        "best_val_loss": best_val,
        "checkpoint": str(checkpoint_path),
        "epochs_completed": len(history),
        "history": history,
        "trainable_parameters": trainable_count,
        "storage_policy": "best LoRA adapter + JSON only; single shared backbone; no tile cache",
    }
    (output_dir / f"summary_{run_name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if use_wandb:
        wandb.summary["best_val_loss"] = best_val
        wandb.summary["checkpoint"] = str(checkpoint_path)
        wandb.summary["trainable_parameters"] = trainable_count
        wandb.finish()


if __name__ == "__main__":
    main()
