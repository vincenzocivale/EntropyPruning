#!/usr/bin/env python
"""LoRA-adapt TITAN's own vision-encoder blocks to recover accuracy lost from
forecaster-guided tile pruning, distilling against the *cached* frozen-TITAN
slide embedding (no unpruned TITAN forward pass at training time -- the teacher
signal was already computed once, offline, by `scripts/wsi_eaf_infer_wsi_fm.py`
and lives in `slide_embedding` inside the wsi_eaf output files).

Mirrors `scripts/finetune_wsi_tile_encoder_pruned_online.py` (task-agnostic
full-vs-pruned embedding distillation for the *tile encoder*), one level up:
here it is TITAN's own tile bag being pruned mid-forward
(`src/models/wsi/pruned_titan.py::PrunedLoRATitanEncoder`), using a frozen
WSI-EAF forecaster trained at the same `--prune-layer`
(`scripts/train_wsi_landmark_forecaster.py --input-source titan_hidden
--titan-hidden-layer <prune-layer>`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.wsi.wsi_forecaster_dataset import (
    WSIForecasterManifestConfig,
    assign_splits,
    build_manifest,
    write_manifest_csv,
)
from src.data.wsi.wsi_pruned_titan_dataset import WSIPrunedTitanDataset
from src.models.online_tile_eaf import embedding_distillation_loss
from src.models.wsi.dense_forecaster import WSIDenseForecaster, WSIDenseForecasterALiBi
from src.models.wsi.pruned_titan import PrunedLoRATitanEncoder
from src.utils import set_seed
from src.wsi_pipeline.wsi_models.titan import TitanAdapter


def _autocast(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _load_forecaster(checkpoint_path: Path, *, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fargs = payload["args"]
    common = dict(embed_dim=768, hidden=fargs["hidden"], n_heads=fargs["n_heads"], n_layers=fargs["n_layers"], dropout=0.0)
    if fargs.get("architecture") == "dense_alibi":
        forecaster = WSIDenseForecasterALiBi(**common)
    else:
        forecaster = WSIDenseForecaster(**common)
    forecaster.load_state_dict(payload["model"])
    forecaster = forecaster.to(device).eval()
    source_layer = fargs.get("titan_hidden_layer")
    if source_layer is None:
        raise ValueError(f"{checkpoint_path} was not trained with --input-source titan_hidden")
    return forecaster, int(source_layer)


def _run_epoch(
    *,
    student: PrunedLoRATitanEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_dtype: str,
    cosine_weight: float,
    mse_weight: float,
    pairwise_weight: float,
    slides_per_step: int,
    train: bool,
    log_every: int,
    global_step: int,
    use_wandb: bool,
    quick_val_loader: DataLoader | None = None,
    quick_val_every: int = 0,
) -> tuple[dict[str, float], int]:
    student.train(train)
    sums = {"loss": 0.0, "cosine": 0.0, "mse": 0.0, "pairwise": 0.0}
    total = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()

    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for step, (tile_embeddings, coords, teacher_embedding, slide_id) in enumerate(iterator):
        tile_embeddings = tile_embeddings.squeeze(0).to(device, non_blocking=True)
        coords = coords.squeeze(0).to(device, non_blocking=True)
        teacher_embedding = teacher_embedding.to(device, non_blocking=True)
        try:
            with torch.set_grad_enabled(train), _autocast(device, amp_dtype):
                student_embedding = student(tile_embeddings, coords)
                loss, parts = embedding_distillation_loss(
                    student_embedding.unsqueeze(0).float(),
                    teacher_embedding.float(),  # already [1,768]: batch_size=1 DataLoader adds the batch dim
                    cosine_weight=cosine_weight,
                    mse_weight=mse_weight,
                    pairwise_weight=pairwise_weight,
                )
            if train:
                (loss / slides_per_step).backward()
        except torch.cuda.OutOfMemoryError as exc:
            # Same rationale as train_wsi_landmark_forecaster.py's oom-skip: a handful
            # of extreme-tile-count slides can still spike memory even after pruning
            # (the un-prunable prefix, blocks[:prune_layer+1], still runs on the FULL
            # bag) -- skip rather than lose the whole run to one slide.
            if train:
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"[oom-skip] slide={slide_id} n_tiles={tile_embeddings.shape[0]} train={train}: {exc}", flush=True)
            continue

        if train:
            if (step + 1) % slides_per_step == 0 or step + 1 == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if use_wandb and global_step % log_every == 0:
                    wandb.log(
                        {
                            "step/loss": float(loss.detach()),
                            "step/cosine": float(parts["cosine"].detach()),
                            "step/mse": float(parts["mse"].detach()),
                            "step/pairwise": float(parts["pairwise"].detach()),
                            "step/grad_norm": float(grad_norm),
                            "trainer/global_step": global_step,
                        },
                        step=global_step,
                    )
                if quick_val_loader is not None and quick_val_every > 0 and global_step % quick_val_every == 0:
                    # A fast pulse check on a small, fixed validation subset -- full
                    # validation only happens once per epoch (~8-9h away on this
                    # corpus), too slow to inform an early stop/keep-going decision.
                    # Does not touch what training itself reads (still every tile of
                    # every training slide) -- this only affects how often we *check*.
                    quick_metrics, _ = _run_epoch(
                        student=student, loader=quick_val_loader, optimizer=None, device=device, amp_dtype=amp_dtype,
                        cosine_weight=cosine_weight, mse_weight=mse_weight, pairwise_weight=pairwise_weight,
                        slides_per_step=slides_per_step, train=False, log_every=log_every, global_step=global_step,
                        use_wandb=False,
                    )
                    student.train(True)
                    print(
                        f"  [quickval @ step {global_step}] loss={quick_metrics['loss']:.4f} "
                        f"cosine={quick_metrics['cosine']:.4f}",
                        flush=True,
                    )
                    if use_wandb:
                        wandb.log({f"quickval/{k}": v for k, v in quick_metrics.items()}, step=global_step)

        sums["loss"] += float(loss.detach())
        sums["cosine"] += float(parts["cosine"].detach())
        sums["mse"] += float(parts["mse"].detach())
        sums["pairwise"] += float(parts["pairwise"].detach())
        total += 1

    elapsed = max(time.perf_counter() - start, 1e-6)
    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    metrics["slides_per_second"] = total / elapsed
    return metrics, global_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-eaf-root", type=Path, required=True)
    parser.add_argument("--wsi-eaf-root", type=Path, required=True)
    parser.add_argument("--forecaster-checkpoint", type=Path, required=True)
    parser.add_argument("--prune-layer", type=int, default=None, help="defaults to the forecaster's own trained source layer")
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=17, help="matches the forecaster's own splits")
    parser.add_argument("--patch-size-level0", type=int, default=512)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--slides-per-step", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--quick-val-every", type=int, default=0,
        help="Run a fast pulse-check validation (fixed small subset, --quick-val-slides) every this many "
        "optimizer steps. 0 disables it. Full validation still happens once per epoch regardless -- this "
        "exists only because a full epoch on this corpus can take hours, too slow to catch a diverging or "
        "already-converged run early.",
    )
    parser.add_argument("--quick-val-slides", type=int, default=100)

    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecaster, forecaster_source_layer = _load_forecaster(args.forecaster_checkpoint, device=device)
    prune_layer = args.prune_layer if args.prune_layer is not None else forecaster_source_layer
    if prune_layer != forecaster_source_layer:
        raise SystemExit(
            f"--prune-layer={prune_layer} does not match the forecaster checkpoint's own "
            f"trained source layer ({forecaster_source_layer}); the forecaster was never "
            "shown any other layer's hidden state and its scores would be meaningless here."
        )

    manifest_config = WSIForecasterManifestConfig(
        tile_eaf_root=args.tile_eaf_root,
        wsi_eaf_root=args.wsi_eaf_root,
        cohorts=tuple(args.cohorts) if args.cohorts else None,
    )
    table = build_manifest(manifest_config)
    table = assign_splits(
        table, train_fraction=args.train_fraction, validation_fraction=args.validation_fraction, seed=args.split_seed
    )
    write_manifest_csv(table, args.output_dir / "manifest.csv")
    print("slides: " + ", ".join(f"{split}={count}" for split, count in table["split"].value_counts().sort_index().items()))

    def make_loader(split: str, *, shuffle: bool) -> DataLoader:
        dataset = WSIPrunedTitanDataset(table, split=split)
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("validation", shuffle=False)

    quick_val_loader = None
    if args.quick_val_every > 0:
        full_val_dataset = WSIPrunedTitanDataset(table, split="validation")
        subset_size = min(args.quick_val_slides, len(full_val_dataset))
        quick_val_dataset = torch.utils.data.Subset(full_val_dataset, range(subset_size))
        quick_val_loader = DataLoader(quick_val_dataset, batch_size=1, shuffle=False, num_workers=min(4, args.num_workers))
        print(f"quick-val: {subset_size} slides, every {args.quick_val_every} optimizer steps")

    titan_model = TitanAdapter(token=args.hf_token).model
    student = PrunedLoRATitanEncoder(
        titan_model,
        forecaster,
        prune_layer=prune_layer,
        keep_ratio=args.keep_ratio,
        patch_size_level0=args.patch_size_level0,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    trainable = [p for p in student.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"trainable (LoRA) parameters: {n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0
    latest_ckpt_path = args.output_dir / "latest_pruned_titan.pt"
    best_ckpt_path = args.output_dir / "best_pruned_titan.pt"
    if args.resume and latest_ckpt_path.exists():
        payload = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        student.load_trainable_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = payload["epoch"] + 1
        global_step = payload["global_step"]
        best_val_loss = payload["best_val_loss"]
        print(f"resumed from {latest_ckpt_path}: epoch={payload['epoch']} val_loss={payload['val_loss']:.4f}")

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    for epoch in range(start_epoch, args.epochs):
        train_metrics, global_step = _run_epoch(
            student=student, loader=train_loader, optimizer=optimizer, device=device, amp_dtype=args.amp_dtype,
            cosine_weight=args.cosine_weight, mse_weight=args.mse_weight, pairwise_weight=args.pairwise_weight,
            slides_per_step=args.slides_per_step, train=True, log_every=args.log_every, global_step=global_step,
            use_wandb=use_wandb, quick_val_loader=quick_val_loader, quick_val_every=args.quick_val_every,
        )
        val_metrics, _ = _run_epoch(
            student=student, loader=val_loader, optimizer=None, device=device, amp_dtype=args.amp_dtype,
            cosine_weight=args.cosine_weight, mse_weight=args.mse_weight, pairwise_weight=args.pairwise_weight,
            slides_per_step=args.slides_per_step, train=False, log_every=args.log_every, global_step=global_step,
            use_wandb=False,
        )
        scheduler.step()
        print(
            f"epoch {epoch:03d} keep_ratio={args.keep_ratio} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_cosine={val_metrics['cosine']:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if use_wandb:
            wandb.log(
                {**{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}, "epoch": epoch},
                step=global_step,
            )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {"model": student.trainable_state_dict(), "args": vars(args), "epoch": epoch, "val_loss": best_val_loss},
                best_ckpt_path,
            )
        torch.save(
            {
                "model": student.trainable_state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_metrics["loss"],
                "best_val_loss": best_val_loss,
            },
            latest_ckpt_path,
        )

    print(f"best val_loss={best_val_loss:.4f}, checkpoint at {best_ckpt_path}")
    if use_wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
