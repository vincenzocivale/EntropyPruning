"""Phase 3: Fine-tune classifier with forecaster-guided token pruning."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import set_seed, get_device
from src.models import (UNILoRAClassifier, AttentionForecaster,
                        UNILoRAWithForecasterPruning)
from src.data.loaders import build_loaders
from src.evaluation import evaluate, benchmark_model


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Fine-tune pruned model")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--forecaster-ckpt", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--far-threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str,
                        default="pruned-finetuning")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--eval-baseline", action="store_true",
                        help="Also evaluate unpruned baseline")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    dataset_name = Path(args.data_dir).name
    base_ckpt = Path("/raid/DATASETS/checkpoints-Attention-Pruning/")

    classifier_ckpt = args.classifier_ckpt or str(
        base_ckpt / dataset_name / "uni_finetuned" / "best_model.pt")
    forecaster_ckpt = args.forecaster_ckpt or str(
        base_ckpt / dataset_name / "forecaster" /
        f"forecaster_src{args.prune_layer:02d}_tgt23.pt")
    output_dir = Path(args.output_dir) if args.output_dir else \
        base_ckpt / dataset_name / "pruned_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names, n_classes = \
        build_loaders(args.data_dir, args.img_size, args.batch_size,
                      args.num_workers, drop_last_train=True)

    # Optionally evaluate baseline
    if args.eval_baseline:
        baseline_model = UNILoRAClassifier(n_classes).to(device)
        ckpt = torch.load(classifier_ckpt, map_location=device)
        baseline_model.load_state_dict(ckpt, strict=False)
        baseline_model.eval()
        for p in baseline_model.parameters():
            p.requires_grad_(False)

        baseline_metrics = evaluate(baseline_model, test_loader, device,
                                    args.far_threshold)
        baseline_bench = benchmark_model(baseline_model, test_loader, device,
                                         label="Baseline (no pruning)")
        print(f"\n-- Baseline Test --")
        print(f"  Accuracy:  {baseline_metrics['acc']:.3f}")
        print(f"  F1 macro:  {baseline_metrics['f1_macro']:.3f}")
        print(f"  ms/img:    {baseline_bench['ms_per_img']:.2f}")
        print(f"  GFLOPs:    {baseline_bench['gflops']:.2f}")
        del baseline_model

    # Load forecaster
    forecaster = AttentionForecaster().to(device)
    forecaster.load_state_dict(
        torch.load(forecaster_ckpt, map_location=device))
    forecaster.eval()
    for p in forecaster.parameters():
        p.requires_grad_(False)
    print("Forecaster loaded and frozen")

    # Build pruned model
    model = UNILoRAWithForecasterPruning(
        n_classes=n_classes, forecaster=forecaster,
        prune_layer=args.prune_layer, keep_ratio=args.keep_ratio,
    ).to(device)

    ckpt = torch.load(classifier_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    # Pre fine-tuning eval
    pre_metrics = evaluate(model, val_loader, device, args.far_threshold)
    print(f"\nPre fine-tuning val -- acc={pre_metrics['acc']:.3f} "
          f"f1={pre_metrics['f1_macro']:.3f}")

    # Optimizer & scheduler
    backbone_params = [p for _, p in model.backbone.named_parameters()
                       if p.requires_grad]
    head_params = list(model.head.parameters())

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    # each optimizer step covers grad_accum mini-batches
    steps_per_epoch = (len(train_loader) + args.grad_accum - 1) // args.grad_accum
    total_steps = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_backbone, args.lr_head],
        total_steps=total_steps, pct_start=0.1)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler('cuda')

    # Training
    run_name = (f"prune_layer{args.prune_layer}"
                f"_keep{int(args.keep_ratio * 100)}")
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args),
               tags=[f"prune_layer{args.prune_layer}",
                     f"keep{int(args.keep_ratio * 100)}", dataset_name])

    best_val_f1 = 0.
    for epoch in range(args.epochs):
        model.train()
        # accumulate on GPU — pull to CPU once per epoch
        total_loss = torch.tensor(0., device=device)
        all_preds, all_labels = [], []

        opt.zero_grad()
        for step, (imgs, labels) in enumerate(
            tqdm(train_loader, leave=False, desc=f"Ep{epoch+1}")
        ):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels) / args.grad_accum

            scaler.scale(loss).backward()

            is_last = (step + 1) == len(train_loader)
            if (step + 1) % args.grad_accum == 0 or is_last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()

            total_loss += loss.detach() * args.grad_accum * len(labels)
            all_preds.append(logits.argmax(-1).cpu())
            all_labels.append(labels.cpu())

        train_f1 = f1_score(torch.cat(all_labels).numpy(),
                            torch.cat(all_preds).numpy(),
                            average='macro', zero_division=0)
        train_loss = total_loss.item() / len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, device, args.far_threshold)

        wandb.log({
            "epoch": epoch + 1, "train/loss": train_loss,
            "train/f1_macro": train_f1,
            "val/acc": val_metrics["acc"],
            "val/f1_macro": val_metrics["f1_macro"],
            "val/tar_at_far": val_metrics["tar_at_far"],
        })

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(),
                       output_dir / f"best_{run_name}.pt")

        print(f"Ep {epoch+1:02d} | loss={train_loss:.4f} | "
              f"train_f1={train_f1:.3f} | "
              f"val_f1={val_metrics['f1_macro']:.3f} | "
              f"best_f1={best_val_f1:.3f}")

    # Test evaluation
    model.load_state_dict(
        torch.load(output_dir / f"best_{run_name}.pt", map_location=device))
    model.eval()

    test_metrics = evaluate(model, test_loader, device, args.far_threshold)
    test_bench = benchmark_model(
        model, test_loader, device,
        label=f"Pruned (layer={args.prune_layer}, "
              f"keep={int(args.keep_ratio * 100)}%)")

    print(f"\n-- Test Results --")
    print(f"  Accuracy:  {test_metrics['acc']:.3f}")
    print(f"  F1 macro:  {test_metrics['f1_macro']:.3f}")
    print(f"  TAR@FAR:   {test_metrics['tar_at_far']:.3f}")
    print(f"  ms/img:    {test_bench['ms_per_img']:.2f}")
    print(f"  GFLOPs:    {test_bench['gflops']:.2f}")

    wandb.log({
        "test/acc": test_metrics["acc"],
        "test/f1_macro": test_metrics["f1_macro"],
        "test/tar_at_far": test_metrics["tar_at_far"],
        "test/ms_per_img": test_bench["ms_per_img"],
        "test/gflops": test_bench["gflops"],
    })
    wandb.finish()


if __name__ == "__main__":
    main()
