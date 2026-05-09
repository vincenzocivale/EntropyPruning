"""Phase 1: Train CropR on a Thunder foundation model and dataset."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, grad_norm, save_results
from src.models import CroprClassifier, ThunderBackboneAdapter
from src.data.thunder_loaders import build_thunder_loaders
from src.evaluation import evaluate


def compute_loss(output, labels, criterion):
    """Sum CE losses over all outputs (main + CropR auxiliary heads)."""
    if isinstance(output, list):
        return sum(criterion(logits, labels) for logits in output)
    return criterion(output, labels)


def run_train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, correct, total, total_gnorm = 0.0, 0, 0, 0.0

    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        with autocast("cuda"):
            output = model(imgs)
            loss = compute_loss(output, labels, criterion)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        total_gnorm += grad_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        final_logits = output[0] if isinstance(output, list) else output
        total_loss += loss.item() * len(labels)
        correct += (final_logits.argmax(1) == labels).sum().item()
        total += len(labels)

    n = len(loader)
    return total_loss / total, correct / total, total_gnorm / n


def main():
    parser = argparse.ArgumentParser("Train CropR on a Thunder backbone + dataset")
    # Model / dataset
    parser.add_argument("--model-name", type=str, required=True,
                        help="Thunder model name (e.g. uni, virchow2, hoptimus0)")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Thunder dataset name (e.g. crc, mhist, break_his)")
    parser.add_argument("--base-data-folder", type=str, required=True,
                        help="Root folder containing data_splits/ and dataset files")
    # CropR hyperparameters
    parser.add_argument("--pruning-rate", type=int, default=8,
                        help="Tokens removed per block (default: 8, as in the paper)")
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--cropr-num-heads", type=int, default=1)
    parser.add_argument("--pre-attn-norm", action="store_true")
    parser.add_argument("--no-mlp", action="store_true", help="Disable MLP in CropR modules")
    # Backbone adaptation
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone weights; train only CropR modules + head")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="LR for CropR modules + classification head")
    parser.add_argument("--lr-backbone", type=float, default=1e-4,
                        help="LR for LoRA backbone params (ignored when --freeze-backbone)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--far-threshold", type=float, default=1e-4)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (omit to disable logging)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    adaptation = "frozen" if args.freeze_backbone else f"lora_r{args.lora_r}"
    print(f"Device: {device} | Model: {args.model_name} | Dataset: {args.dataset_name} | "
          f"Adaptation: {adaptation} | Pruning rate: {args.pruning_rate}")

    # --- Backbone ---
    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))

    # --- Data ---
    train_loader, val_loader, test_loader, class_names, n_classes = build_thunder_loaders(
        args.dataset_name, args.base_data_folder, transform,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Classes ({n_classes}): {class_names}")
    print(f"Train / Val / Test batches: {len(train_loader)} / {len(val_loader)} / {len(test_loader)}")

    # --- Model ---
    adapter = ThunderBackboneAdapter(raw_backbone)
    model = CroprClassifier(
        backbone=raw_backbone,
        adapter=adapter,
        n_classes=n_classes,
        pruning_rate=args.pruning_rate,
        num_queries=args.num_queries,
        cropr_num_heads=args.cropr_num_heads,
        pre_attn_norm=args.pre_attn_norm,
        mlp=not args.no_mlp,
        freeze_backbone=args.freeze_backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    # --- Optimizer ---
    # CropR modules + head share lr_head; backbone LoRA params use lr_backbone.
    cropr_head_params = (
        list(model.cropr_modules.parameters()) + list(model.head.parameters())
    )
    param_groups = [{"params": cropr_head_params, "lr": args.lr_head}]
    backbone_params = model.trainable_backbone_params
    if backbone_params:
        param_groups.insert(0, {"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # --- Output dir ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path("/raid/DATASETS/checkpoints-Attention-Pruning")
            / args.dataset_name
            / f"{args.model_name}_cropr_{adaptation}_pr{args.pruning_rate}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    run_name = (f"{args.model_name}_{args.dataset_name}_cropr_{adaptation}"
                f"_pr{args.pruning_rate}")
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=[args.model_name, args.dataset_name, "cropr",
                  f"pr{args.pruning_rate}", adaptation],
        )

    # --- Training loop ---
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc, train_gnorm = run_train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        val_metrics = evaluate(model, val_loader, device, args.far_threshold)

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | val_f1={val_metrics['f1_macro']:.4f} | "
            f"gnorm={train_gnorm:.3f}"
        )

        if args.wandb_project:
            wandb.log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/grad_norm": train_gnorm,
                "val/acc": val_metrics["acc"],
                "val/f1_macro": val_metrics["f1_macro"],
                "val/tar_at_far": val_metrics["tar_at_far"],
                "epoch": epoch,
            })

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> New best val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={args.early_stopping_patience})")
                break

    # --- Final test evaluation ---
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device, args.far_threshold)
    print(
        f"Test: acc={test_metrics['acc']:.4f} | f1={test_metrics['f1_macro']:.4f} | "
        f"tar@far={test_metrics['tar_at_far']:.4f}"
    )

    results = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "pruning_rate": args.pruning_rate,
        "adaptation": adaptation,
        "best_val_acc": float(best_val_acc),
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        "config": vars(args),
    }
    save_results(output_dir / "results.json", results)

    if args.wandb_project:
        wandb.summary.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
        wandb.finish()

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
