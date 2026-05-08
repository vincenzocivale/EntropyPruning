"""Phase 1: Train a classifier on a Thunder backbone using a configurable adaptation strategy."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, build_optimizer, grad_norm, save_results
from src.models import ThunderBackboneAdapter, build_classifier, STRATEGIES
from src.data.thunder_loaders import build_thunder_loaders
from src.evaluation import evaluate


def run_train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    """Run one training epoch. Returns (loss, acc, avg_grad_norm)."""
    model.train()
    total_loss, correct, total = 0., 0, 0
    total_gnorm = 0.

    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        with autocast("cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Capture grad norm before clipping — useful for diagnosing instability
        gnorm = grad_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
        total_gnorm += gnorm

    n_batches = len(loader)
    return total_loss / total, correct / total, total_gnorm / n_batches


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Train base classifier with configurable adaptation")
    # --- Thunder model / dataset ---
    parser.add_argument("--model-name", type=str, required=True,
                        help="Thunder model name (e.g. uni, hoptimus0, virchow, dinov2base)")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Thunder dataset name (e.g. crc, break_his, mhist)")
    parser.add_argument("--base-data-folder", type=str, required=True,
                        help="Path containing data_splits/ and datasets/")
    # --- Adaptation strategy ---
    parser.add_argument("--adaptation", type=str, default="lora",
                        choices=STRATEGIES,
                        help="Backbone adaptation strategy (default: lora)")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank — only used when --adaptation lora")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha — only used when --adaptation lora")
    # --- Training hyperparameters ---
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # --- Experiment tracking ---
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (e.g. 'eaf'). Omit to skip W&B.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name. Auto-generated if not set.")
    parser.add_argument("--early-stopping-metric", type=str, default="f1_macro",
                        choices=["acc", "f1_macro"],
                        help="Metric used to select the best checkpoint (default: f1_macro)")
    parser.add_argument("--far-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # --- Backbone ---
    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)

    # --- Data ---
    train_loader, val_loader, test_loader, class_names, n_classes = \
        build_thunder_loaders(
            args.dataset_name, args.base_data_folder, transform,
            args.batch_size, args.num_workers, drop_last_train=True,
        )

    # --- Output directory ---
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(f"checkpoints/{args.dataset_name}/{args.model_name}_{args.adaptation}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Model ---
    kwargs = {"dropout": args.dropout}
    if args.adaptation == "lora":
        kwargs.update(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    model = build_classifier(args.adaptation, raw_backbone, adapter, n_classes,
                             **kwargs).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * n_trainable / n_total

    print(f"\nDevice: {device} | Model: {args.model_name} | "
          f"Dataset: {args.dataset_name} | Adaptation: {args.adaptation}")
    print(f"embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}  "
          f"n_patches={adapter.n_patches}  prefix={adapter.num_prefix_tokens}")
    print(f"Classes ({n_classes}): {class_names}")
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({trainable_pct:.2f}%)")

    # --- W&B ---
    run_name = args.run_name or (
        f"{args.model_name}_{args.dataset_name}_{args.adaptation}"
    )
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            job_type="phase1",
            group=f"{args.dataset_name}/{args.model_name}",
            config={
                **vars(args),
                "n_trainable_params": n_trainable,
                "n_total_params": n_total,
                "trainable_pct": trainable_pct,
                "n_classes": n_classes,
                "embed_dim": adapter.embed_dim,
                "n_blocks": adapter.n_blocks,
            },
            tags=[args.model_name, args.dataset_name, args.adaptation, "phase1"],
        )

    # --- Optimizer & scheduler ---
    optimizer = build_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler("cuda")

    # --- Training loop ---
    best_metric, best_epoch = -1., 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, gnorm = run_train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device)

        val_metrics = evaluate(model, val_loader, device, args.far_threshold)

        current_lr = scheduler.get_last_lr()
        lr_backbone = current_lr[0] if len(current_lr) > 1 else 0.
        lr_head = current_lr[-1]

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_grad_norm": gnorm,
            "val_acc": val_metrics["acc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_tar_at_far": val_metrics["tar_at_far"],
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
        }
        history.append(row)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  gnorm={gnorm:.3f}  |  "
              f"val_acc={val_metrics['acc']:.4f}  val_f1={val_metrics['f1_macro']:.4f}")

        if use_wandb:
            wandb.log(row)

        # Checkpoint on chosen metric
        current = val_metrics[args.early_stopping_metric]
        if current > best_metric:
            best_metric, best_epoch = current, epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  → saved (val_{args.early_stopping_metric}={best_metric:.4f})")

    print(f"\nBest val_{args.early_stopping_metric}={best_metric:.4f} @ epoch {best_epoch}")

    # --- Test evaluation ---
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Test"):
            all_preds.append(model(imgs.to(device)).argmax(1).cpu())
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    test_metrics = evaluate(model, test_loader, device, args.far_threshold)

    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    print(report_str)

    # --- Persist results ---
    (output_dir / "classification_report.txt").write_text(report_str)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2))

    results = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "adaptation": args.adaptation,
        "n_classes": n_classes,
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
        "trainable_pct": round(trainable_pct, 4),
        "best_epoch": best_epoch,
        "best_val_metric": args.early_stopping_metric,
        "best_val_value": round(best_metric, 6),
        "test_acc": round(float(test_metrics["acc"]), 6),
        "test_f1_macro": round(float(test_metrics["f1_macro"]), 6),
        "test_tar_at_far": round(float(test_metrics["tar_at_far"]), 6),
        "args": vars(args),
    }
    path = save_results(output_dir / "results.json", results)
    print(f"\nResults saved to: {path}")

    if use_wandb:
        wandb.log({
            "test/acc": test_metrics["acc"],
            "test/f1_macro": test_metrics["f1_macro"],
            "test/tar_at_far": test_metrics["tar_at_far"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
