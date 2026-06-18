"""Phase 3: Fine-tune classifier with forecaster-guided token pruning."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, grad_norm, save_results
from src.models import (AttentionForecaster, GenericLoRAWithForecasterPruning,
                        ThunderBackboneAdapter, STRATEGIES)
from src.data.thunder_loaders import build_thunder_loaders
from src.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Fine-tune pruned model")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--forecaster-ckpt", type=str, default=None)
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
    parser.add_argument("--hidden", type=int, default=256,
                        help="AttentionForecaster hidden dim — must match Phase 2.")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (default: None = skip W&B).")
    parser.add_argument("--adaptation", type=str, default="lora", choices=STRATEGIES,
                        help="Phase 1 adaptation strategy.")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Epochs without improvement before stopping (default: 3)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Model: {args.model_name} | Dataset: {args.dataset_name}")

    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    print(f"embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}  "
          f"n_patches={adapter.n_patches}  prefix={adapter.num_prefix_tokens}")
    assert args.prune_layer < adapter.n_blocks - 1, \
        (f"--prune-layer {args.prune_layer} leaves no blocks to LoRA-adapt "
         f"(n_blocks={adapter.n_blocks})")

    base_ckpt = Path("checkpoints")
    classifier_ckpt = args.classifier_ckpt or str(
        base_ckpt / args.dataset_name / f"{args.model_name}_{args.adaptation}" / "best_model.pt")
    forecaster_ckpt = args.forecaster_ckpt or str(
        base_ckpt / args.dataset_name / f"{args.model_name}_forecaster" /
        f"forecaster_src{args.prune_layer:02d}_tgt{adapter.n_blocks-1:02d}.pt")
    output_dir = Path(args.output_dir) if args.output_dir else \
        base_ckpt / args.dataset_name / f"{args.model_name}_pruned"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names, n_classes = \
        build_thunder_loaders(args.dataset_name, args.base_data_folder, transform,
                              args.batch_size, args.num_workers, drop_last_train=True)
    print(f"Classes ({n_classes}): {class_names}")

    # --- W&B init (before baseline so baseline logs appear at step 0) ---
    run_name = (f"{args.model_name}_{args.dataset_name}"
                f"_prune{args.prune_layer}_keep{int(args.keep_ratio * 100)}")
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            job_type="phase3",
            group=f"{args.dataset_name}/{args.model_name}",
            config=vars(args),
            tags=[args.model_name, args.dataset_name,
                  f"prune_layer_{args.prune_layer}",
                  f"keep_{int(args.keep_ratio * 100)}pct",
                  "phase3"],
        )

    # --- Forecaster ---
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=args.hidden, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    forecaster.load_state_dict(torch.load(forecaster_ckpt, map_location=device))
    forecaster.eval()
    for p in forecaster.parameters():
        p.requires_grad_(False)
    print(f"Forecaster loaded: {forecaster_ckpt}")

    # --- Pruned model ---
    model = GenericLoRAWithForecasterPruning(
        backbone=raw_backbone, adapter=adapter, n_classes=n_classes,
        forecaster=forecaster, prune_layer=args.prune_layer, keep_ratio=args.keep_ratio,
    ).to(device)
    if Path(classifier_ckpt).exists():
        missing, unexpected = model.load_state_dict(
            torch.load(classifier_ckpt, map_location=device), strict=False)
        print(f"Checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"No Phase 1 classifier checkpoint at {classifier_ckpt} — "
              f"starting from pretrained backbone + freshly initialized head "
              f"(unsupervised EAF flow, no Phase 1 required).")

    pre = evaluate(model, val_loader, device, args.far_threshold)
    print(f"\nPre fine-tuning val: acc={pre['acc']:.3f}  f1={pre['f1_macro']:.3f}")
    if use_wandb:
        wandb.config.update({
            "pre_val_acc": pre["acc"],
            "pre_val_f1_macro": pre["f1_macro"],
        })

    # --- Optimizer (AMP) ---
    backbone_params = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_backbone, args.lr_head],
        total_steps=total_steps, pct_start=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler("cuda")

    # --- Training ---
    best_val_f1 = 0.
    ckpt_name = f"best_{run_name}.pt"
    history = []
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_gnorm = 0., 0.
        all_preds, all_labels_ep = [], []

        for imgs, labels in tqdm(train_loader, leave=False, desc=f"Ep{epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            total_gnorm += grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item() * len(labels)
            all_preds.append(logits.argmax(-1).cpu())
            all_labels_ep.append(labels.cpu())

        n_batches = len(train_loader)
        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels_ep).numpy()
        train_acc = (all_preds_np == all_labels_np).mean()

        val_m = evaluate(model, val_loader, device, args.far_threshold)

        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader.dataset),
            "train_acc": float(train_acc),
            "train_grad_norm": total_gnorm / n_batches,
            "val_acc": val_m["acc"],
            "val_f1_macro": val_m["f1_macro"],
            "val_tar_at_far": val_m["tar_at_far"],
        }
        history.append(row)

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": row["train_loss"],
                "train/acc": row["train_acc"],
                "train/grad_norm": row["train_grad_norm"],
                "val/acc": val_m["acc"],
                "val/f1_macro": val_m["f1_macro"],
                "val/tar_at_far": val_m["tar_at_far"],
            })

        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / ckpt_name)
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping: no improvement for {epochs_without_improvement} epochs")
                break

        print(f"Ep {epoch+1:02d} | loss={row['train_loss']:.4f}  "
              f"gnorm={row['train_grad_norm']:.3f}  "
              f"f1_val={val_m['f1_macro']:.4f}  best={best_val_f1:.4f}")

    # --- Test evaluation ---
    model.load_state_dict(torch.load(output_dir / ckpt_name, map_location=device))
    model.eval()
    test_m = evaluate(model, test_loader, device, args.far_threshold)

    print(f"\n-- Test --  acc={test_m['acc']:.3f}  f1={test_m['f1_macro']:.3f}  "
          f"TAR@FAR={test_m['tar_at_far']:.3f}")

    # --- Persist results ---
    results = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "prune_layer": args.prune_layer,
        "keep_ratio": args.keep_ratio,
        "n_classes": n_classes,
        "pre_val_acc": round(pre["acc"], 6),
        "pre_val_f1_macro": round(pre["f1_macro"], 6),
        "best_val_f1_macro": round(best_val_f1, 6),
        "test_acc": round(float(test_m["acc"]), 6),
        "test_f1_macro": round(float(test_m["f1_macro"]), 6),
        "test_tar_at_far": round(float(test_m["tar_at_far"]), 6),
        "args": vars(args),
    }
    path = save_results(output_dir / f"results_{run_name}.json", results)
    print(f"Results saved to: {path}")

    if use_wandb:
        wandb.log({
            "test/acc": test_m["acc"],
            "test/f1_macro": test_m["f1_macro"],
            "test/tar_at_far": test_m["tar_at_far"],
            "val/best_f1_macro": best_val_f1,
        }, step=args.epochs)
        wandb.finish()


if __name__ == "__main__":
    main()
