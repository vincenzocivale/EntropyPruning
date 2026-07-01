"""Phase 3 (multi-dataset): Fine-tune pruned encoder across Thunder training datasets.

Uses the holdout plan from Phase 1 to exclude the same datasets.
Applies forecaster-guided token pruning via MultiHeadLoRAWithForecasterPruning.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, grad_norm, save_results
from src.models import AttentionForecaster, ThunderBackboneAdapter, STRATEGIES
from src.models.multi_head_classifier import MultiHeadLoRAWithForecasterPruning
from src.data.thunder_multi import ThunderDatasetRegistry, build_multi_thunder_train_loaders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multi_head_loss(criterion, output: dict, labels: torch.Tensor,
                     dataset_indices: torch.Tensor) -> torch.Tensor:
    losses = []
    for k, logits in output.items():
        mask = dataset_indices == k
        losses.append(criterion(logits, labels[mask]))
    return torch.stack(losses).mean()


@torch.no_grad()
def _evaluate(model, loader, device, dataset_info: dict) -> dict:
    model.eval()
    per_correct: dict = defaultdict(int)
    per_total: dict = defaultdict(int)

    for imgs, labels, dataset_indices in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        dataset_indices = dataset_indices.to(device)
        output = model(imgs, dataset_indices)
        for k, logits in output.items():
            mask = dataset_indices == k
            per_correct[k] += (logits.argmax(1) == labels[mask]).sum().item()
            per_total[k] += mask.sum().item()

    per_dataset_acc = {
        dataset_info[k]["name"]: per_correct[k] / max(per_total[k], 1)
        for k in per_total
    }
    macro_acc = sum(per_dataset_acc.values()) / max(len(per_dataset_acc), 1)
    micro_acc = sum(per_correct.values()) / max(sum(per_total.values()), 1)
    return {"macro_acc": macro_acc, "micro_acc": micro_acc, "per_dataset": per_dataset_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 (multi-dataset): Fine-tune pruned encoder on Thunder datasets")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    # Holdout
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--holdout-plan", type=str,
                     help="Path to holdout_plan.json from Phase 1 (recommended)")
    grp.add_argument("--n-holdout", type=int,
                     help="Re-compute holdout using N smallest datasets")
    parser.add_argument("--holdout-datasets", type=str, nargs="+", default=None)
    # Phase 1 / 2 checkpoints
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--forecaster-ckpt", type=str, required=True)
    parser.add_argument("--adaptation", type=str, default="lora", choices=STRATEGIES)
    # Pruning
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    # Forecaster arch (must match Phase 2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    # Output / W&B
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # --- Registry ---
    if args.holdout_plan:
        registry = ThunderDatasetRegistry.from_plan(
            args.holdout_plan, args.base_data_folder)
    else:
        registry = ThunderDatasetRegistry(
            args.base_data_folder,
            n_holdout=args.n_holdout,
            holdout_datasets=args.holdout_datasets,
        )
    print(f"Train  ({len(registry.train_datasets)}): {registry.train_datasets}")
    print(f"Holdout ({len(registry.holdout_datasets)}): {registry.holdout_datasets}")

    # --- Backbone ---
    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    print(f"embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}  "
          f"n_patches={adapter.n_patches}")
    assert args.prune_layer < adapter.n_blocks, \
        f"--prune-layer {args.prune_layer} >= n_blocks {adapter.n_blocks}"

    # --- Data ---
    train_loader, val_loader, dataset_info = build_multi_thunder_train_loaders(
        registry.train_datasets,
        args.base_data_folder,
        transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last_train=True,
    )

    # --- Output dir ---
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(f"checkpoints/multi_thunder/{args.model_name}_pruned")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Forecaster (frozen) ---
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=args.hidden, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    forecaster.load_state_dict(torch.load(args.forecaster_ckpt, map_location=device))
    forecaster.eval()
    for p in forecaster.parameters():
        p.requires_grad_(False)
    print(f"Forecaster loaded: {args.forecaster_ckpt}")

    # --- Pruned model ---
    model = MultiHeadLoRAWithForecasterPruning(
        backbone=raw_backbone, adapter=adapter, dataset_info=dataset_info,
        forecaster=forecaster, prune_layer=args.prune_layer, keep_ratio=args.keep_ratio,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, dropout=args.dropout,
    ).to(device)

    if args.classifier_ckpt:
        missing, unexpected = model.load_state_dict(
            torch.load(args.classifier_ckpt, map_location=device), strict=False)
        print(f"Checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")

    pre = _evaluate(model, val_loader, device, dataset_info)
    print(f"\nPre fine-tuning val: macro_acc={pre['macro_acc']:.3f}  "
          f"micro_acc={pre['micro_acc']:.3f}")

    # --- W&B ---
    run_name = args.run_name or (
        f"{args.model_name}_multi_prune{args.prune_layer}_keep{int(args.keep_ratio*100)}"
    )
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project, name=run_name, job_type="phase3_multi",
            config={**vars(args), "train_datasets": registry.train_datasets,
                    "holdout_datasets": registry.holdout_datasets},
            tags=[args.model_name, f"prune_layer_{args.prune_layer}",
                  f"keep_{int(args.keep_ratio*100)}pct", "phase3", "multi_dataset"],
        )

    # --- Optimizer ---
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups = [{"params": model.heads.parameters(), "lr": args.lr_head}]
    if backbone_params:
        param_groups.insert(0, {"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_head] if backbone_params else [args.lr_head],
        total_steps=total_steps,
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler("cuda")

    # --- Training ---
    best_val_macro = 0.
    ckpt_name = f"best_{run_name}.pt"
    history = []
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_correct, total_n, total_gnorm = 0., 0, 0, 0.

        for imgs, labels, dataset_indices in tqdm(train_loader, leave=False,
                                                   desc=f"Ep{epoch+1}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            dataset_indices = dataset_indices.to(device)

            with autocast("cuda"):
                output = model(imgs, dataset_indices)
                loss = _multi_head_loss(criterion, output, labels, dataset_indices)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_gnorm += grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            n = len(labels)
            total_loss += loss.item() * n
            total_correct += sum(
                (logits.argmax(1) == labels[dataset_indices == k]).sum().item()
                for k, logits in output.items()
            )
            total_n += n

        n_batches = len(train_loader)
        val_m = _evaluate(model, val_loader, device, dataset_info)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total_n,
            "train_acc": total_correct / total_n,
            "train_grad_norm": total_gnorm / n_batches,
            "val_macro_acc": val_m["macro_acc"],
            "val_micro_acc": val_m["micro_acc"],
        }
        history.append(row)

        if use_wandb:
            log = {"epoch": epoch + 1, **{f"train/{k}": v for k, v in row.items()
                                           if k != "epoch"},
                   "val/macro_acc": val_m["macro_acc"],
                   "val/micro_acc": val_m["micro_acc"]}
            log.update({f"val/{n}_acc": a for n, a in val_m["per_dataset"].items()})
            wandb.log(log)

        if val_m["macro_acc"] > best_val_macro:
            best_val_macro = val_m["macro_acc"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / ckpt_name)
        else:
            epochs_no_improve += 1
            if (args.early_stopping_patience > 0
                    and epochs_no_improve >= args.early_stopping_patience):
                print(f"Early stopping after {epochs_no_improve} epochs")
                break

        print(f"Ep {epoch+1:02d} | loss={row['train_loss']:.4f}  "
              f"gnorm={row['train_grad_norm']:.3f}  "
              f"val_macro={val_m['macro_acc']:.4f}  best={best_val_macro:.4f}")

    # --- Final val ---
    model.load_state_dict(torch.load(output_dir / ckpt_name, map_location=device))
    final_val = _evaluate(model, val_loader, device, dataset_info)
    print(f"\nFinal val: macro_acc={final_val['macro_acc']:.4f}  "
          f"micro_acc={final_val['micro_acc']:.4f}")
    print("Per dataset:")
    for name, acc in sorted(final_val["per_dataset"].items()):
        print(f"  {name}: {acc:.4f}")

    # --- Persist ---
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    results = {
        "model_name": args.model_name,
        "prune_layer": args.prune_layer,
        "keep_ratio": args.keep_ratio,
        "train_datasets": registry.train_datasets,
        "holdout_datasets": registry.holdout_datasets,
        "pre_val_macro_acc": round(pre["macro_acc"], 6),
        "best_val_macro_acc": round(best_val_macro, 6),
        "final_val_macro_acc": round(final_val["macro_acc"], 6),
        "final_val_micro_acc": round(final_val["micro_acc"], 6),
        "final_val_per_dataset": {k: round(v, 6)
                                   for k, v in final_val["per_dataset"].items()},
        "args": vars(args),
    }
    save_results(output_dir / f"results_{run_name}.json", results)
    print(f"\nResults saved to: {output_dir}")

    if use_wandb:
        wandb.log({"val/best_macro_acc": best_val_macro,
                   "val/final_macro_acc": final_val["macro_acc"]})
        wandb.finish()


if __name__ == "__main__":
    main()
