"""Phase 1 (multi-dataset): Train a multi-head classifier on Thunder datasets.

Trains a shared backbone + one classification head per Thunder dataset.
The N smallest datasets (by train sample count) are held out and saved to
holdout_plan.json — Phase 2 and 3 scripts reload this file to use the same split.
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
from transformers import get_cosine_schedule_with_warmup
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, grad_norm, save_results
from src.models import ThunderBackboneAdapter, AttentionForecaster, STRATEGIES
from src.models.multi_head_classifier import MultiHeadThunderClassifier
from src.models.online_tile_eaf import PrunedLoRAEncoder
from src.data.thunder_multi import ThunderDatasetRegistry, build_multi_thunder_train_loaders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multi_head_loss(criterion, output: dict, labels: torch.Tensor,
                     dataset_indices: torch.Tensor) -> torch.Tensor:
    """Compute mean cross-entropy loss across all dataset subsets in the batch."""
    losses = []
    for dataset_idx, logits in output.items():
        mask = dataset_indices == dataset_idx
        losses.append(criterion(logits, labels[mask]))
    return torch.stack(losses).mean()


def _run_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, total_correct, total_samples, total_gnorm = 0., 0, 0, 0.

    for imgs, labels, dataset_indices in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        dataset_indices = dataset_indices.to(device)

        with autocast("cuda"):
            output = model(imgs, dataset_indices)
            loss = _multi_head_loss(criterion, output, labels, dataset_indices)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = grad_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        n = len(labels)
        correct = sum(
            (logits.argmax(1) == labels[dataset_indices == k]).sum().item()
            for k, logits in output.items()
        )
        total_loss += loss.item() * n
        total_correct += correct
        total_samples += n
        total_gnorm += gnorm

    n_batches = len(loader)
    return total_loss / total_samples, total_correct / total_samples, total_gnorm / n_batches


@torch.no_grad()
def _evaluate(model, loader, device, dataset_info: dict) -> dict:
    """Returns macro_acc, per_dataset_acc, and total micro_acc."""
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
        description="Phase 1 (multi-dataset): Train multi-head classifier on Thunder")

    # Thunder model / data
    parser.add_argument("--model-name", type=str, required=True,
                        help="Thunder model name (e.g. uni, hoptimus0)")
    parser.add_argument("--base-data-folder", type=str, required=True,
                        help="Root containing data_splits/ and datasets/")
    # Holdout
    parser.add_argument("--n-holdout", type=int, default=3,
                        help="Number of smallest datasets to hold out (default 3)")
    parser.add_argument("--holdout-datasets", type=str, nargs="+", default=None,
                        help="Explicit holdout dataset names (overrides --n-holdout)")
    # Adaptation
    parser.add_argument("--adaptation", type=str, default="lora", choices=STRATEGIES)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    # Tile-EAF pruned-encoder linear probing (Stage 3 evaluation). Omit
    # --pruned-adapter-ckpt to train a plain (non-pruned) baseline classifier as
    # before -- everything below only applies once it is given.
    parser.add_argument(
        "--pruned-adapter-ckpt", default=None,
        help=(
            "Stage-2 checkpoint from finetune_wsi_tile_encoder_pruned_online.py "
            "(best_*_adapter.pt). When given, --model-name is wrapped in a frozen "
            "forecaster-pruned PrunedLoRAEncoder and --adaptation is forced to "
            "'linear_probing' -- this evaluates the pruned tile encoder's linear-probe "
            "quality, not a fresh adaptation of it."
        ),
    )
    parser.add_argument(
        "--forecaster-ckpt", default=None,
        help="Overrides the forecaster path recorded in --pruned-adapter-ckpt's checkpoint",
    )
    parser.add_argument(
        "--prune-layer", type=int, default=None,
        help="Overrides the prune_layer recorded in --pruned-adapter-ckpt's checkpoint",
    )
    parser.add_argument(
        "--keep-ratio", type=float, default=None,
        help="Overrides the keep_ratio recorded in --pruned-adapter-ckpt's checkpoint",
    )
    parser.add_argument("--pruned-lora-r", type=int, default=8)
    parser.add_argument("--pruned-lora-alpha", type=int, default=32)
    parser.add_argument("--pruned-lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--forecaster-hidden", type=int, default=256,
        help="Must match --hidden used at Stage-1 forecaster training time",
    )
    parser.add_argument("--forecaster-n-heads", type=int, default=4)
    parser.add_argument("--forecaster-n-layers", type=int, default=2)
    parser.add_argument("--forecaster-dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    # Output / W&B
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # --- Dataset registry & holdout plan ---
    registry = ThunderDatasetRegistry(
        args.base_data_folder,
        n_holdout=args.n_holdout,
        holdout_datasets=args.holdout_datasets,
    )
    print(f"\nTrain datasets ({len(registry.train_datasets)}): {registry.train_datasets}")
    print(f"Holdout datasets ({len(registry.holdout_datasets)}): {registry.holdout_datasets}")

    # --- Backbone ---
    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    print(f"\nBackbone: {args.model_name}  "
          f"embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}  "
          f"n_patches={adapter.n_patches}")

    # --- Optional: wrap in a frozen forecaster-pruned encoder (Stage 3 eval) ---
    # backbone_for_model is what MultiHeadThunderClassifier actually adapts;
    # `adapter` stays the same object either way (PrunedLoRAEncoder is built from it).
    backbone_for_model = raw_backbone
    pruning_info: dict | None = None
    if args.pruned_adapter_ckpt:
        pruned_payload = torch.load(args.pruned_adapter_ckpt, map_location="cpu")
        pruned_config = pruned_payload.get("config", {})
        forecaster_ckpt = args.forecaster_ckpt or pruned_payload.get("forecaster_checkpoint")
        if not forecaster_ckpt:
            raise ValueError(
                "--forecaster-ckpt was not given and --pruned-adapter-ckpt's checkpoint "
                "does not record a 'forecaster_checkpoint' path"
            )
        prune_layer = args.prune_layer if args.prune_layer is not None else pruned_config.get("prune_layer")
        keep_ratio = args.keep_ratio if args.keep_ratio is not None else pruned_config.get("keep_ratio")
        if prune_layer is None or keep_ratio is None:
            raise ValueError(
                "--prune-layer/--keep-ratio were not given and could not be resolved "
                "from --pruned-adapter-ckpt's checkpoint config"
            )
        forecaster = AttentionForecaster(
            embed_dim=adapter.embed_dim,
            hidden=args.forecaster_hidden,
            n_heads=args.forecaster_n_heads,
            n_layers=args.forecaster_n_layers,
            dropout=args.forecaster_dropout,
        )
        forecaster_payload = torch.load(forecaster_ckpt, map_location="cpu")
        forecaster.load_state_dict(forecaster_payload["model"], strict=True)
        pruned_encoder = PrunedLoRAEncoder(
            raw_backbone, adapter, forecaster,
            prune_layer=int(prune_layer), keep_ratio=float(keep_ratio),
            lora_r=args.pruned_lora_r, lora_alpha=args.pruned_lora_alpha,
            lora_dropout=args.pruned_lora_dropout,
        )
        pruned_encoder.load_trainable_state_dict(pruned_payload["trainable_state_dict"])
        pruned_encoder.eval()
        backbone_for_model = pruned_encoder
        pruning_info = {
            "pruned_adapter_ckpt": str(args.pruned_adapter_ckpt),
            "forecaster_ckpt": str(forecaster_ckpt),
            "prune_layer": int(prune_layer),
            "keep_ratio": float(keep_ratio),
        }
        if args.adaptation != "linear_probing":
            print(
                f"[eval] --pruned-adapter-ckpt given: forcing --adaptation "
                f"linear_probing (was {args.adaptation!r}) -- the pruned encoder is "
                "frozen/already-distilled, not something to re-adapt per dataset"
            )
            args.adaptation = "linear_probing"
        print(
            f"[eval] Frozen forecaster-pruned encoder: prune_layer={prune_layer} "
            f"keep_ratio={keep_ratio} (from {args.pruned_adapter_ckpt})"
        )

    # --- Data ---
    train_loader, val_loader, dataset_info = build_multi_thunder_train_loaders(
        registry.train_datasets,
        args.base_data_folder,
        transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last_train=True,
    )
    print(f"\nDataset info:")
    for idx, info in dataset_info.items():
        count = registry.sample_counts.get(info["name"], "?")
        print(f"  [{idx}] {info['name']}: {info['n_classes']} classes, {count} train samples")

    # --- Output dir ---
    default_dir_name = f"{args.model_name}_{args.adaptation}"
    if pruning_info is not None:
        keep_pct = int(round(pruning_info["keep_ratio"] * 100))
        default_dir_name += f"_pruned_src{pruning_info['prune_layer']:02d}_keep{keep_pct}pct"
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(f"checkpoints/multi_thunder/{default_dir_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save holdout plan (before training — Phases 2/3 need it) ---
    registry.save_plan(str(output_dir / "holdout_plan.json"))

    # --- Model ---
    kwargs: dict = {"dropout": args.dropout}
    if args.adaptation == "lora":
        kwargs.update(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    model = MultiHeadThunderClassifier(
        backbone_for_model, adapter, dataset_info, args.adaptation, **kwargs
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")

    # --- W&B ---
    run_name = args.run_name or f"{args.model_name}_multi_{default_dir_name[len(args.model_name) + 1:]}"
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project, name=run_name, job_type="phase1_multi",
            config={
                **vars(args),
                "train_datasets": registry.train_datasets,
                "holdout_datasets": registry.holdout_datasets,
                "n_trainable_params": n_trainable,
                "n_total_params": n_total,
                "pruning": pruning_info,
            },
            tags=[args.model_name, args.adaptation, "phase1", "multi_dataset"]
            + (["pruned_tile_eaf"] if pruning_info is not None else ["baseline"]),
        )

    # --- Optimizer ---
    backbone_params = model.trainable_backbone_params
    param_groups = [{"params": model.heads.parameters(), "lr": args.lr_head}]
    if backbone_params:
        param_groups.insert(0, {"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler("cuda")

    # --- Training loop ---
    best_macro_acc, best_epoch = -1., 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, gnorm = _run_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_metrics = _evaluate(model, val_loader, device, dataset_info)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_grad_norm": gnorm,
            "val_macro_acc": val_metrics["macro_acc"],
            "val_micro_acc": val_metrics["micro_acc"],
            "val_per_dataset": val_metrics["per_dataset"],
        }
        history.append(row)

        print(f"Epoch {epoch:02d}/{args.epochs}  loss={tr_loss:.4f}  "
              f"tr_acc={tr_acc:.4f}  gnorm={gnorm:.3f}  |  "
              f"val_macro={val_metrics['macro_acc']:.4f}  "
              f"val_micro={val_metrics['micro_acc']:.4f}")

        if use_wandb:
            log = {
                "epoch": epoch,
                "train/loss": tr_loss, "train/acc": tr_acc,
                "train/grad_norm": gnorm,
                "val/macro_acc": val_metrics["macro_acc"],
                "val/micro_acc": val_metrics["micro_acc"],
            }
            log.update({f"val/{n}_acc": a
                        for n, a in val_metrics["per_dataset"].items()})
            wandb.log(log)

        if val_metrics["macro_acc"] > best_macro_acc:
            best_macro_acc, best_epoch = val_metrics["macro_acc"], epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  → saved (val_macro_acc={best_macro_acc:.4f})")
        else:
            epochs_no_improve += 1
            if (args.early_stopping_patience > 0
                    and epochs_no_improve >= args.early_stopping_patience):
                print(f"Early stopping after {epochs_no_improve} epochs without improvement")
                break

    print(f"\nBest val_macro_acc={best_macro_acc:.4f} @ epoch {best_epoch}")

    # --- Final eval on val with best checkpoint ---
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    final_val = _evaluate(model, val_loader, device, dataset_info)
    print("\nFinal val (best checkpoint):")
    for name, acc in sorted(final_val["per_dataset"].items()):
        count = registry.sample_counts.get(name, "?")
        print(f"  {name}: {acc:.4f}  (train_samples={count})")

    # --- Persist ---
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    results = {
        "model_name": args.model_name,
        "adaptation": args.adaptation,
        "pruning": pruning_info,
        "train_datasets": registry.train_datasets,
        "holdout_datasets": registry.holdout_datasets,
        "best_epoch": best_epoch,
        "best_val_macro_acc": round(best_macro_acc, 6),
        "final_val_macro_acc": round(final_val["macro_acc"], 6),
        "final_val_micro_acc": round(final_val["micro_acc"], 6),
        "final_val_per_dataset": {k: round(v, 6) for k, v in final_val["per_dataset"].items()},
        "n_trainable_params": n_trainable,
        "args": vars(args),
    }
    save_results(output_dir / "results.json", results)
    print(f"\nResults saved to: {output_dir}")

    if use_wandb:
        wandb.log({
            "val/best_macro_acc": best_macro_acc, "val/best_epoch": best_epoch,
            "val/final_macro_acc": final_val["macro_acc"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
