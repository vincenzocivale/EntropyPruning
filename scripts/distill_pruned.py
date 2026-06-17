"""Phase 3 (Approach 3): dataset-agnostic CLS-token distillation.

Trains the blocks *after* ``--prune-layer`` (via LoRA) so that the EAF-pruned
backbone reproduces the CLS token its own frozen, unpruned self would have
produced. No labels and no classification head are involved -- the corpus is
the union of several Thunder datasets, and the supervision signal is purely
"be close to the teacher's CLS embedding". The result is a single backbone
checkpoint reusable across every downstream dataset via linear probing
(see scripts/linear_probe_pruned_eaf.py --backbone-ckpt).

Usage example
-------------
python scripts/distill_pruned.py \\
    --model-name uni \\
    --base-data-folder /raid/DATASETS \\
    --cache-dir /raid/DATASETS/checkpoints/unsupervised \\
    --prune-layer 2 \\
    --keep-ratio 0.1 \\
    --epochs 10 \\
    --wandb-project eaf-distill

Then, to linear-probe the distilled backbone on a specific dataset:
python scripts/linear_probe_pruned_eaf.py \\
    --model-name uni \\
    --base-data-folder /raid/DATASETS \\
    --cache-dir /raid/DATASETS/checkpoints/unsupervised \\
    --eaf-types universal \\
    --keep-ratios 0.1 \\
    --backbone-ckpt checkpoints/unsupervised/uni_distilled/distilled_uni_prune2_keep10.pt \\
    --backbone-tag distilled
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, grad_norm, save_results
from src.models import DistilledPrunedBackbone, ThunderBackboneAdapter, load_forecaster
from src.data import build_multi_dataset_loaders

DEFAULT_DATASETS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
    "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
    "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds",
]


@torch.no_grad()
def _evaluate(student, teacher, loader, device, mse_weight, cosine_weight):
    student.eval()
    total_loss, total_mse, total_cos, n_batches = 0., 0., 0., 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        teacher_cls = teacher.forward_features(imgs)[:, 0]
        student_cls = student(imgs)
        mse = F.mse_loss(student_cls, teacher_cls)
        cos_sim = F.cosine_similarity(student_cls, teacher_cls, dim=-1).mean()
        loss = mse_weight * mse + cosine_weight * (1 - cos_sim)
        total_loss += loss.item()
        total_mse += mse.item()
        total_cos += cos_sim.item()
        n_batches += 1
    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "cosine_sim": total_cos / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: dataset-agnostic CLS distillation")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--forecaster-ckpt", type=str, default=None,
                        help="Default: universal forecaster at "
                             "{cache-dir}/{model}_forecaster/forecaster_{model}_src{prune_layer:02d}"
                             "_attn{layer_target:02d}_universal.pt")
    parser.add_argument("--cache-dir", type=str, default="checkpoints/unsupervised",
                        help="Used to resolve the default --forecaster-ckpt path.")
    parser.add_argument("--forecaster-n-heads", type=int, default=4)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--layer-target", type=int, default=None,
                        help="Defaults to last block. Must match the forecaster's training.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (default: None = skip W&B).")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Epochs without val/loss improvement before stopping (0 = disabled).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Model: {args.model_name}")

    raw_teacher, transform, _ = get_model_from_name(args.model_name, str(device))
    raw_student, _, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_student)
    layer_target = args.layer_target if args.layer_target is not None else adapter.n_blocks - 1
    print(f"embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}  "
          f"prune_layer={args.prune_layer}  keep_ratio={args.keep_ratio}")
    assert args.prune_layer < adapter.n_blocks - 1, \
        f"--prune-layer {args.prune_layer} leaves no blocks to distill (n_blocks={adapter.n_blocks})"

    teacher = raw_teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    forecaster_ckpt = args.forecaster_ckpt or str(
        Path(args.cache_dir) / f"{args.model_name}_forecaster" /
        f"forecaster_{args.model_name}_src{args.prune_layer:02d}_attn{layer_target:02d}_universal.pt")
    forecaster = load_forecaster(forecaster_ckpt, device, args.forecaster_n_heads)
    print(f"Forecaster loaded: {forecaster_ckpt}")

    student = DistilledPrunedBackbone(
        backbone=raw_student, adapter=adapter, forecaster=forecaster,
        prune_layer=args.prune_layer, keep_ratio=args.keep_ratio,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    ).to(device)

    train_loader, val_loader, used_datasets = build_multi_dataset_loaders(
        args.datasets, args.base_data_folder, transform, args.batch_size, args.num_workers)
    print(f"Corpus ({len(used_datasets)} datasets): {used_datasets}")
    print(f"Samples: train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.cache_dir) / f"{args.model_name}_distilled"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.model_name}_prune{args.prune_layer}_keep{int(args.keep_ratio * 100)}"
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project, name=run_name, job_type="phase3_distill",
            group=f"distill/{args.model_name}", config=vars(args),
            tags=[args.model_name, f"prune_layer_{args.prune_layer}",
                  f"keep_{int(args.keep_ratio * 100)}pct", "phase3", "distillation"],
        )

    pre = _evaluate(student, teacher, val_loader, device, args.mse_weight, args.cosine_weight)
    print(f"\nPre-training val: loss={pre['loss']:.4f}  cosine_sim={pre['cosine_sim']:.4f}")

    trainable_params = [p for p in student.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    scaler = GradScaler("cuda")

    lora_ckpt = output_dir / f"lora_{run_name}.pt"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        student.train()
        total_loss, total_gnorm = 0., 0.

        for imgs, _ in tqdm(train_loader, leave=False, desc=f"Ep{epoch+1}"):
            imgs = imgs.to(device)
            with torch.no_grad():
                teacher_cls = teacher.forward_features(imgs)[:, 0]

            with autocast("cuda"):
                student_cls = student(imgs)
                mse = F.mse_loss(student_cls, teacher_cls)
                cos_sim = F.cosine_similarity(student_cls, teacher_cls, dim=-1).mean()
                loss = args.mse_weight * mse + args.cosine_weight * (1 - cos_sim)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            total_gnorm += grad_norm(student)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()

        n_batches = len(train_loader)
        val_m = _evaluate(student, teacher, val_loader, device, args.mse_weight, args.cosine_weight)

        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / n_batches,
            "train_grad_norm": total_gnorm / n_batches,
            "val_loss": val_m["loss"],
            "val_mse": val_m["mse"],
            "val_cosine_sim": val_m["cosine_sim"],
        }
        history.append(row)
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": row["train_loss"],
                "train/grad_norm": row["train_grad_norm"],
                "val/loss": val_m["loss"],
                "val/mse": val_m["mse"],
                "val/cosine_sim": val_m["cosine_sim"],
            })

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            epochs_without_improvement = 0
            torch.save(student.state_dict(), lora_ckpt)
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping: no improvement for {epochs_without_improvement} epochs")
                break

        print(f"Ep {epoch+1:02d} | loss={row['train_loss']:.4f}  gnorm={row['train_grad_norm']:.3f}  "
              f"val_loss={val_m['loss']:.4f}  val_cos={val_m['cosine_sim']:.4f}  best={best_val_loss:.4f}")

    # --- Reload best checkpoint, merge LoRA into the base weights once, save ---
    student.load_state_dict(torch.load(lora_ckpt, map_location=device))
    student.eval()
    final_val = _evaluate(student, teacher, val_loader, device, args.mse_weight, args.cosine_weight)
    print(f"\n-- Best checkpoint -- val_loss={final_val['loss']:.4f}  "
          f"val_cosine_sim={final_val['cosine_sim']:.4f}")

    merged_backbone = student.backbone.merge_and_unload()
    backbone_ckpt = output_dir / f"distilled_{run_name}.pt"
    torch.save(merged_backbone.state_dict(), backbone_ckpt)
    print(f"Distilled backbone saved to: {backbone_ckpt}")

    results = {
        "model_name": args.model_name,
        "datasets": used_datasets,
        "prune_layer": args.prune_layer,
        "keep_ratio": args.keep_ratio,
        "forecaster_ckpt": forecaster_ckpt,
        "pre_val_loss": round(pre["loss"], 6),
        "pre_val_cosine_sim": round(pre["cosine_sim"], 6),
        "best_val_loss": round(best_val_loss, 6),
        "final_val_cosine_sim": round(final_val["cosine_sim"], 6),
        "backbone_ckpt": str(backbone_ckpt),
        "args": vars(args),
    }
    path = save_results(output_dir / f"results_{run_name}.json", results)
    print(f"Results saved to: {path}")

    if use_wandb:
        wandb.log({
            "final/val_loss": final_val["loss"],
            "final/val_cosine_sim": final_val["cosine_sim"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
