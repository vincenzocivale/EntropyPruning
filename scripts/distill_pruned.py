"""Phase 3 (Approach 3): dataset-agnostic feature distillation.

Trains the blocks *after* ``--prune-layer`` (via LoRA) so that the EAF-pruned
backbone reproduces the final hidden states its own frozen, unpruned self would
have produced. The main signal is token-level cosine matching on the kept patch
tokens, with auxiliary CLS cosine and CLS norm-magnitude matching. No labels and
no classification head are involved -- the corpus is the union of several
Thunder datasets. The result is a single backbone checkpoint reusable across
every downstream dataset via linear probing (see
scripts/linear_probe_pruned_eaf.py --backbone-ckpt).

Training reads a per-dataset HDF5 feature cache (built automatically below,
or ahead of time with scripts/build_distill_cache.py) instead of raw images.
The blocks up to and including ``--prune-layer`` are frozen and identical
between the teacher and the student, and the teacher itself never changes --
so for a fixed image their output is constant for the whole run. The cache
stores that output once; every training step then runs only the LoRA blocks
actually being trained, on the already-pruned token sequence. See
src/collection/distill_cache.py for the extraction details.

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

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
from pathlib import Path

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, grad_norm, save_results
from src.models import DistilledPrunedBackbone, ThunderBackboneAdapter, load_forecaster
from src.data import MultiDistillH5Dataset, BlockShuffleH5Dataset
from src.collection import build_distill_cache

DEFAULT_DATASETS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
    "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
    "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds",
]


def _distill_forward(student, seq_prune, teacher_patches):
    """Shared forward pass: student outputs, plus the teacher tokens at the
    indices the student kept -- both the cached prune-layer sequence and the
    teacher targets come straight from the feature cache, so no backbone
    forward pass (frozen prefix or teacher) happens here."""
    student_out = student.forward_from_seq(seq_prune, return_tokens=True)
    kept_idx = student_out["kept_indices"]
    teacher_tokens = teacher_patches.gather(
        1,
        kept_idx.unsqueeze(-1).expand(-1, -1, teacher_patches.shape[-1]),
    )
    return student_out, teacher_tokens


def _distill_losses(student_out, teacher_cls, teacher_tokens, token_weight, cls_weight, mag_weight):
    token_cos = F.cosine_similarity(student_out["tokens"], teacher_tokens, dim=-1)
    cls_cos = F.cosine_similarity(student_out["cls"], teacher_cls, dim=-1)
    student_norm = student_out["cls"].norm(dim=-1)
    teacher_norm = teacher_cls.norm(dim=-1)

    token_loss = (1.0 - token_cos).mean()
    cls_loss = (1.0 - cls_cos).mean()
    mag_loss = F.smooth_l1_loss(student_norm, teacher_norm)
    loss = token_weight * token_loss + cls_weight * cls_loss + mag_weight * mag_loss

    return {
        "loss": loss,
        "token_loss": token_loss,
        "cls_loss": cls_loss,
        "mag_loss": mag_loss,
        "token_cosine_sim": token_cos.mean(),
        "cls_cosine_sim": cls_cos.mean(),
        "cls_mag_rel_error": ((student_norm - teacher_norm).abs() / teacher_norm.clamp_min(1e-6)).mean(),
    }


def _distill_step(student, seq_prune, teacher_cls, teacher_patches, token_weight, cls_weight, mag_weight):
    student_out, teacher_tokens = _distill_forward(student, seq_prune, teacher_patches)
    return _distill_losses(student_out, teacher_cls, teacher_tokens, token_weight, cls_weight, mag_weight)


def _accum(total, batch_m):
    for key, value in batch_m.items():
        total[key] = total.get(key, 0.0) + float(value.detach().item())


def _retrieval_recall_at_k(student_cls, teacher_cls, k):
    """In-batch top-k retrieval consistency: for each sample, the fraction of its
    teacher-space top-k nearest neighbours (cosine, self excluded) also found
    among its student-space top-k. The batch is the retrieval pool, so ``k``
    must be < batch size -- this is a cheap per-step proxy, not a corpus-wide
    retrieval eval."""
    B = student_cls.shape[0]
    k = min(k, B - 1)
    if k < 1:
        return None
    s = F.normalize(student_cls.float(), dim=-1)
    t = F.normalize(teacher_cls.float(), dim=-1)
    sim_s = s @ s.T
    sim_t = t @ t.T
    eye = torch.eye(B, device=student_cls.device, dtype=torch.bool)
    sim_s.masked_fill_(eye, float("-inf"))
    sim_t.masked_fill_(eye, float("-inf"))
    topk_s = sim_s.topk(k, dim=-1).indices
    topk_t = sim_t.topk(k, dim=-1).indices
    match = (topk_s.unsqueeze(-1) == topk_t.unsqueeze(1)).any(-1).float().sum(-1)
    return (match / k).mean()


def _cka_linear(student_feats, teacher_feats):
    """Linear CKA between two (N, D) CLS feature matrices accumulated over a
    full eval pass (centering and the Frobenius norms need the whole split,
    not a single batch)."""
    x = student_feats - student_feats.mean(dim=0, keepdim=True)
    y = teacher_feats - teacher_feats.mean(dim=0, keepdim=True)
    hsic = (x.T @ y).norm() ** 2
    norm_x = (x.T @ x).norm()
    norm_y = (y.T @ y).norm()
    return (hsic / (norm_x * norm_y).clamp_min(1e-12)).item()


@torch.no_grad()
def _evaluate(student, loader, device, token_weight, cls_weight, mag_weight,
              retrieval_k=5, desc="eval"):
    student.eval()
    total, n_batches = {}, 0
    retrieval_total, retrieval_batches = 0.0, 0
    student_cls_cpu, teacher_cls_cpu = [], []
    for seq_prune, teacher_cls, teacher_patches, _ in tqdm(loader, leave=False, desc=desc):
        seq_prune = seq_prune.to(device)
        teacher_cls = teacher_cls.to(device)
        teacher_patches = teacher_patches.to(device)
        with autocast("cuda"):
            student_out, teacher_tokens = _distill_forward(student, seq_prune, teacher_patches)
            batch_m = _distill_losses(student_out, teacher_cls, teacher_tokens,
                                       token_weight, cls_weight, mag_weight)
            retrieval_recall = _retrieval_recall_at_k(student_out["cls"], teacher_cls, retrieval_k)

        _accum(total, {
            "loss": batch_m["loss"],
            "token_cosine_sim": batch_m["token_cosine_sim"],
            "cls_cosine_sim": batch_m["cls_cosine_sim"],
            "cls_mag_rel_error": batch_m["cls_mag_rel_error"],
        })
        if retrieval_recall is not None:
            retrieval_total += float(retrieval_recall.detach().item())
            retrieval_batches += 1
        student_cls_cpu.append(student_out["cls"].float().cpu())
        teacher_cls_cpu.append(teacher_cls.float().cpu())
        n_batches += 1

    metrics = {key: value / n_batches for key, value in total.items()}
    metrics["retrieval_recall_at_k"] = retrieval_total / max(retrieval_batches, 1)
    metrics["cka_linear"] = _cka_linear(torch.cat(student_cls_cpu), torch.cat(teacher_cls_cpu))
    return metrics


def _build_caches(datasets, model_name, base_data_folder, cache_dir, prune_layer,
                   teacher, adapter, transform, device, cache_batch_size, cache_num_workers,
                   max_samples_per_split=None):
    """Build/reuse the per-dataset distillation feature cache for every
    dataset in ``datasets``. Datasets missing a data split are skipped."""
    cache_paths = {}
    for dataset_name in datasets:
        split_path = Path(base_data_folder) / "data_splits" / f"{dataset_name}.json"
        if not split_path.exists():
            print(f"[{dataset_name}] SKIP: missing data split {split_path}")
            continue
        save_path = Path(cache_dir) / f"{dataset_name}_{model_name}_distill_prune{prune_layer}.h5"
        build_distill_cache(
            teacher, adapter, transform, dataset_name, base_data_folder, save_path, device,
            prune_layer=prune_layer, batch_size=cache_batch_size, num_workers=cache_num_workers,
            max_samples_per_split=max_samples_per_split,
        )
        cache_paths[dataset_name] = save_path
    return cache_paths


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
                        help="Used to resolve the default --forecaster-ckpt path and to store/reuse "
                             "the per-dataset distillation feature caches.")
    parser.add_argument("--forecaster-n-heads", type=int, default=4)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--layer-target", type=int, default=None,
                        help="Defaults to last block. Must match the forecaster's training.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--token-weight", type=float, default=1.0,
                        help="Weight for token-level cosine matching on kept tokens.")
    parser.add_argument("--cls-weight", type=float, default=0.5,
                        help="Weight for auxiliary CLS cosine matching.")
    parser.add_argument("--mag-weight", type=float, default=0.1,
                        help="Weight for SmoothL1 matching of CLS feature norms.")
    parser.add_argument("--keep-ratio-min", type=float, default=None,
                        help="If set, sample a training keep ratio uniformly in "
                             "[keep-ratio-min, keep-ratio] each step; validation uses keep-ratio.")
    parser.add_argument("--retrieval-k", type=int, default=5,
                        help="k for the in-batch retrieval-consistency recall@k eval metric "
                             "(must be < --batch-size).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shuffle-block-size", type=int, default=32,
                        help="Rows per contiguous on-disk micro-block for the train loader's "
                             "block-shuffle (must divide --batch-size). Set to 1 to recover plain "
                             "per-row shuffling.")
    parser.add_argument("--cache-batch-size", type=int, default=64,
                        help="Image batch size used only while building the feature cache.")
    parser.add_argument("--cache-num-workers", type=int, default=4,
                        help="DataLoader workers used only while building the feature cache.")
    parser.add_argument("--max-samples-per-split", type=int, default=None,
                        help="Debug cap on samples per split when building the cache.")
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
    if args.keep_ratio_min is not None and not (0 < args.keep_ratio_min <= args.keep_ratio):
        raise ValueError("--keep-ratio-min must satisfy 0 < keep-ratio-min <= keep-ratio")

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

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = _build_caches(
        args.datasets, args.model_name, args.base_data_folder, cache_dir, args.prune_layer,
        teacher, adapter, transform, device,
        args.cache_batch_size, args.cache_num_workers, args.max_samples_per_split,
    )
    if not cache_paths:
        raise RuntimeError("No dataset caches available -- check --base-data-folder / --datasets.")
    used_datasets = list(cache_paths.keys())
    print(f"Corpus ({len(used_datasets)} datasets): {used_datasets}")

    # The teacher and the student's frozen prefix are now fully captured in
    # the cache -- free the teacher's weights before building the student.
    del teacher, raw_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

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

    train_ds = MultiDistillH5Dataset(cache_paths, "train")
    val_ds = MultiDistillH5Dataset(cache_paths, "val")
    test_ds = MultiDistillH5Dataset(cache_paths, "test")
    block_train_ds = BlockShuffleH5Dataset(
        train_ds, batch_size=args.batch_size, micro_block_size=args.shuffle_block_size,
        seed=args.seed, drop_last=True,
    )
    train_loader = DataLoader(
        block_train_ds, batch_size=None, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=False,
    )
    eval_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                   persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, shuffle=False, **eval_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **eval_kw)
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Steps/epoch (approx, block-shuffled): {len(block_train_ds)}")

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
        block_train_ds.set_epoch(epoch)
        total, total_gnorm = {}, 0.

        for seq_prune, teacher_cls, teacher_patches, _ in tqdm(train_loader, leave=False, desc=f"Ep{epoch+1}"):
            seq_prune = seq_prune.to(device)
            teacher_cls = teacher_cls.to(device)
            teacher_patches = teacher_patches.to(device)
            if args.keep_ratio_min is not None:
                student.keep_ratio = (
                    args.keep_ratio_min
                    + torch.rand((), device=device).item() * (args.keep_ratio - args.keep_ratio_min)
                )
            with autocast("cuda"):
                batch_m = _distill_step(
                    student, seq_prune, teacher_cls, teacher_patches,
                    args.token_weight, args.cls_weight, args.mag_weight,
                )
                loss = batch_m["loss"]

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            total_gnorm += grad_norm(student)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            _accum(total, batch_m)

        n_batches = len(train_loader)
        student.keep_ratio = args.keep_ratio
        val_m = _evaluate(student, val_loader, device,
                           args.token_weight, args.cls_weight, args.mag_weight,
                           retrieval_k=args.retrieval_k, desc=f"val ep{epoch+1}")

        row = {
            "epoch": epoch + 1,
            "train_loss": total["loss"] / n_batches,
            "train_token_loss": total["token_loss"] / n_batches,
            "train_cls_loss": total["cls_loss"] / n_batches,
            "train_mag_loss": total["mag_loss"] / n_batches,
            "train_token_cosine_sim": total["token_cosine_sim"] / n_batches,
            "train_cls_cosine_sim": total["cls_cosine_sim"] / n_batches,
            "train_cls_mag_rel_error": total["cls_mag_rel_error"] / n_batches,
            "train_grad_norm": total_gnorm / n_batches,
            "val_loss": val_m["loss"],
            "val_token_cosine_sim": val_m["token_cosine_sim"],
            "val_cls_cosine_sim": val_m["cls_cosine_sim"],
            "val_cls_mag_rel_error": val_m["cls_mag_rel_error"],
            "val_cka_linear": val_m["cka_linear"],
            "val_retrieval_recall_at_k": val_m["retrieval_recall_at_k"],
        }
        history.append(row)
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": row["train_loss"],
                "train/token_loss": row["train_token_loss"],
                "train/cls_loss": row["train_cls_loss"],
                "train/mag_loss": row["train_mag_loss"],
                "train/token_cosine_sim": row["train_token_cosine_sim"],
                "train/cls_cosine_sim": row["train_cls_cosine_sim"],
                "train/cls_mag_rel_error": row["train_cls_mag_rel_error"],
                "train/grad_norm": row["train_grad_norm"],
                "val/loss": val_m["loss"],
                "val/token_cosine_sim": val_m["token_cosine_sim"],
                "val/cls_cosine_sim": val_m["cls_cosine_sim"],
                "val/cls_mag_rel_error": val_m["cls_mag_rel_error"],
                "val/cka_linear": val_m["cka_linear"],
                "val/retrieval_recall_at_k": val_m["retrieval_recall_at_k"],
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
              f"val_loss={val_m['loss']:.4f}  val_tok={val_m['token_cosine_sim']:.4f}  "
              f"val_cls={val_m['cls_cosine_sim']:.4f}  best={best_val_loss:.4f}")

    # --- Reload best checkpoint, merge LoRA into the base weights once, save ---
    student.load_state_dict(torch.load(lora_ckpt, map_location=device))
    student.eval()
    test_m = _evaluate(student, test_loader, device,
                        args.token_weight, args.cls_weight, args.mag_weight,
                        retrieval_k=args.retrieval_k, desc="final test")
    print(f"\n-- Best checkpoint on test set -- loss={test_m['loss']:.4f}  "
          f"token_cos={test_m['token_cosine_sim']:.4f}  cls_cos={test_m['cls_cosine_sim']:.4f}  "
          f"cka={test_m['cka_linear']:.4f}  recall@{args.retrieval_k}={test_m['retrieval_recall_at_k']:.4f}")

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
        "loss_weights": {
            "token": args.token_weight,
            "cls": args.cls_weight,
            "mag": args.mag_weight,
        },
        "retrieval_k": args.retrieval_k,
        "best_val_loss": round(best_val_loss, 6),
        "test_loss": round(test_m["loss"], 6),
        "test_token_cosine_sim": round(test_m["token_cosine_sim"], 6),
        "test_cls_cosine_sim": round(test_m["cls_cosine_sim"], 6),
        "test_cls_mag_rel_error": round(test_m["cls_mag_rel_error"], 6),
        "test_cka_linear": round(test_m["cka_linear"], 6),
        "test_retrieval_recall_at_k": round(test_m["retrieval_recall_at_k"], 6),
        "backbone_ckpt": str(backbone_ckpt),
        "args": vars(args),
    }
    path = save_results(output_dir / f"results_{run_name}.json", results)
    print(f"Results saved to: {path}")

    if use_wandb:
        wandb.log({
            "test/loss": test_m["loss"],
            "test/token_cosine_sim": test_m["token_cosine_sim"],
            "test/cls_cosine_sim": test_m["cls_cosine_sim"],
            "test/cls_mag_rel_error": test_m["cls_mag_rel_error"],
            "test/cka_linear": test_m["cka_linear"],
            "test/retrieval_recall_at_k": test_m["retrieval_recall_at_k"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
