"""Ablation study: Evaluate pruning by varying source and target layers."""

import os
# Disable HDF5 file locking to avoid [Errno 11] on some filesystems
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import h5py
import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import set_seed, get_device
from src.models import (UNILoRAClassifier, AttentionForecaster,
                        UNILoRAWithForecasterPruning)
from src.data.loaders import build_loaders
from src.evaluation import evaluate, benchmark_model
from scripts.train_forecaster import train_forecaster, collect_and_save_dataset


def fine_tune_and_eval(args, layer_source, layer_target, forecaster_ckpt,
                       n_classes, train_loader, val_loader, test_loader,
                       device, output_dir):
    """Fine-tune the pruned model for a given source/target pair."""

    # Load forecaster (must match training config: hidden=256, n_heads=4, n_layers=2, dropout=0.2)
    forecaster = AttentionForecaster(hidden=256, n_heads=4, n_layers=2, dropout=0.2).to(device)
    forecaster.load_state_dict(
        torch.load(forecaster_ckpt, map_location=device))
    forecaster.eval()
    for p in forecaster.parameters():
        p.requires_grad_(False)

    # Build pruned model
    model = UNILoRAWithForecasterPruning(
        n_classes=n_classes, forecaster=forecaster,
        prune_layer=layer_source, keep_ratio=args.keep_ratio,
    ).to(device)

    # Load base classifier (pre-trained LoRA)
    # Only load backbone and head keys — forecaster keys are absent from classifier ckpt
    # and strict=False would silently reset forecaster weights to random init.
    classifier_ckpt = args.classifier_ckpt or str(
        Path("/raid/DATASETS/checkpoints-Attention-Pruning/") / Path(args.data_dir).name /
        "uni_finetuned" / "best_model.pt")
    ckpt = torch.load(classifier_ckpt, map_location=device)
    model_state = model.state_dict()
    # Only update keys present in the classifier checkpoint (backbone + head)
    filtered = {k: v for k, v in ckpt.items() if k in model_state}
    model_state.update(filtered)
    missing = [k for k in model_state if k not in ckpt and not k.startswith("forecaster")]
    print(f"  Loaded {len(filtered)} keys from classifier ckpt | "
          f"Non-forecaster missing: {len(missing)}")
    model.load_state_dict(model_state, strict=True)

    # Simple training loop (adapted from finetune_pruned.py)
    backbone_params = [p for _, p in model.backbone.named_parameters()
                       if p.requires_grad]
    head_params = list(model.head.parameters())

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_backbone, args.lr_head],
        total_steps=total_steps, pct_start=0.1)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    run_name = f"src{layer_source:02d}_tgt{layer_target:02d}_keep{int(args.keep_ratio * 100)}"
    wandb.init(project=args.wandb_project, name=f"ft_{run_name}",
               config=vars(args), tags=["ablation", "finetuning"],
               reinit=True)

    scaler = torch.amp.GradScaler("cuda")
    best_val_f1 = 0.
    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

        val_metrics = evaluate(model, val_loader, device, args.far_threshold)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), output_dir / f"best_{run_name}.pt")

        wandb.log({"epoch": epoch+1, "val/f1": val_metrics["f1_macro"]})

    # Test eval
    model.load_state_dict(
        torch.load(output_dir / f"best_{run_name}.pt", map_location=device))
    model.eval()

    test_metrics = evaluate(model, test_loader, device, args.far_threshold)
    test_bench = benchmark_model(model, test_loader, device, label=run_name)

    wandb.log({
        "test/f1_macro": test_metrics["f1_macro"],
        "test/acc": test_metrics["acc"],
        "test/ms_per_img": test_bench["ms_per_img"],
        "test/gflops": test_bench["gflops"],
    })

    print(f"\n  [{run_name}] Test results:")
    print(f"    F1 macro:  {test_metrics['f1_macro']:.4f}")
    print(f"    Accuracy:  {test_metrics['acc']:.4f}")
    print(f"    ms/img:    {test_bench['ms_per_img']:.2f}")
    print(f"    GFLOPs:    {test_bench['gflops']:.2f}")

    wandb.finish()

    return {
        "layer_source": layer_source,
        "layer_target": layer_target,
        "f1_macro": test_metrics["f1_macro"],
        "acc": test_metrics["acc"],
        "tar_at_far": test_metrics["tar_at_far"],
        "ms_per_img": test_bench["ms_per_img"],
        "gflops": test_bench["gflops"],
    }


def main():
    parser = argparse.ArgumentParser(description="Layer Ablation Study")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--layers-source", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--layers-target", type=int, nargs="+", default=[23, 22, 21])
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--far-threshold", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="/raid/DATASETS/data_cache")
    parser.add_argument("--wandb-project", type=str, default="layer-ablation")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    dataset_name = Path(args.data_dir).name
    src_tag = "_".join(str(l) for l in sorted(args.layers_source))
    tgt_tag = "_".join(str(l) for l in sorted(args.layers_target))
    base_ckpt = Path("/raid/DATASETS/checkpoints-Attention-Pruning/") / dataset_name
    output_dir = base_ckpt / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / f"layer_ablation_results_src{src_tag}_tgt{tgt_tag}.csv"

    train_loader, val_loader, test_loader, class_names, n_classes = \
        build_loaders(args.data_dir, args.img_size, args.batch_size,
                      args.num_workers, drop_last_train=True)

    # 1. Feature extraction check
    # Include layers in filename to avoid conflicts when running multiple processes in parallel
    dataset_cache = Path(args.cache_dir) / f"{dataset_name}_ablation_src{src_tag}_tgt{tgt_tag}.h5"
    
    should_extract = not dataset_cache.exists()
    if not should_extract:
        try:
            # Verify all layers are present
            with h5py.File(dataset_cache, 'r') as f:
                if "train" not in f:
                    should_extract = True
                else:
                    existing_keys = f["train"].keys()
                    for ls in args.layers_source:
                        if f"emb_layer{ls}" not in existing_keys:
                            print(f"Missing source layer {ls} in cache.")
                            should_extract = True
                            break
                    if not should_extract:
                        for lt in args.layers_target:
                            if f"attn_layer{lt}" not in existing_keys:
                                print(f"Missing target layer {lt} in cache.")
                                should_extract = True
                                break
        except Exception as e:
            print(f"Error reading cache (possibly corrupted): {e}")
            should_extract = True

    if should_extract:
        if dataset_cache.exists():
            print("Cache is incomplete or corrupted. Re-extracting...")
            dataset_cache.unlink()
        
        dataset_cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting features for source {args.layers_source} and target {args.layers_target}...")
        classifier_ckpt = args.classifier_ckpt or str(
            base_ckpt / "uni_finetuned" / "best_model.pt")
        model = UNILoRAClassifier(n_classes).to(device)
        model.load_state_dict(torch.load(classifier_ckpt, map_location=device), strict=False)
        model.eval()
        try:
            collect_and_save_dataset(
                model, {"train": train_loader, "val": val_loader, "test": test_loader},
                device, layers_source=args.layers_source, layers_target=args.layers_target,
                save_path=dataset_cache,
            )
        except Exception as e:
            print(f"Error during extraction: {e}")
            if dataset_cache.exists(): dataset_cache.unlink()
            raise e
    else:
        print(f"Using existing complete cache: {dataset_cache}")

    for lt in args.layers_target:
        for ls in args.layers_source:
            print(f"\n>>> Starting ablation: src={ls}, tgt={lt}")

            # 2. Train forecaster
            forecaster_cfg = {
                "dataset_name": dataset_name,
                "dataset_cache": dataset_cache,
                "forecaster_dir": base_ckpt / "forecaster",
                "hidden": 256, "n_heads": 4, "n_layers": 2, "dropout": 0.2,
                "epochs": 20, "lr": 1e-4, "weight_decay": 0.05,
                "wandb_project": args.wandb_project + "-forecaster"
            }
            forecaster_cfg["forecaster_dir"].mkdir(parents=True, exist_ok=True)

            f_res = train_forecaster(ls, lt, forecaster_cfg, device)
            f_ckpt = forecaster_cfg["forecaster_dir"] / f"forecaster_src{ls:02d}_tgt{lt:02d}.pt"

            # 3. Fine-tune pruned model
            res = fine_tune_and_eval(
                args, ls, lt, f_ckpt, n_classes,
                train_loader, val_loader, test_loader,
                device, output_dir
            )

            # Combine results
            res.update({
                "forecaster_rho": f_res["test_rho_forecaster"],
                "forecaster_kl": f_res["best_val_kl"],
            })

            # Append to CSV immediately (create if not exists, append if exists)
            row_df = pd.DataFrame([res])
            if results_csv.exists():
                row_df.to_csv(results_csv, mode='a', header=False, index=False)
            else:
                row_df.to_csv(results_csv, index=False)
            print(f"Results appended to {results_csv}")

    print("\nAblation study complete.")
    print(pd.read_csv(results_csv))


if __name__ == "__main__":
    main()
