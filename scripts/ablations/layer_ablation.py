"""Ablation study: Evaluate pruning performance across different source/target layer pairs."""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn as nn
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device
from src.models import (AttentionForecaster, GenericLoRAClassifier,
                        GenericLoRAWithForecasterPruning, ThunderBackboneAdapter)
from src.collection import collect_and_save_dataset
from src.data.thunder_loaders import build_thunder_loaders
from src.evaluation import evaluate, benchmark_model
from scripts.train_forecaster import train_forecaster


def fine_tune_and_eval(args, adapter, layer_source, layer_target, forecaster_ckpt,
                       raw_backbone, n_classes, train_loader, val_loader, test_loader,
                       device, output_dir):
    """Fine-tune the pruned model for a given source/target layer pair."""
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=256, n_heads=4, n_layers=2, dropout=0.2,
    ).to(device)
    forecaster.load_state_dict(torch.load(forecaster_ckpt, map_location=device))
    forecaster.eval()
    for p in forecaster.parameters():
        p.requires_grad_(False)

    model = GenericLoRAWithForecasterPruning(
        backbone=raw_backbone, adapter=adapter, n_classes=n_classes,
        forecaster=forecaster, prune_layer=layer_source, keep_ratio=args.keep_ratio,
    ).to(device)

    classifier_ckpt = args.classifier_ckpt or str(
        Path("checkpoints") / args.dataset_name /
        f"{args.model_name}_finetuned" / "best_model.pt")
    model.load_state_dict(
        torch.load(classifier_ckpt, map_location=device), strict=False)

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

    run_name = f"src{layer_source:02d}_tgt{layer_target:02d}_keep{int(args.keep_ratio*100)}"
    wandb.init(project=args.wandb_project, name=f"ft_{run_name}",
               config=vars(args), tags=["ablation", "finetuning"], reinit=True)

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
        val_m = evaluate(model, val_loader, device, args.far_threshold)
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            torch.save(model.state_dict(), output_dir / f"best_{run_name}.pt")
        wandb.log({"epoch": epoch+1, "val/f1": val_m["f1_macro"]})

    model.load_state_dict(torch.load(output_dir / f"best_{run_name}.pt", map_location=device))
    model.eval()
    test_m = evaluate(model, test_loader, device, args.far_threshold)
    test_b = benchmark_model(model, test_loader, device, label=run_name)
    wandb.log({"test/f1_macro": test_m["f1_macro"], "test/acc": test_m["acc"],
               "test/ms_per_img": test_b["ms_per_img"], "test/gflops": test_b["gflops"]})
    print(f"\n  [{run_name}]  f1={test_m['f1_macro']:.4f}  acc={test_m['acc']:.4f}  "
          f"ms/img={test_b['ms_per_img']:.2f}  GFLOPs={test_b['gflops']:.2f}")
    wandb.finish()
    return {"layer_source": layer_source, "layer_target": layer_target,
            "f1_macro": test_m["f1_macro"], "acc": test_m["acc"],
            "tar_at_far": test_m["tar_at_far"],
            "ms_per_img": test_b["ms_per_img"], "gflops": test_b["gflops"]}


def main():
    parser = argparse.ArgumentParser(description="Layer Ablation Study")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Thunder model name (e.g. uni, hoptimus0)")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Thunder dataset name (e.g. crc, break_his)")
    parser.add_argument("--base-data-folder", type=str, required=True,
                        help="Path to Thunder base data folder")
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--layers-source", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--layers-target", type=int, nargs="+", default=None,
                        help="Defaults to [n_blocks-1].")
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--far-threshold", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="layer-ablation")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    print(f"Backbone: embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}  "
          f"n_patches={adapter.n_patches}  prefix={adapter.num_prefix_tokens}")

    layers_target = args.layers_target if args.layers_target else [adapter.n_blocks - 1]
    for ls in args.layers_source:
        assert ls < adapter.n_blocks, f"--layers-source {ls} >= n_blocks {adapter.n_blocks}"
    for lt in layers_target:
        assert lt < adapter.n_blocks, f"--layers-target {lt} >= n_blocks {adapter.n_blocks}"

    base_ckpt = Path("checkpoints")
    cache_dir = Path(args.cache_dir) if args.cache_dir else base_ckpt / args.dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_tag = "_".join(str(l) for l in sorted(args.layers_source))
    tgt_tag = "_".join(str(l) for l in sorted(layers_target))
    output_dir = base_ckpt / args.dataset_name / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / f"ablation_{args.model_name}_src{src_tag}_tgt{tgt_tag}.csv"

    train_loader, val_loader, test_loader, _, n_classes = build_thunder_loaders(
        args.dataset_name, args.base_data_folder, transform,
        args.batch_size, args.num_workers, drop_last_train=True,
    )

    # Feature cache (includes all source and target layers)
    dataset_cache = (
        cache_dir / f"{args.dataset_name}_{args.model_name}_ablation_src{src_tag}_tgt{tgt_tag}.h5"
    )
    should_extract = not dataset_cache.exists()
    if not should_extract:
        try:
            with h5py.File(dataset_cache, 'r') as f:
                if "train" not in f:
                    should_extract = True
                else:
                    keys = set(f["train"].keys())
                    for ls in args.layers_source:
                        if f"emb_layer{ls}" not in keys:
                            should_extract = True; break
                    for lt in layers_target:
                        if f"attn_layer{lt}" not in keys:
                            should_extract = True; break
        except Exception as e:
            print(f"Cache unreadable ({e}), re-extracting.")
            should_extract = True

    if should_extract:
        if dataset_cache.exists():
            dataset_cache.unlink()
        dataset_cache.parent.mkdir(parents=True, exist_ok=True)
        classifier_ckpt = args.classifier_ckpt or str(
            base_ckpt / args.dataset_name / f"{args.model_name}_finetuned" / "best_model.pt")
        extractor = GenericLoRAClassifier(raw_backbone, adapter, n_classes).to(device)
        extractor.load_state_dict(
            torch.load(classifier_ckpt, map_location=device), strict=False)
        extractor.eval()
        for p in extractor.parameters():
            p.requires_grad_(False)
        collect_and_save_dataset(
            extractor,
            {"train": train_loader, "val": val_loader, "test": test_loader},
            device, layers_source=args.layers_source, layers_target=layers_target,
            save_path=dataset_cache,
        )
    else:
        print(f"Using cache: {dataset_cache}")

    for lt in layers_target:
        for ls in args.layers_source:
            print(f"\n>>> src={ls}, tgt={lt}")

            forecaster_dir = base_ckpt / args.dataset_name / f"{args.model_name}_forecaster"
            forecaster_dir.mkdir(parents=True, exist_ok=True)

            forecaster_cfg = dict(
                model_name=args.model_name, dataset_name=args.dataset_name,
                embed_dim=adapter.embed_dim, dataset_cache=dataset_cache,
                forecaster_dir=forecaster_dir,
                hidden=256, n_heads=4, n_layers=2, dropout=0.2,
                epochs=20, lr=1e-4, weight_decay=0.05,
                wandb_project=args.wandb_project + "-forecaster",
            )
            f_res = train_forecaster(ls, lt, forecaster_cfg, device)
            f_ckpt = forecaster_dir / f"forecaster_src{ls:02d}_tgt{lt:02d}.pt"

            res = fine_tune_and_eval(
                args, adapter, ls, lt, f_ckpt, raw_backbone,
                n_classes, train_loader, val_loader, test_loader,
                device, output_dir,
            )
            res.update(forecaster_rho=f_res["test_rho_forecaster"],
                       forecaster_kl=f_res["best_val_kl"])

            row_df = pd.DataFrame([res])
            if results_csv.exists():
                row_df.to_csv(results_csv, mode='a', header=False, index=False)
            else:
                row_df.to_csv(results_csv, index=False)
            print(f"Appended to {results_csv}")

    print("\nAblation study complete.")
    print(pd.read_csv(results_csv))


if __name__ == "__main__":
    main()
