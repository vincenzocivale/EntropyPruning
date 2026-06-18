"""Inference-only evaluation of pruned checkpoints.

Reloads trained checkpoints and evaluates on the test split without retraining.
Requires the same --model-name used during the original training run.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import get_device, set_seed
from src.models import (AttentionForecaster, GenericLoRAWithForecasterPruning,
                        LoRAWithEViTPruning, ThunderBackboneAdapter,
                        parse_evit_drop_locs)
from src.data.thunder_loaders import build_thunder_loaders
from src.evaluation import evaluate, benchmark_model


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def evaluate_one(
    model_name: str,
    dataset_name: str,
    base_data_folder: str,
    ckpt_root: Path,
    pruning_method: str,
    keep_ratio: float,
    prune_layer: int,
    evit_drop_loc: str,
    evit_base_keep_rate: float | None,
    evit_fuse_token: bool,
    batch_size: int,
    num_workers: int,
    far_threshold: float,
    seed: int,
    hidden: int,
    n_heads: int,
    n_layers: int,
    forecaster_dropout: float,
):
    keep_pct = int(round(keep_ratio * 100))
    pruned_dir = ckpt_root / dataset_name / f"{model_name}_pruned"

    if pruning_method == "eaf":
        run_name = f"{model_name}_prune{prune_layer}_keep{keep_pct}"
        current_run_name = f"{model_name}_{dataset_name}_eaf_prune{prune_layer}_keep{keep_pct}"
        pruned_ckpt = first_existing([
            pruned_dir / f"best_{current_run_name}.pt",
            pruned_dir / f"best_{run_name}.pt",
        ])
        forecaster_candidates = list(
            (ckpt_root / dataset_name / f"{model_name}_forecaster").glob(
                f"forecaster_src{prune_layer:02d}_tgt*.pt"
            )
        )
    else:
        rate = evit_base_keep_rate if evit_base_keep_rate is not None else keep_ratio
        keep_ratio = rate
        keep_pct = int(round(rate * 100))
        drop_locs = parse_evit_drop_locs(evit_drop_loc, n_blocks=10**9)
        drop_tag = "-".join(str(x) for x in drop_locs)
        fuse_tag = "fuse" if evit_fuse_token else "nofuse"
        run_name = (
            f"{model_name}_{dataset_name}_evit_drop{drop_tag}"
            f"_basekeep{int(rate * 100)}_{fuse_tag}"
        )
        legacy_run_name = f"{model_name}_{dataset_name}_evit_drop{drop_tag}_basekeep{int(rate * 100)}"
        pruned_ckpt = first_existing([
            pruned_dir / f"best_{run_name}.pt",
            pruned_dir / f"best_{legacy_run_name}.pt",
        ])
        forecaster_candidates = []

    result = {
        "method": pruning_method, "model": model_name, "dataset": dataset_name,
        "keep_ratio": keep_ratio, "keep_pct": keep_pct, "prune_layer": prune_layer,
        "evit_drop_locs": "", "evit_base_keep_rate": None, "evit_fuse_token": None,
        "status": "ok", "checkpoint": str(pruned_ckpt),
        "acc": None, "f1_macro": None, "tar_at_far": None,
        "threshold": None, "ms_per_img": None, "gflops": None, "message": "",
    }

    if not pruned_ckpt.exists():
        result.update(status="checkpoint_missing",
                      message=f"Missing: {pruned_ckpt}")
        return result
    if pruning_method == "eaf" and not forecaster_candidates:
        result.update(status="checkpoint_missing",
                      message=f"No forecaster for src={prune_layer} in "
                              f"{ckpt_root/dataset_name/f'{model_name}_forecaster'}")
        return result

    set_seed(seed)
    device = get_device()

    raw_backbone, transform, _ = get_model_from_name(model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)

    _, _, test_loader, _, n_classes = build_thunder_loaders(
        dataset_name, base_data_folder, transform,
        batch_size, num_workers, drop_last_train=False,
    )

    if pruning_method == "eaf":
        forecaster_ckpt = forecaster_candidates[0]
        forecaster = AttentionForecaster(
            embed_dim=adapter.embed_dim,
            hidden=hidden, n_heads=n_heads, n_layers=n_layers, dropout=forecaster_dropout,
        ).to(device)
        forecaster.load_state_dict(torch.load(forecaster_ckpt, map_location=device))
        forecaster.eval()
        for p in forecaster.parameters():
            p.requires_grad_(False)

        model = GenericLoRAWithForecasterPruning(
            backbone=raw_backbone, adapter=adapter, n_classes=n_classes,
            forecaster=forecaster, prune_layer=prune_layer, keep_ratio=keep_ratio,
        ).to(device)
    else:
        drop_locs = parse_evit_drop_locs(evit_drop_loc, adapter.n_blocks)
        rate = evit_base_keep_rate if evit_base_keep_rate is not None else keep_ratio
        model = LoRAWithEViTPruning(
            backbone=raw_backbone, adapter=adapter, n_classes=n_classes,
            base_keep_rate=rate, drop_locs=drop_locs, fuse_token=evit_fuse_token,
        ).to(device)
        model.set_keep_rate(rate)
        result.update(
            evit_drop_locs=",".join(str(x) for x in drop_locs),
            evit_base_keep_rate=rate,
            evit_fuse_token=evit_fuse_token,
        )

    missing, unexpected = model.load_state_dict(
        torch.load(pruned_ckpt, map_location=device), strict=False)
    model.eval()

    metrics = evaluate(model, test_loader, device, far_threshold=far_threshold)
    bench = benchmark_model(
        model, test_loader, device,
        label=f"{dataset_name} {model_name} keep={keep_pct}%",
    )

    result.update(
        acc=float(metrics["acc"]), f1_macro=float(metrics["f1_macro"]),
        tar_at_far=float(metrics["tar_at_far"]), threshold=float(metrics["threshold"]),
        ms_per_img=float(bench["ms_per_img"]),
        gflops=None if bench["gflops"] is None else float(bench["gflops"]),
        message=f"missing={len(missing)} unexpected={len(unexpected)}",
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Inference-only evaluation of EAF/EViT pruned checkpoints")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Thunder model name used during training (e.g. uni, hoptimus0)")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Thunder dataset name (e.g. crc, break_his)")
    parser.add_argument("--base-data-folder", type=str, required=True,
                        help="Path to Thunder base data folder")
    parser.add_argument("--ckpt-root", type=str, default="checkpoints",
                        help="Root directory for model checkpoints")
    parser.add_argument("--pruning-method", type=str, default="eaf",
                        choices=["eaf", "evit"])
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--prune-layers", type=int, nargs="+", default=[2])
    parser.add_argument("--evit-drop-loc", type=str, default="3,6,9")
    parser.add_argument("--evit-base-keep-rates", type=float, nargs="+", default=None,
                        help="EViT base keep rates to evaluate. Defaults to --keep-ratios.")
    parser.add_argument("--evit-fuse-token", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--far-threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=256,
                        help="Forecaster hidden dim — must match Phase 2.")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--forecaster-dropout", type=float, default=0.2)
    parser.add_argument("--output-csv", type=str,
                        default="results/eval_pruned_checkpoints.csv")
    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root)
    rows = []

    prune_layers = args.prune_layers if args.pruning_method == "eaf" else [None]
    keep_values = args.keep_ratios
    if args.pruning_method == "evit" and args.evit_base_keep_rates is not None:
        keep_values = args.evit_base_keep_rates

    for prune_layer in prune_layers:
        for keep_ratio in keep_values:
            print(f"\n[RUN] model={args.model_name} dataset={args.dataset_name} "
                  f"method={args.pruning_method} prune_layer={prune_layer} keep={keep_ratio}")
            row = evaluate_one(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                base_data_folder=args.base_data_folder,
                ckpt_root=ckpt_root,
                pruning_method=args.pruning_method,
                keep_ratio=keep_ratio,
                prune_layer=-1 if prune_layer is None else prune_layer,
                evit_drop_loc=args.evit_drop_loc,
                evit_base_keep_rate=keep_ratio if args.pruning_method == "evit" else None,
                evit_fuse_token=args.evit_fuse_token,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                far_threshold=args.far_threshold,
                seed=args.seed,
                hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers,
                forecaster_dropout=args.forecaster_dropout,
            )
            rows.append(row)
            print(f"  status={row['status']} acc={row['acc']} "
                  f"f1={row['f1_macro']} ms={row['ms_per_img']}")
            if row["message"]:
                print(f"  note: {row['message']}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "model", "dataset", "keep_ratio", "keep_pct", "prune_layer",
                  "evit_drop_locs", "evit_base_keep_rate", "evit_fuse_token",
                  "status", "checkpoint", "acc", "f1_macro", "tar_at_far",
                  "threshold", "ms_per_img", "gflops", "message"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    main()
