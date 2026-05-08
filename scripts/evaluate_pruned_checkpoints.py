"""Inference-only evaluation of EAF pruned checkpoints.

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
                        ThunderBackboneAdapter)
from src.data.thunder_loaders import build_thunder_loaders
from src.evaluation import evaluate, benchmark_model


def evaluate_one(
    model_name: str,
    dataset_name: str,
    base_data_folder: str,
    ckpt_root: Path,
    keep_ratio: float,
    prune_layer: int,
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
    run_name = f"{model_name}_prune{prune_layer}_keep{keep_pct}"

    pruned_ckpt = (
        ckpt_root / dataset_name / f"{model_name}_pruned" / f"best_{run_name}.pt"
    )
    forecaster_ckpt = (
        ckpt_root / dataset_name / f"{model_name}_forecaster" /
        f"forecaster_src{prune_layer:02d}_tgt??.pt"
    )
    # Resolve wildcard for target layer
    forecaster_candidates = list(
        (ckpt_root / dataset_name / f"{model_name}_forecaster").glob(
            f"forecaster_src{prune_layer:02d}_tgt*.pt"
        )
    )

    result = {
        "model": model_name, "dataset": dataset_name,
        "keep_ratio": keep_ratio, "keep_pct": keep_pct, "prune_layer": prune_layer,
        "status": "ok", "checkpoint": str(pruned_ckpt),
        "acc": None, "f1_macro": None, "tar_at_far": None,
        "threshold": None, "ms_per_img": None, "gflops": None, "message": "",
    }

    if not pruned_ckpt.exists():
        result.update(status="checkpoint_missing",
                      message=f"Missing: {pruned_ckpt}")
        return result
    if not forecaster_candidates:
        result.update(status="checkpoint_missing",
                      message=f"No forecaster for src={prune_layer} in "
                              f"{ckpt_root/dataset_name/f'{model_name}_forecaster'}")
        return result

    forecaster_ckpt = forecaster_candidates[0]  # pick first match
    set_seed(seed)
    device = get_device()

    raw_backbone, transform, _ = get_model_from_name(model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)

    _, _, test_loader, _, n_classes = build_thunder_loaders(
        dataset_name, base_data_folder, transform,
        batch_size, num_workers, drop_last_train=False,
    )

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
        description="Inference-only evaluation of EAF pruned checkpoints")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Thunder model name used during training (e.g. uni, hoptimus0)")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Thunder dataset name (e.g. crc, break_his)")
    parser.add_argument("--base-data-folder", type=str, required=True,
                        help="Path to Thunder base data folder")
    parser.add_argument("--ckpt-root", type=str, default="checkpoints",
                        help="Root directory for model checkpoints")
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--prune-layers", type=int, nargs="+", default=[2])
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

    for prune_layer in args.prune_layers:
        for keep_ratio in args.keep_ratios:
            print(f"\n[RUN] model={args.model_name} dataset={args.dataset_name} "
                  f"prune_layer={prune_layer} keep_ratio={keep_ratio}")
            row = evaluate_one(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                base_data_folder=args.base_data_folder,
                ckpt_root=ckpt_root,
                keep_ratio=keep_ratio,
                prune_layer=prune_layer,
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
    fieldnames = ["model", "dataset", "keep_ratio", "keep_pct", "prune_layer",
                  "status", "checkpoint", "acc", "f1_macro", "tar_at_far",
                  "threshold", "ms_per_img", "gflops", "message"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    main()
