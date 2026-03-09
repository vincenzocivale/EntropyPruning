#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from entropy_pruning import (  # noqa: E402
    AttentionForecaster,
    UNILoRAClassifier,
    build_attention_cache,
    build_loaders,
    finetune_pruned_classifier,
    set_seed,
    train_forecaster,
)


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Layer ablation for source->target forecaster pairs")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--classifier-ckpt", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--forecaster-dir", required=True)
    parser.add_argument("--results-csv", default="ablation_results.csv")

    parser.add_argument("--source-layers", default="2")
    parser.add_argument("--target-layers", default="23")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")

    parser.add_argument("--forecaster-epochs", type=int, default=30)
    parser.add_argument("--forecaster-num-workers", type=int, default=0)
    parser.add_argument("--forecaster-lr", type=float, default=1e-4)
    parser.add_argument("--forecaster-wd", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--run-pruning", action="store_true")
    parser.add_argument("--reuse-forecaster-ckpt", action="store_true")
    parser.add_argument("--phase3-only-summary", action="store_true")
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--pruning-epochs", type=int, default=10)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--cache-flush-every", type=int, default=16)
    parser.add_argument("--cache-compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--cache-chunk-size", type=int, default=64)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_layers = parse_int_list(args.source_layers)
    target_layers = parse_int_list(args.target_layers)

    loaders = build_loaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_column=args.image_column,
        label_column=args.label_column,
    )

    cache_path = Path(args.cache_path)
    forecaster_dir = Path(args.forecaster_dir)
    forecaster_dir.mkdir(parents=True, exist_ok=True)

    base_model = UNILoRAClassifier(loaders.n_classes).to(device)
    ckpt = torch.load(args.classifier_ckpt, map_location=device)
    base_model.load_state_dict(ckpt, strict=False)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    def cache_has_required_splits(path: Path) -> bool:
        if not path.exists():
            return False
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            has_val = "val" in keys or "validation" in keys
            return "train" in keys and has_val and "test" in keys

    if args.force_rebuild_cache or not cache_has_required_splits(cache_path):
        compression = None if args.cache_compression == "none" else args.cache_compression
        build_attention_cache(
            model=base_model,
            loaders={
                "train": loaders.train_loader,
                "val": loaders.val_loader,
                "test": loaders.test_loader,
            },
            device=device,
            source_layers=source_layers,
            target_layers=target_layers,
            save_path=cache_path,
            flush_every=args.cache_flush_every,
            compression=compression,
            chunk_size=args.cache_chunk_size,
        )

    rows = []
    for src in source_layers:
        for tgt in target_layers:
            run_name = f"src{src:02d}_tgt{tgt:02d}"
            forecaster_path = forecaster_dir / f"forecaster_{run_name}.pt"
            used_existing_forecaster = False

            if args.reuse_forecaster_ckpt and forecaster_path.exists():
                forecaster = AttentionForecaster(
                    embed_dim=1024,
                    hidden=args.hidden,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                ).to(device)
                forecaster.load_state_dict(torch.load(forecaster_path, map_location=device))
                forecaster.eval()
                for p in forecaster.parameters():
                    p.requires_grad_(False)
                result = {
                    "layer_source": src,
                    "layer_target": tgt,
                    "best_val_kl": "",
                    "best_val_rho": "",
                    "test_rho_forecaster": "",
                    "test_rho_token_norm": "",
                    "model": forecaster,
                }
                used_existing_forecaster = True
            else:
                result = train_forecaster(
                    h5_cache_path=cache_path,
                    layer_source=src,
                    layer_target=tgt,
                    device=device,
                    hidden=args.hidden,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    batch_size=64,
                    num_workers=args.forecaster_num_workers,
                    epochs=args.forecaster_epochs,
                    lr=args.forecaster_lr,
                    weight_decay=args.forecaster_wd,
                    save_path=forecaster_path,
                )

            row = {
                "source_layer": src,
                "target_layer": tgt,
                "forecaster_from_ckpt": int(used_existing_forecaster),
                "best_val_kl": result["best_val_kl"],
                "best_val_rho": result["best_val_rho"],
                "test_rho_forecaster": result["test_rho_forecaster"],
                "test_rho_token_norm": result["test_rho_token_norm"],
                "delta_rho_vs_norm": (
                    ""
                    if result["test_rho_forecaster"] == "" or result["test_rho_token_norm"] == ""
                    else result["test_rho_forecaster"] - result["test_rho_token_norm"]
                ),
                "pruned_val_best_f1": "",
                "pruned_test_acc": "",
                "pruned_test_f1_macro": "",
            }

            if args.run_pruning:
                prune_result = finetune_pruned_classifier(
                    n_classes=loaders.n_classes,
                    classifier_ckpt=args.classifier_ckpt,
                    forecaster=result["model"],
                    prune_layer=src,
                    keep_ratio=args.keep_ratio,
                    train_loader=loaders.train_loader,
                    val_loader=loaders.val_loader,
                    test_loader=loaders.test_loader,
                    device=device,
                    epochs=args.pruning_epochs,
                    lr_backbone=args.lr_backbone,
                    lr_head=args.lr_head,
                    weight_decay=args.weight_decay,
                    label_smoothing=args.label_smoothing,
                )
                row["pruned_val_best_f1"] = prune_result["val_best_f1"]
                row["pruned_test_acc"] = prune_result["test_acc"]
                row["pruned_test_f1_macro"] = prune_result["test_f1_macro"]

            rows.append(row)
            print(row)

    out_csv = Path(args.results_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved ablation results to: {out_csv}")

    if args.run_pruning:
        phase3_rows = [r for r in rows if r["pruned_test_f1_macro"] != ""]
        phase3_rows.sort(key=lambda r: float(r["pruned_test_f1_macro"]), reverse=True)
        print("\nPhase-3 ranking (by pruned_test_f1_macro):")
        for r in phase3_rows:
            print(
                f"src{int(r['source_layer']):02d}->tgt{int(r['target_layer']):02d} | "
                f"test_f1={float(r['pruned_test_f1_macro']):.4f} | "
                f"test_acc={float(r['pruned_test_acc']):.4f}"
            )

        if args.phase3_only_summary:
            phase3_csv = out_csv.with_name(f"{out_csv.stem}_phase3{out_csv.suffix}")
            phase3_fields = ["source_layer", "target_layer", "pruned_val_best_f1", "pruned_test_acc", "pruned_test_f1_macro"]
            with phase3_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase3_fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {k: r[k] for k in phase3_fields}
                        for r in phase3_rows
                    ]
                )
            print(f"Saved phase-3 summary to: {phase3_csv}")


if __name__ == "__main__":
    main()
