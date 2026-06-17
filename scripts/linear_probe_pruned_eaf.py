"""Linear probing on a frozen encoder with EAF-guided token pruning.

For each (eaf_type × dataset × keep_ratio) configuration:
  1. Load the appropriate forecaster (per-dataset or universal).
  2. Build a FrozenPrunedLinearProbe: frozen backbone + frozen forecaster +
     trainable linear head.
  3. Train only the head for --epochs epochs.
  4. Evaluate on the test split.
  5. Append a row to results/{model_name}.csv (resumes safely if re-run).

EAF types
---------
per_dataset : one forecaster per dataset, from train_per_dataset_eaf.py
              checkpoint path: {per-dataset-dir}/{dataset}/forecaster_src{L:02d}_attn{T:02d}.pt
universal   : one shared forecaster, from train_forecaster_unsupervised.py
              checkpoint path: {cache-dir}/{model}_forecaster/forecaster_{model}_{src_tag}_attn{T:02d}_universal.pt

Usage example
-------------
python scripts/linear_probe_pruned_eaf.py \\
    --model-name uni \\
    --base-data-folder /raid/DATASETS \\
    --cache-dir /raid/DATASETS/checkpoints/unsupervised \\
    --eaf-types per_dataset universal \\
    --layers-source 2 \\
    --keep-ratios 0.1 0.25 0.5 0.75 \\
    --epochs 20 \\
    --results-dir results/linear_probe_pruned
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy("file_system")

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.collection.unsupervised_cache import build_frozen_model
from src.data.thunder_loaders import build_thunder_loaders
from src.evaluation.metrics import evaluate
from src.models import FrozenPrunedLinearProbe, load_forecaster
from src.utils import get_device, set_seed


CSV_FIELDS = [
    "timestamp",
    "model_name",
    "dataset",
    "eaf_type",
    "backbone_variant",
    "layers_source",
    "prune_layer",
    "keep_ratio",
    "n_classes",
    "n_train",
    "n_val",
    "n_test",
    "best_epoch",
    "best_val_acc",
    "best_val_f1",
    "test_acc",
    "test_f1_macro",
    "test_auroc",
    "test_tar_at_far",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forecaster_path(eaf_type, dataset, model_name, layers_source,
                     layer_target, cache_dir, per_dataset_dir):
    if eaf_type == "per_dataset":
        ls = layers_source[0]
        return (
            Path(per_dataset_dir)
            / dataset
            / f"forecaster_src{ls:02d}_attn{layer_target:02d}.pt"
        )
    src_tag = "src" + "+".join(f"{ls:02d}" for ls in layers_source)
    return (
        Path(cache_dir)
        / f"{model_name}_forecaster"
        / f"forecaster_{model_name}_{src_tag}_attn{layer_target:02d}_universal.pt"
    )


def _load_done(csv_path):
    """Return a set of already-completed experiment keys for safe resume."""
    done = set()
    if not Path(csv_path).exists():
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((
                row["model_name"],
                row["dataset"],
                row["eaf_type"],
                row.get("backbone_variant", "pretrained"),
                row["layers_source"],
                int(row["prune_layer"]),
                float(row["keep_ratio"]),
            ))
    return done


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_head(model, train_loader, val_loader, epochs, lr, weight_decay, device):
    """Train model.head only; return (best_epoch, best_val_acc, best_val_f1)."""
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_f1, best_acc, best_ep = -1.0, 0.0, 0

    for ep in range(1, epochs + 1):
        model.train()  # FrozenPrunedLinearProbe.train() keeps backbone/forecaster in eval
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"ep{ep}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            loss = F.cross_entropy(model(imgs), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        val_m = evaluate(model, val_loader, device)
        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            best_acc = val_m["acc"]
            best_ep = ep

    return best_ep, float(best_acc), float(best_f1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Linear probing on frozen encoder with EAF token pruning"
    )
    ap.add_argument("--model-name",         type=str,   required=True)
    ap.add_argument("--base-data-folder",   type=str,   required=True)
    ap.add_argument("--cache-dir",          type=str,   default="checkpoints/unsupervised",
                    help="Directory with HDF5 caches and universal forecaster sub-folder.")
    ap.add_argument("--per-dataset-dir",    type=str,   default=None,
                    help="Directory of per-dataset forecaster checkpoints. "
                         "Default: {cache-dir}/per_dataset")
    ap.add_argument("--backbone-ckpt",      type=str,   default=None,
                    help="Optional state_dict to load onto the backbone before probing "
                         "(e.g. a CLS-distilled backbone from scripts/distill_pruned.py). "
                         "Default: unmodified pretrained weights.")
    ap.add_argument("--backbone-tag",       type=str,   default="pretrained",
                    help="Label for the backbone variant, written to the CSV "
                         "(e.g. 'distilled'). Purely informational.")
    ap.add_argument("--datasets",           type=str,   nargs="+", default=None,
                    help="Dataset names. Default: auto-discovered from HDF5 cache files.")
    ap.add_argument("--eaf-types",          type=str,   nargs="+",
                    default=["per_dataset", "universal"],
                    choices=["per_dataset", "universal"])
    ap.add_argument("--layers-source",      type=int,   nargs="+", default=[2],
                    help="Source block indices for the forecaster input (e.g. 1 2 3 4 5).")
    ap.add_argument("--layer-target",       type=int,   default=None,
                    help="Attention layer index the forecaster was trained to predict. "
                         "Default: last block.")
    ap.add_argument("--keep-ratios",        type=float, nargs="+", default=[0.25, 0.5, 0.75],
                    help="Fraction(s) of patch tokens to retain after pruning.")
    ap.add_argument("--epochs",             type=int,   default=20)
    ap.add_argument("--lr",                 type=float, default=1e-3)
    ap.add_argument("--weight-decay",       type=float, default=0.01)
    ap.add_argument("--batch-size",         type=int,   default=64)
    ap.add_argument("--num-workers",        type=int,   default=4)
    ap.add_argument("--forecaster-n-heads", type=int,   default=4,
                    help="Number of attention heads in the forecaster (must match training).")
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--results-dir",        type=str,   default="results/linear_probe_pruned",
                    help="Output directory; results saved to {results-dir}/{model_name}.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Load backbone once; share it across all experiments
    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    if args.backbone_ckpt:
        state = torch.load(args.backbone_ckpt, map_location=device, weights_only=True)
        raw_backbone.load_state_dict(state)
        print(f"Backbone weights loaded from: {args.backbone_ckpt} (tag={args.backbone_tag})")
    _, adapter = build_frozen_model(args.model_name, raw_backbone, device)
    layer_target = args.layer_target if args.layer_target is not None else adapter.n_blocks - 1
    per_dataset_dir = (
        Path(args.per_dataset_dir)
        if args.per_dataset_dir
        else Path(args.cache_dir) / "per_dataset"
    )
    cache_dir = Path(args.cache_dir)
    layers_source = sorted(args.layers_source)
    src_label = "+".join(map(str, layers_source))
    prune_layer = max(layers_source)

    # Discover datasets
    if args.datasets is None:
        h5s = sorted(cache_dir.glob(f"*_{args.model_name}_attn_features.h5"))
        dataset_names = [
            p.stem.replace(f"_{args.model_name}_attn_features", "") for p in h5s
        ]
        if not dataset_names:
            raise RuntimeError(
                f"No cache files matching *_{args.model_name}_attn_features.h5 "
                f"found in {cache_dir}. Run build_unsupervised_cache.py first."
            )
    else:
        dataset_names = args.datasets

    print(
        f"Model : {args.model_name} | embed_dim={adapter.embed_dim} "
        f"n_blocks={adapter.n_blocks} n_patches={adapter.n_patches}"
    )
    print(f"EAF   : {args.eaf_types}")
    print(f"Layers: source={layers_source} prune_at={prune_layer} target={layer_target}")
    print(f"Ratios: {args.keep_ratios}")
    print(f"Data  : {dataset_names}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"{args.model_name}.csv"
    done = _load_done(out_csv)
    write_header = not out_csv.exists()

    with open(out_csv, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for eaf_type in args.eaf_types:
            # Universal forecaster is loaded once and reused across all datasets
            universal_forecaster = None
            if eaf_type == "universal":
                ckpt = _forecaster_path(
                    "universal", None, args.model_name, layers_source,
                    layer_target, cache_dir, per_dataset_dir,
                )
                if not ckpt.exists():
                    print(f"\n[universal] Checkpoint not found: {ckpt} — skipping")
                    continue
                universal_forecaster = load_forecaster(ckpt, device, args.forecaster_n_heads)
                print(f"\n[universal] Forecaster loaded: {ckpt}")

            for dataset_name in dataset_names:
                for keep_ratio in args.keep_ratios:
                    tag = f"[{eaf_type}|{dataset_name}|keep={keep_ratio}]"
                    key = (args.model_name, dataset_name, eaf_type, args.backbone_tag,
                           src_label, prune_layer, keep_ratio)

                    if key in done:
                        print(f"{tag} already in CSV — skip")
                        continue

                    # Resolve forecaster
                    if eaf_type == "per_dataset":
                        ckpt = _forecaster_path(
                            "per_dataset", dataset_name, args.model_name, layers_source,
                            layer_target, cache_dir, per_dataset_dir,
                        )
                        if not ckpt.exists():
                            print(f"{tag} Forecaster not found: {ckpt} — skip")
                            continue
                        forecaster = load_forecaster(ckpt, device, args.forecaster_n_heads)
                    else:
                        forecaster = universal_forecaster

                    # Build data loaders
                    try:
                        train_ldr, val_ldr, test_ldr, class_names, n_classes = (
                            build_thunder_loaders(
                                dataset_name,
                                args.base_data_folder,
                                transform,
                                args.batch_size,
                                args.num_workers,
                            )
                        )
                    except Exception as exc:
                        print(f"{tag} Failed to build loaders: {exc}")
                        continue

                    print(
                        f"\n{tag} n_classes={n_classes} "
                        f"train={len(train_ldr.dataset)} "
                        f"val={len(val_ldr.dataset)} "
                        f"test={len(test_ldr.dataset)}"
                    )

                    set_seed(args.seed)
                    model = FrozenPrunedLinearProbe(
                        backbone=raw_backbone,
                        adapter=adapter,
                        forecaster=forecaster,
                        n_classes=n_classes,
                        layers_source=layers_source,
                        keep_ratio=keep_ratio,
                    ).to(device)

                    best_ep, best_acc, best_f1 = _train_head(
                        model, train_ldr, val_ldr,
                        args.epochs, args.lr, args.weight_decay, device,
                    )
                    test_m = evaluate(model, test_ldr, device)

                    row = {
                        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_name":      args.model_name,
                        "dataset":         dataset_name,
                        "eaf_type":        eaf_type,
                        "backbone_variant": args.backbone_tag,
                        "layers_source":   src_label,
                        "prune_layer":     prune_layer,
                        "keep_ratio":      keep_ratio,
                        "n_classes":       n_classes,
                        "n_train":         len(train_ldr.dataset),
                        "n_val":           len(val_ldr.dataset),
                        "n_test":          len(test_ldr.dataset),
                        "best_epoch":      best_ep,
                        "best_val_acc":    round(best_acc, 6),
                        "best_val_f1":     round(best_f1, 6),
                        "test_acc":        round(float(test_m["acc"]), 6),
                        "test_f1_macro":   round(float(test_m["f1_macro"]), 6),
                        "test_auroc":      round(float(test_m["auroc"]), 6),
                        "test_tar_at_far": round(float(test_m["tar_at_far"]), 6),
                    }
                    writer.writerow(row)
                    fout.flush()
                    print(
                        f"  → test_acc={row['test_acc']:.4f}  "
                        f"test_f1={row['test_f1_macro']:.4f}  "
                        f"auroc={row['test_auroc']:.4f}  "
                        f"tar@far={row['test_tar_at_far']:.4f}"
                    )

    print(f"\nDone. Results → {out_csv}")


if __name__ == "__main__":
    main()
