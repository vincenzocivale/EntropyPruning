#!/usr/bin/env python3
"""Evaluate the same tile encoder full vs EAF-pruned on THUNDER datasets.

EAF weights and LoRA distillation weights are frozen. Only a linear logistic
regression head is fitted downstream, with the same official THUNDER splits
for both representations. This script never trains EAF on THUNDER labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thunder.models.pretrained_models import get_model_from_name
from src.data.thunder_loaders import build_thunder_loaders, discover_thunder_datasets
from src.models import AttentionForecaster, ThunderBackboneAdapter
from src.models.online_tile_eaf import PrunedLoRAEncoder, unwrap_checkpoint_state
from src.utils import set_seed
from src.wsi_pipeline.experiment_registry import add_experiment_arguments, prepare_experiment_run
from src.wsi_pipeline.experiment_results import publish_run_summary


def _autocast(device: torch.device, dtype: str):
    amp = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda")


def _load_student(args, device: torch.device):
    payload = torch.load(args.pruned_checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model_name = args.model_name or config.get("model_name") or payload.get("base_model")
    if not model_name:
        raise ValueError("Could not resolve tile model name from CLI/checkpoint")
    raw, transform, _ = get_model_from_name(model_name, str(device))
    raw = raw.to(device)
    adapter = ThunderBackboneAdapter(raw, transform=transform)

    forecaster_path = args.forecaster_checkpoint or payload.get("forecaster_checkpoint")
    if not forecaster_path:
        raise ValueError("Could not resolve forecaster checkpoint")
    fpayload = torch.load(forecaster_path, map_location="cpu", weights_only=False)
    fcfg = fpayload.get("config", fpayload.get("args", {}))
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=int(config.get("hidden", fcfg.get("hidden", 256))),
        n_heads=int(config.get("n_heads", fcfg.get("n_heads", 4))),
        n_layers=int(config.get("n_layers", fcfg.get("n_layers", 2))),
        dropout=0.0,
    )
    forecaster.load_state_dict(unwrap_checkpoint_state(fpayload), strict=True)
    forecaster = forecaster.to(device).eval()

    student = PrunedLoRAEncoder(
        raw,
        adapter,
        forecaster,
        prune_layer=int(config["prune_layer"]),
        keep_ratio=float(config["keep_ratio"]),
        lora_r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
    ).to(device)
    student.load_trainable_state_dict(payload["trainable_state_dict"])
    student.eval()
    return student, transform, model_name, config


@torch.inference_mode()
def extract_pair(loader, student, *, device, amp_dtype):
    full, pruned, labels = [], [], []
    full_seconds = 0.0
    pruned_seconds = 0.0
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with _autocast(device, amp_dtype):
            full_embedding = student.full_teacher_embedding(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        full_seconds += time.perf_counter() - start

        start = time.perf_counter()
        with _autocast(device, amp_dtype):
            pruned_embedding = student(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pruned_seconds += time.perf_counter() - start

        full.append(full_embedding.float().cpu().numpy())
        pruned.append(pruned_embedding.float().cpu().numpy())
        labels.append(y.numpy())
    return (
        np.concatenate(full),
        np.concatenate(pruned),
        np.concatenate(labels).astype(int),
        full_seconds,
        pruned_seconds,
    )


def choose_c(X_train, y_train, X_val, y_val, c_grid):
    scaler = StandardScaler().fit(X_train)
    train = scaler.transform(X_train)
    val = scaler.transform(X_val)
    best = None
    for c in c_grid:
        clf = LogisticRegression(
            C=c, max_iter=3000, class_weight="balanced", multi_class="auto"
        )
        clf.fit(train, y_train)
        score = balanced_accuracy_score(y_val, clf.predict(val))
        candidate = (float(score), -float(c), float(c))
        if best is None or candidate > best:
            best = candidate
    return best[2]


def fit_and_score(X_train, y_train, X_val, y_val, X_test, y_test, c_grid):
    c = choose_c(X_train, y_train, X_val, y_val, c_grid)
    X_fit = np.concatenate([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    scaler = StandardScaler().fit(X_fit)
    clf = LogisticRegression(
        C=c, max_iter=3000, class_weight="balanced", multi_class="auto"
    )
    clf.fit(scaler.transform(X_fit), y_fit)
    X_test_s = scaler.transform(X_test)
    pred = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)
    result = {
        "C": c,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    classes = np.unique(y_test)
    if len(classes) == 2:
        result["auroc"] = float(roc_auc_score(y_test, proba[:, 1]))
    elif len(classes) > 2:
        try:
            result["auroc_ovr_macro"] = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--base-data-folder", required=True)
    parser.add_argument("--dataset", action="append", help="Repeatable; omit to evaluate every installed THUNDER dataset")
    parser.add_argument("--pruned-checkpoint", type=Path, required=True)
    parser.add_argument("--forecaster-checkpoint", type=Path)
    parser.add_argument("--model-name", help="Override model name stored in the distillation checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    experiment_run = prepare_experiment_run(args, family="tile_eaf", stage="evaluation")
    output_path = experiment_run.result_dir / "results.csv"
    if args.output is not None and args.output.expanduser().resolve() != output_path.resolve():
        raise ValueError(f"--output must equal canonical path: {output_path}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, transform, model_name, config = _load_student(args, device)
    datasets = args.dataset or discover_thunder_datasets(args.base_data_folder)
    rows = []

    for dataset in datasets:
        print(f"[THUNDER] {dataset}", flush=True)
        train_loader, val_loader, test_loader, class_names, n_classes = build_thunder_loaders(
            dataset,
            args.base_data_folder,
            transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            balanced_train=False,
        )
        split_values = {}
        timing = {"full": 0.0, "pruned": 0.0}
        for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
            full, pruned, y, full_s, pruned_s = extract_pair(
                loader, student, device=device, amp_dtype=args.amp_dtype
            )
            split_values[split] = {"full": full, "pruned": pruned, "y": y}
            timing["full"] += full_s
            timing["pruned"] += pruned_s

        for arm in ("full", "pruned"):
            metrics = fit_and_score(
                split_values["train"][arm], split_values["train"]["y"],
                split_values["val"][arm], split_values["val"]["y"],
                split_values["test"][arm], split_values["test"]["y"],
                args.c_grid,
            )
            n_images = sum(len(split_values[s]["y"]) for s in ("train", "val", "test"))
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "model": model_name,
                    "n_classes": n_classes,
                    "class_names": json.dumps(class_names),
                    "keep_ratio": 1.0 if arm == "full" else float(config["keep_ratio"]),
                    "prune_layer": "" if arm == "full" else int(config["prune_layer"]),
                    "embedding_seconds": timing[arm],
                    "images_per_second": n_images / max(timing[arm], 1e-9),
                    **metrics,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {output_path}")
    publish_run_summary(run=experiment_run, args=args, summary={"output_csv": str(output_path), "result_rows": len(rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
