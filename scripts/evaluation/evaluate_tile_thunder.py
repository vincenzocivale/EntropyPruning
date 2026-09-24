#!/usr/bin/env python3
"""Evaluate the same tile encoder full vs EAF-pruned on THUNDER datasets.

EAF weights and LoRA distillation weights are frozen. Only a linear logistic
regression head is fitted downstream, with the same official THUNDER splits
for both representations. This script never trains EAF on THUNDER labels.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# THUNDER's largest slides (e.g. crc) decode to 150M+ pixel arrays; the default
# "file_descriptor" strategy shares each worker tensor via a dup'd fd and can
# corrupt sibling workers' handles under memory pressure (EBADF, aborted
# worker). "file_system" shares via named /dev/shm files instead, which is
# slower per-tensor but does not fail this way.
torch.multiprocessing.set_sharing_strategy("file_system")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.thunder_loaders import build_thunder_loaders, discover_thunder_datasets
from src.evaluation.checkpoints import _load_student
from src.utils import set_seed
from src.wsi_pipeline.experiment_registry import add_experiment_arguments, prepare_experiment_run
from src.wsi_pipeline.experiment_results import publish_run_summary


def _autocast(device: torch.device, dtype: str):
    amp = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda")


@torch.inference_mode()
def extract_pair(loader, student, *, device, amp_dtype, compute_full: bool = True):
    """Extract pruned (and, unless skipped, full-teacher) embeddings for one split.

    ``compute_full=False`` skips the frozen full-teacher forward entirely: it is
    identical across every pruned/keep_ratio variant of the same encoder, so a
    multi-variant run only needs it once per dataset (see main()).
    """
    full, pruned, labels = [], [], []
    full_seconds = 0.0
    pruned_seconds = 0.0
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if compute_full:
            start = time.perf_counter()
            with _autocast(device, amp_dtype):
                full_embedding = student.full_teacher_embedding(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            full_seconds += time.perf_counter() - start
            full.append(full_embedding.float().cpu().numpy())

        start = time.perf_counter()
        with _autocast(device, amp_dtype):
            pruned_embedding = student(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pruned_seconds += time.perf_counter() - start

        pruned.append(pruned_embedding.float().cpu().numpy())
        labels.append(y.numpy())
    return (
        np.concatenate(full) if compute_full else None,
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
            C=c, max_iter=3000, class_weight="balanced"
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
        C=c, max_iter=3000, class_weight="balanced"
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


FIELDNAMES = [
    "dataset", "arm", "model", "n_classes", "class_names", "keep_ratio",
    "prune_layer", "embedding_seconds", "images_per_second", "C",
    "accuracy", "balanced_accuracy", "macro_f1", "auroc", "auroc_ovr_macro",
]


def _prepare_job_output(job_args) -> tuple[object, Path, set[str], dict[str, dict]]:
    """Resolve one variant's canonical results.csv and its resumable state.

    Returns (experiment_run, output_path, done_datasets, full_rows_by_dataset).
    ``full_rows_by_dataset`` lets a later job in the same invocation reuse this
    job's already-written "full" arm rows for datasets it already finished,
    instead of recomputing them.
    """
    experiment_run = prepare_experiment_run(job_args, family="tile_eaf", stage="evaluation")
    output_path = experiment_run.result_dir / "results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_datasets: set[str] = set()
    full_rows: dict[str, dict] = {}
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        counts: dict[str, int] = {}
        for row in existing:
            counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
            if row["arm"] == "full":
                full_rows[row["dataset"]] = row
        # A dataset is only "done" once both arms (full + pruned) are recorded;
        # a crash mid-dataset must not skip its retry.
        done_datasets = {name for name, count in counts.items() if count >= 2}
        print(f"[THUNDER] {job_args.variant_id}: resuming; {len(done_datasets)} dataset(s) already complete: {sorted(done_datasets)}", flush=True)
    else:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()

    return experiment_run, output_path, done_datasets, full_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--base-data-folder", required=True)
    parser.add_argument("--dataset", action="append", help="Repeatable; omit to evaluate every installed THUNDER dataset")
    parser.add_argument("--pruned-checkpoint", type=Path, required=True)
    parser.add_argument("--forecaster-checkpoint", type=Path)
    parser.add_argument("--model-name", help="Override model name stored in the distillation checkpoint")
    parser.add_argument(
        "--also-variant-id", action="append", default=[],
        help="Additional variant_id(s) to evaluate in this same process, reusing the "
             "full-teacher embeddings/fit computed for the primary --variant-id instead "
             "of recomputing them (the full arm never depends on the pruned checkpoint). "
             "Must be paired 1:1 with --also-pruned-checkpoint (and optionally "
             "--also-forecaster-checkpoint).",
    )
    parser.add_argument("--also-pruned-checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--also-forecaster-checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is not None:
        raise ValueError("--output is not supported; the canonical results.csv path is derived per variant_id")
    if len(args.also_variant_id) != len(args.also_pruned_checkpoint):
        raise ValueError("--also-variant-id and --also-pruned-checkpoint must be given the same number of times")
    also_forecasters = args.also_forecaster_checkpoint or [None] * len(args.also_variant_id)
    if len(also_forecasters) != len(args.also_variant_id):
        raise ValueError("--also-forecaster-checkpoint must be given once per --also-variant-id, or not at all")

    jobs = [(args.variant_id, args.pruned_checkpoint, args.forecaster_checkpoint)]
    jobs += list(zip(args.also_variant_id, args.also_pruned_checkpoint, also_forecasters))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.dataset or discover_thunder_datasets(args.base_data_folder)

    # Populated with the "full" arm row for each dataset the first time it is
    # computed (by whichever job hits it first, including from a resumed job's
    # own prior results.csv); every later job reuses it instead of
    # recomputing the frozen full-teacher forward + linear probe fit.
    full_row_cache: dict[str, dict] = {}

    for job_index, (variant_id, pruned_checkpoint, forecaster_checkpoint) in enumerate(jobs):
        job_args = copy.copy(args)
        job_args.variant_id = variant_id
        job_args.pruned_checkpoint = pruned_checkpoint
        job_args.forecaster_checkpoint = forecaster_checkpoint

        experiment_run, output_path, done_datasets, existing_full_rows = _prepare_job_output(job_args)
        for name, row in existing_full_rows.items():
            full_row_cache.setdefault(name, row)
        total_rows = len(done_datasets) * 2

        student = None
        transform = None
        model_name = None
        config = None

        for dataset in datasets:
            if dataset in done_datasets:
                print(f"[THUNDER] {variant_id}: {dataset} (skipped, already in {output_path})", flush=True)
                continue

            need_full = dataset not in full_row_cache
            if student is None:
                set_seed(args.seed)
                student, transform, model_name, config = _load_student(job_args, device)

            print(f"[THUNDER] {variant_id}: {dataset}" + ("" if need_full else " (full arm reused)"), flush=True)
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
                    loader, student, device=device, amp_dtype=args.amp_dtype, compute_full=need_full,
                )
                split_values[split] = {"full": full, "pruned": pruned, "y": y}
                timing["full"] += full_s
                timing["pruned"] += pruned_s

            dataset_rows = []
            if need_full:
                metrics = fit_and_score(
                    split_values["train"]["full"], split_values["train"]["y"],
                    split_values["val"]["full"], split_values["val"]["y"],
                    split_values["test"]["full"], split_values["test"]["y"],
                    args.c_grid,
                )
                n_images = sum(len(split_values[s]["y"]) for s in ("train", "val", "test"))
                full_row = {
                    "dataset": dataset, "arm": "full", "model": model_name,
                    "n_classes": n_classes, "class_names": json.dumps(class_names),
                    "keep_ratio": 1.0, "prune_layer": "",
                    "embedding_seconds": timing["full"],
                    "images_per_second": n_images / max(timing["full"], 1e-9),
                    **metrics,
                }
                full_row_cache[dataset] = full_row
            else:
                full_row = full_row_cache[dataset]
            dataset_rows.append(full_row)

            pruned_metrics = fit_and_score(
                split_values["train"]["pruned"], split_values["train"]["y"],
                split_values["val"]["pruned"], split_values["val"]["y"],
                split_values["test"]["pruned"], split_values["test"]["y"],
                args.c_grid,
            )
            n_images = sum(len(split_values[s]["y"]) for s in ("train", "val", "test"))
            dataset_rows.append(
                {
                    "dataset": dataset, "arm": "pruned", "model": model_name,
                    "n_classes": n_classes, "class_names": json.dumps(class_names),
                    "keep_ratio": float(config["keep_ratio"]), "prune_layer": int(config["prune_layer"]),
                    "embedding_seconds": timing["pruned"],
                    "images_per_second": n_images / max(timing["pruned"], 1e-9),
                    **pruned_metrics,
                }
            )

            # Flush after every dataset so a crash on dataset N+1 never loses the
            # embeddings/fit already computed for datasets 1..N.
            with output_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writerows(dataset_rows)
            total_rows += len(dataset_rows)
            print(f"[THUNDER] {variant_id}: {dataset} done -> appended {len(dataset_rows)} rows ({total_rows} total) to {output_path}", flush=True)

        print(f"[THUNDER] {variant_id}: wrote {total_rows} rows -> {output_path}")
        publish_run_summary(run=experiment_run, args=job_args, summary={"output_csv": str(output_path), "result_rows": total_rows})

        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
