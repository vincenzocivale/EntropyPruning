#!/usr/bin/env python
"""Downstream evaluation of WSI-FM slide embeddings on labeled datasets from
the configured downstream benchmark bank (see docs/pipeline.md).

Compares the frozen baseline WSI-FM (TITAN) against one or more WSI-EAF
Stage-2 checkpoints (`scripts/training/distill_wsi_titan.py`,
`checkpoints/wsi_eaf_pruned/<pair>/<run>/`) by training a downstream head on
each model's slide embeddings for each labeled task. No backbone/forecaster
weights are updated here -- probing only, mirroring the tile-EAF Stage-3
evaluation (`scripts/training/train_multi_thunder_classifier.py --adaptation
linear_probing`) one level up. Three protocols, chosen by which of
--train-cohort/--test-cohort/--classifier are given:

- Default (neither flag): within-cohort `StratifiedGroupKFold` linear probe,
  `evaluate_linear_probe`. Not comparable to EAGLE's reported AUROCs -- no
  external-cohort generalization test.
- `--train-cohort`/`--test-cohort`, `--classifier logreg` (default once these
  are set): single LogisticRegression fit on the pooled train cohorts,
  evaluated once on the pooled test cohorts, `evaluate_train_test`.
- `--train-cohort`/`--test-cohort`, `--classifier mlp`: EAGLE's actual
  main-benchmark (Fig. 1-5) protocol -- 5-fold-ensemble MLP head,
  `evaluate_fig2_protocol` -- see docs/pipeline.md "Evaluation only" for the
  full recipe and an example command.

Two embedding sources per task:

- Baseline (frozen TITAN): read the pre-cached `slide_embedding` straight out
  of the `eaf.wsi.fm_output.v1` cache (`scripts/features/cache_wsi_teacher.py`
  output, `--teacher-wsi-root`) -- no model forward pass at all.
- Each `--pruned-checkpoint`: read per-slide `coords`/`tile_embeddings` from
  the matching Tile-EAF cache (`--tile-input-root`) and run them live through
  `PrunedLoRATitanEncoder` (forecaster-guided pruning + LoRA), exactly as
  `distill_wsi_titan.py` does during training -- there is no
  precomputed pruned slide_embedding cache, pruning is cheap enough to run at
  eval time and this avoids maintaining yet another cache namespace per
  checkpoint.

Task/dataset discovery is automatic (`discover_tasks`): any
`<labels-root>/<cohort>/labels/<task>.csv` (two columns: `slide_id,label`) is
a candidate task, but a task is only evaluated once matching embeddings are
actually found for at least `--min-slides` slides on both sides (label +
embedding) -- so this script runs correctly today against a mostly-empty
downstream tree (reports 0 usable tasks) and picks up more cohorts
automatically as TCGA/CPTAC/Patho-Bench downloads complete, with no code change.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.wsi_pipeline.numpy_store import preferred_path, read_array  # noqa: E402
from src.data.wsi.layout import StoreLayout  # noqa: E402
from src.utils import set_seed  # noqa: E402
from src.wsi_pipeline.io import read_wsi_output_record  # noqa: E402
from src.wsi_pipeline.experiment_registry import add_experiment_arguments, prepare_experiment_run  # noqa: E402
from src.wsi_pipeline.experiment_results import publish_run_summary  # noqa: E402
from src.evaluation.checkpoints import _load_pruned_titan, _load_wsi_forecaster  # noqa: E402


@dataclass(frozen=True)
class TaskSpec:
    cohort: str
    task: str
    labels_path: Path
    members: tuple[tuple[str, Path], ...] = ()


def discover_tasks(labels_root: Path) -> list[TaskSpec]:
    """Scan `<labels-root>/<cohort>/labels/*.csv` -- no hardcoded cohort/task list.

    Returns an empty list (not an error) when `labels_root` doesn't exist yet or
    holds no `labels/` subdirectories.
    """
    tasks: list[TaskSpec] = []
    if not labels_root.is_dir():
        return tasks
    for cohort_dir in sorted(labels_root.iterdir()):
        labels_dir = cohort_dir / "labels"
        if not labels_dir.is_dir():
            continue
        for csv_path in sorted(labels_dir.glob("*.csv")):
            tasks.append(TaskSpec(cohort=cohort_dir.name, task=csv_path.stem, labels_path=csv_path))
    # Some cross-cohort diagnoses are encoded as one class per cohort (for
    # example LUAD versus LUSC). Their individual files cannot support CV, but
    # their union can. Keep the individual tasks visible as skipped rows and add
    # one explicitly named combined task when the classes complement each other.
    by_name: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        by_name.setdefault(task.task, []).append(task)
    for name, members in sorted(by_name.items()):
        if len(members) < 2:
            continue
        classes = [set(_read_labels(member.labels_path).values()) for member in members]
        if all(len(values) == 1 for values in classes) and len(set().union(*classes)) > 1:
            tasks.append(TaskSpec(
                cohort="+".join(member.cohort for member in members),
                task=name, labels_path=members[0].labels_path,
                members=tuple((member.cohort, member.labels_path) for member in members),
            ))
    return tasks


def _read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {"slide_id", "label"} - set(reader.fieldnames):
            raise ValueError(f"{path} must have columns slide_id,label (found {reader.fieldnames})")
        for row in reader:
            if row["slide_id"] and row["label"] not in (None, ""):
                labels[row["slide_id"]] = row["label"]
    return labels


def _baseline_embedding(cohort_dir: Path, slide_id: str) -> np.ndarray | None:
    path = preferred_path(cohort_dir / f"{slide_id}.h5")
    if not path.exists():
        return None
    record = read_wsi_output_record(path)
    embedding = np.asarray(record.slide_embedding, dtype=np.float32)
    return embedding.reshape(-1)


def _pruned_embedding(
    cohort_dir: Path, slide_id: str, *, student: PrunedLoRATitanEncoder, device: torch.device
) -> np.ndarray | None:
    path = preferred_path(cohort_dir / f"{slide_id}.h5")
    if not path.exists():
        return None
    coords = read_array(path, "coords")
    tile_embeddings = read_array(path, "tile_embeddings")
    with torch.no_grad():
        embedding = student(
            torch.from_numpy(tile_embeddings).float().to(device),
            torch.from_numpy(coords).long().to(device),
        )
    return embedding.detach().float().cpu().numpy().reshape(-1)


def collect_embeddings(
    task: TaskSpec,
    *,
    cache_root: Path,
    embed_fn,
    min_slides: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Returns (X, labels, patient_groups, slide_keys, n_found, n_missing). Labels are returned as the
    original string values; caller label-encodes once both model passes share the
    same slide set (a slide missing under one model's cache must not silently
    change the label encoding used for another)."""
    rows: list[np.ndarray] = []
    kept_labels: list[str] = []
    patient_groups: list[str] = []
    slide_keys: list[str] = []
    missing = 0
    members = task.members or ((task.cohort, task.labels_path),)
    for cohort, labels_path in members:
        labels = _read_labels(labels_path)
        cohort_dir = cache_root / cohort
        for slide_id, label in labels.items():
            embedding = embed_fn(cohort_dir, slide_id)
            if embedding is None:
                missing += 1
                continue
            rows.append(embedding)
            kept_labels.append(label)
            patient_groups.append(slide_id[:12] if slide_id.startswith("TCGA-") else slide_id)
            slide_keys.append(f"{cohort}/{slide_id}")
    if not rows:
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)), len(rows), missing
    return np.stack(rows), np.asarray(kept_labels), np.asarray(patient_groups), np.asarray(slide_keys), len(rows), missing


def _paired_indices(keys_by_model: list[np.ndarray]) -> list[np.ndarray]:
    """Indices of the same slides, in first-model order, for every model."""
    if not keys_by_model:
        return []
    shared = set(keys_by_model[0])
    for keys in keys_by_model[1:]:
        shared.intersection_update(keys)
    ordered = [key for key in keys_by_model[0] if key in shared]
    positions = [{key: index for index, key in enumerate(keys)} for keys in keys_by_model]
    return [np.asarray([mapping[key] for key in ordered], dtype=int) for mapping in positions]


def _cv_splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, folds: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    groups_per_class = [len(set(groups[y == label])) for label in np.unique(y)]
    usable_folds = min(folds, min(groups_per_class)) if groups_per_class else 0
    if usable_folds < 2:
        return usable_folds, []
    splitter = StratifiedGroupKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    return usable_folds, list(splitter.split(X, y, groups))


# TCGA (build_tcga_cptac_benchmark_labels.py) and CPTAC/Patho-Bench
# (build_cptac_pathobench_labels.py) name the same biomarker task differently.
# Canonicalize both to one key so a train cohort's task can be matched against
# a same-biomarker task in a disjoint test cohort. Values are the ad hoc,
# lowercased names actually observed on disk for each source.
_TASK_ALIASES: dict[str, str] = {
    "pik3ca_mutation": "pik3ca_mutation",
    "kras_mutation": "kras_mutation",
    "braf_mutation": "braf_mutation",
    "egfr_mutation": "egfr_mutation",
    "stk11_mutation": "stk11_mutation",
    "tp53_mutation": "tp53_mutation",
    "msi_status": "msi_status", "msi_h": "msi_status",
}


def canonical_task_name(name: str) -> str:
    return _TASK_ALIASES.get(name.lower(), name.lower())


def evaluate_train_test(
    X_train: np.ndarray, y_train_labels: np.ndarray, X_test: np.ndarray, y_test_labels: np.ndarray,
) -> dict:
    """Fit once on `train`, evaluate once on disjoint `test` -- no CV, no shared
    patients possible since train/test come from different cohorts. Mirrors
    EAGLE's train-on-TCGA/test-on-external-cohort protocol, unlike
    `evaluate_linear_probe`'s within-cohort k-fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train_labels, y_test_labels]))
    y_train, y_test = encoder.transform(y_train_labels), encoder.transform(y_test_labels)
    n_classes = len(encoder.classes_)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"status": "skipped", "reason": "train or test split lacks one class"}

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    result = {
        "status": "ok",
        "n_train_slides": int(X_train.shape[0]),
        "n_test_slides": int(X_test.shape[0]),
        "n_classes": n_classes,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
    }
    if n_classes == 2:
        proba = clf.predict_proba(X_test_s)[:, 1]
        result["auroc"] = float(roc_auc_score(y_test, proba))
    return result


class _MLPHead(torch.nn.Module):
    """The classification head EAGLE trains on every slide encoder's embedding
    for its main 31-task benchmark (Methods, "Once a slide-level or patient-
    level embedding was computed, it was fed into a small multilayer
    perceptron..."): hidden=256, SiLU, dropout, binary/multi-class logits."""

    def __init__(self, in_dim: int, n_classes: int, *, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_head(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    *, n_classes: int, device: torch.device, epochs: int = 32, lr: float = 1e-4, weight_decay: float = 1e-2,
    seed: int = 42,
) -> _MLPHead:
    """AdamW + one-cycle LR, class-weighted cross-entropy, early stopping on
    validation loss -- mirrors EAGLE's Methods recipe for the main-benchmark
    MLP classifier (768-in-dim in the paper was CONCH/CTransPath-specific;
    here `in_dim` is inferred from whatever embedding is being evaluated)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weight = torch.tensor(class_counts.sum() / np.maximum(class_counts, 1), dtype=torch.float32, device=device)
    class_weight = class_weight / class_weight.mean()

    model = _MLPHead(X_train.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, (len(X_train) + 63) // 64)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    best_val_loss = float("inf")
    best_state = None
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t), generator=generator)
        for start in range(0, len(perm), 64):
            idx = perm[start : start + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def mlp_predict_proba(model: _MLPHead, X: np.ndarray, *, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        return torch.softmax(logits, dim=-1).cpu().numpy()


def evaluate_fig2_protocol(
    X_train_pool: np.ndarray, y_train_labels: np.ndarray, groups_train: np.ndarray,
    X_test: np.ndarray, y_test_labels: np.ndarray,
    *, device: torch.device, folds: int = 5, seed: int = 42,
) -> dict:
    """EAGLE's main-benchmark (Fig. 1-5) protocol: 5-fold split of the train
    pool (80% train / 20% validation per fold, grouped by patient), one MLP
    trained per fold; each of the 5 fold-models scores the (disjoint,
    never-trained-on) external test set once; per-slide test probabilities
    are averaged across the 5 fold-models (ensemble) before computing metrics
    -- not 5 separate AUROCs averaged, one AUROC of the averaged scores."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train_labels, y_test_labels]))
    y_train_pool = encoder.transform(y_train_labels)
    y_test = encoder.transform(y_test_labels)
    n_classes = len(encoder.classes_)
    if len(np.unique(y_train_pool)) < 2 or len(np.unique(y_test)) < 2:
        return {"status": "skipped", "reason": "train pool or test split lacks one class"}

    usable_folds, splits = _cv_splits(X_train_pool, y_train_pool, groups_train, folds=folds, seed=seed)
    if usable_folds < 2:
        return {"status": "skipped", "reason": "smallest class in train pool has fewer than 2 patient groups"}

    test_proba_sum = np.zeros((X_test.shape[0], n_classes), dtype=np.float64)
    n_fold_models = 0
    for train_idx, val_idx in splits:
        if len(np.unique(y_train_pool[train_idx])) < 2 or len(np.unique(y_train_pool[val_idx])) < 2:
            continue
        scaler = StandardScaler().fit(X_train_pool[train_idx])
        X_tr = scaler.transform(X_train_pool[train_idx])
        X_val = scaler.transform(X_train_pool[val_idx])
        X_te = scaler.transform(X_test)
        model = train_mlp_head(
            X_tr, y_train_pool[train_idx], X_val, y_train_pool[val_idx], n_classes=n_classes, device=device, seed=seed
        )
        test_proba_sum += mlp_predict_proba(model, X_te, device=device)
        n_fold_models += 1

    if n_fold_models < 2:
        return {"status": "skipped", "reason": "fewer than 2 usable fold-models trained"}

    test_proba = test_proba_sum / n_fold_models
    preds = test_proba.argmax(axis=-1)
    result = {
        "status": "ok",
        "n_train_slides": int(X_train_pool.shape[0]),
        "n_test_slides": int(X_test.shape[0]),
        "n_classes": n_classes,
        "n_fold_models": n_fold_models,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
    }
    if n_classes == 2:
        result["auroc"] = float(roc_auc_score(y_test, test_proba[:, 1]))
    return result


def evaluate_linear_probe(X: np.ndarray, y_labels: np.ndarray, groups: np.ndarray, *, folds: int, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)
    n_classes = len(encoder.classes_)
    usable_folds, splits = _cv_splits(X, y, groups, folds=folds, seed=seed)
    if usable_folds < 2:
        return {"status": "skipped", "reason": "smallest class has fewer than 2 patient groups"}

    accuracies, f1_macros, aurocs = [], [], []
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            return {"status": "skipped", "reason": "a patient-disjoint fold lacks one class"}
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_test)
        accuracies.append(accuracy_score(y[test_idx], preds))
        f1_macros.append(f1_score(y[test_idx], preds, average="macro"))
        if n_classes == 2:
            proba = clf.predict_proba(X_test)[:, 1]
            if len(np.unique(y[test_idx])) == 2:
                aurocs.append(roc_auc_score(y[test_idx], proba))
    result = {
        "status": "ok",
        "n_slides": int(X.shape[0]),
        "n_patients": int(len(set(groups))),
        "cv_unit": "patient",
        "n_classes": n_classes,
        "folds": usable_folds,
        "accuracy": float(np.mean(accuracies)),
        "f1_macro": float(np.mean(f1_macros)),
    }
    if aurocs:
        result["auroc"] = float(np.mean(aurocs))
    return result


def _write_results(results: list[dict], experiment_run, args) -> Path:
    output_csv = experiment_run.result_dir / "results.csv"
    if args.output_csv is not None and args.output_csv.expanduser().resolve() != output_csv.resolve():
        raise ValueError(f"--output-csv must equal canonical path: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in results for key in row})
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} rows to {output_csv}")
    publish_run_summary(
        run=experiment_run, args=args,
        summary={"output_csv": str(output_csv), "result_rows": len(results)},
    )
    return output_csv


def _collect_pooled(
    tasks: list[TaskSpec], *, cache_root: Path, embed_fn, min_slides: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Concatenate collect_embeddings across every TaskSpec in `tasks` (one per
    cohort sharing the same canonical task name) into a single pooled set.
    Returns (X, y, groups, keys, n_found, n_missing)."""
    X_parts, y_parts, groups_parts, keys_parts = [], [], [], []
    n_found_total, n_missing_total = 0, 0
    for task in tasks:
        X, y, groups, keys, n_found, n_missing = collect_embeddings(
            task, cache_root=cache_root, embed_fn=embed_fn, min_slides=min_slides
        )
        if X.shape[0]:
            X_parts.append(X)
            y_parts.append(y)
            groups_parts.append(groups)
            keys_parts.append(keys)
        n_found_total += n_found
        n_missing_total += n_missing
    if not X_parts:
        empty = np.empty((0,))
        return empty, empty, empty, empty, n_found_total, n_missing_total
    return (
        np.concatenate(X_parts), np.concatenate(y_parts), np.concatenate(groups_parts), np.concatenate(keys_parts),
        n_found_total, n_missing_total,
    )


def run_cross_cohort(
    tasks: list[TaskSpec], *, models: list[tuple[str, object]], train_cohorts: set[str], test_cohorts: set[str],
    teacher_wsi_root: Path, tile_input_root: Path | None, device: torch.device, min_slides: int, use_wandb: bool,
) -> list[dict]:
    """EAGLE-style protocol: fit on all labeled slides pooled from `train_cohorts`,
    evaluate once on `test_cohorts` -- no k-fold, no shared patients (disjoint
    cohorts). Tasks are matched across train/test by canonical_task_name, since
    TCGA and CPTAC/Patho-Bench label files use different task-name conventions
    for the same biomarker."""
    by_canonical: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        if task.members:  # skip synthetic cross-cohort union tasks; this function does its own union
            continue
        by_canonical.setdefault(canonical_task_name(task.task), []).append(task)

    results: list[dict] = []
    for canonical, members in sorted(by_canonical.items()):
        train_tasks = [t for t in members if t.cohort in train_cohorts]
        test_tasks = [t for t in members if t.cohort in test_cohorts]
        if not train_tasks or not test_tasks:
            continue
        for model_name, student in models:
            if student is None:
                embed_fn, cache_root = _baseline_embedding, teacher_wsi_root
            else:
                embed_fn = lambda cohort_dir, slide_id, _s=student: _pruned_embedding(
                    cohort_dir, slide_id, student=_s, device=device
                )
                cache_root = tile_input_root

            X_train, y_train, _g1, _k1, n_found_tr, n_missing_tr = _collect_pooled(
                train_tasks, cache_root=cache_root, embed_fn=embed_fn, min_slides=min_slides
            )
            X_test, y_test, _g2, _k2, n_found_te, n_missing_te = _collect_pooled(
                test_tasks, cache_root=cache_root, embed_fn=embed_fn, min_slides=min_slides
            )
            row = {
                "cohort": f"{'+'.join(sorted(train_cohorts))}->{'+'.join(sorted(test_cohorts))}",
                "task": canonical,
                "model": model_name,
                "n_train_found": n_found_tr, "n_train_missing": n_missing_tr,
                "n_test_found": n_found_te, "n_test_missing": n_missing_te,
                "protocol": "cross_cohort_holdout",
            }
            if n_found_tr < min_slides or n_found_te < min_slides:
                row["status"] = "skipped"
                row["reason"] = f"train={n_found_tr} test={n_found_te} slides found (need >= {min_slides} each)"
                print(f"[skip] {row['cohort']}/{canonical} model={model_name}: {row['reason']}")
            elif len(set(y_train)) < 2 or len(set(y_test)) < 2:
                row["status"] = "skipped"
                row["reason"] = "train or test split has only 1 class"
                print(f"[skip] {row['cohort']}/{canonical} model={model_name}: {row['reason']}")
            else:
                metrics = evaluate_train_test(X_train, y_train, X_test, y_test)
                row.update(metrics)
                print(f"[eval] {row['cohort']}/{canonical} model={model_name}: {metrics}")
            results.append(row)
            if use_wandb:
                import wandb

                wandb.log({f"{canonical}/{model_name}/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
    return results


def run_fig2_protocol(
    tasks: list[TaskSpec], *, models: list[tuple[str, object]], train_cohorts: set[str], test_cohorts: set[str],
    teacher_wsi_root: Path, tile_input_root: Path | None, device: torch.device, min_slides: int, use_wandb: bool,
    folds: int, seed: int,
) -> list[dict]:
    """EAGLE's main-benchmark (Fig. 1-5) protocol: 5-fold-ensemble MLP head
    trained on `train_cohorts`, scored once on `test_cohorts`. See
    `evaluate_fig2_protocol` for the fold/ensemble mechanics; this function
    only does task matching and embedding collection, mirroring
    `run_cross_cohort`'s structure."""
    by_canonical: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        if task.members:
            continue
        by_canonical.setdefault(canonical_task_name(task.task), []).append(task)

    results: list[dict] = []
    for canonical, members in sorted(by_canonical.items()):
        train_tasks = [t for t in members if t.cohort in train_cohorts]
        test_tasks = [t for t in members if t.cohort in test_cohorts]
        if not train_tasks or not test_tasks:
            continue
        for model_name, student in models:
            if student is None:
                embed_fn, cache_root = _baseline_embedding, teacher_wsi_root
            else:
                embed_fn = lambda cohort_dir, slide_id, _s=student: _pruned_embedding(
                    cohort_dir, slide_id, student=_s, device=device
                )
                cache_root = tile_input_root

            X_train, y_train, groups_train, _k1, n_found_tr, n_missing_tr = _collect_pooled(
                train_tasks, cache_root=cache_root, embed_fn=embed_fn, min_slides=min_slides
            )
            X_test, y_test, _g2, _k2, n_found_te, n_missing_te = _collect_pooled(
                test_tasks, cache_root=cache_root, embed_fn=embed_fn, min_slides=min_slides
            )
            row = {
                "cohort": f"{'+'.join(sorted(train_cohorts))}->{'+'.join(sorted(test_cohorts))}",
                "task": canonical,
                "model": model_name,
                "n_train_found": n_found_tr, "n_train_missing": n_missing_tr,
                "n_test_found": n_found_te, "n_test_missing": n_missing_te,
                "protocol": "fig2_mlp_ensemble",
            }
            if n_found_tr < min_slides or n_found_te < min_slides:
                row["status"] = "skipped"
                row["reason"] = f"train={n_found_tr} test={n_found_te} slides found (need >= {min_slides} each)"
                print(f"[skip] {row['cohort']}/{canonical} model={model_name}: {row['reason']}")
            elif len(set(y_train)) < 2 or len(set(y_test)) < 2:
                row["status"] = "skipped"
                row["reason"] = "train or test split has only 1 class"
                print(f"[skip] {row['cohort']}/{canonical} model={model_name}: {row['reason']}")
            else:
                metrics = evaluate_fig2_protocol(
                    X_train, y_train, groups_train, X_test, y_test, device=device, folds=folds, seed=seed
                )
                row.update(metrics)
                print(f"[eval] {row['cohort']}/{canonical} model={model_name}: {metrics}")
            results.append(row)
            if use_wandb:
                import wandb

                wandb.log({f"{canonical}/{model_name}/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--data-root", type=Path, default=None, help="Canonical $EAF_WSI_ROOT; defaults to the env var")
    parser.add_argument(
        "--labels-root", type=Path, default=None,
        help="Defaults to <data-root>/datasets/downstream/wsi_level; scanned for <cohort>/labels/*.csv",
    )
    parser.add_argument(
        "--teacher-wsi-root", type=Path, required=True,
        help="Baseline WSI-FM output cache root, one subdirectory per cohort "
        "(<root>/<cohort>/<slide_id>.npyd, eaf.wsi.fm_output.v1 schema; legacy .h5 accepted)",
    )
    parser.add_argument(
        "--tile-input-root", type=Path, default=None,
        help="Tile-EAF cache root feeding --pruned-checkpoint (required if any given), "
        "one subdirectory per cohort matching --teacher-wsi-root's",
    )
    parser.add_argument(
        "--pruned-checkpoint", type=Path, action="append", default=[],
        help="Repeatable: a WSI-EAF Stage-2 best_<run>.pt (distill_wsi_titan.py). "
        "Evaluated in addition to the always-included frozen baseline.",
    )
    parser.add_argument("--min-slides", type=int, default=20, help="Skip a task if fewer labeled+cached slides are found")
    parser.add_argument("--task", action="append", help="Evaluate only this task name; repeat to select more")
    parser.add_argument(
        "--train-cohort", action="append", default=None,
        help="Repeatable. With --test-cohort, switch from within-cohort k-fold CV to "
        "EAGLE-style protocol: fit once on all labeled slides pooled from these cohorts, "
        "evaluate once on --test-cohort. Tasks are matched across cohorts by canonical "
        "biomarker name (see canonical_task_name), since TCGA and CPTAC/Patho-Bench name "
        "the same task differently (e.g. tcga's kras_mutation == cptac's KRAS_mutation).",
    )
    parser.add_argument(
        "--test-cohort", action="append", default=None,
        help="Repeatable. Cohorts held out for testing; see --train-cohort.",
    )
    parser.add_argument(
        "--classifier", choices=("logreg", "mlp"), default="logreg",
        help="With --train-cohort/--test-cohort: 'logreg' fits evaluate_train_test's single "
        "LogisticRegression (default); 'mlp' switches to run_fig2_protocol, EAGLE's main-"
        "benchmark recipe (5-fold-ensemble MLP head, one fold-model per TCGA fold, "
        "predictions averaged on the external test cohort). Ignored in within-cohort CV mode.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Defaults to $EAF_WSI_ROOT/results/wsi_eaf/evaluation/<timestamp>/results.csv",
    )
    parser.add_argument("--wandb-project", default="EAF-WSI-level-LinearProbe")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    args = parser.parse_args()

    experiment_run = prepare_experiment_run(args, family="wsi_eaf", stage="evaluation")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = StoreLayout.from_root(args.data_root)

    labels_root = args.labels_root or (layout.datasets / "downstream" / "wsi_level")
    tasks = discover_tasks(labels_root)
    if args.task:
        tasks = [task for task in tasks if task.task in args.task]
    print(f"discovered {len(tasks)} labeled task(s) under {labels_root}")
    if not tasks:
        print("nothing to evaluate -- no <cohort>/labels/*.csv found yet (expected while TCGA/CPTAC downloads are in progress)")
        return 0

    if args.pruned_checkpoint and args.tile_input_root is None:
        raise SystemExit("--tile-input-root is required when --pruned-checkpoint is given")

    models: list[tuple[str, object]] = [("baseline", None)]
    for ckpt in args.pruned_checkpoint:
        student, meta = _load_pruned_titan(ckpt, device=device, hf_token=args.hf_token)
        models.append((meta["run_name"], student))
        print(f"loaded pruned checkpoint: {meta}")

    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=vars(args))

    if args.train_cohort or args.test_cohort:
        if not (args.train_cohort and args.test_cohort):
            raise SystemExit("--train-cohort and --test-cohort must both be given")
        if args.classifier == "mlp":
            results = run_fig2_protocol(
                tasks, models=models, train_cohorts=set(args.train_cohort), test_cohorts=set(args.test_cohort),
                teacher_wsi_root=args.teacher_wsi_root, tile_input_root=args.tile_input_root, device=device,
                min_slides=args.min_slides, use_wandb=use_wandb, folds=args.folds, seed=args.seed,
            )
        else:
            results = run_cross_cohort(
                tasks, models=models, train_cohorts=set(args.train_cohort), test_cohorts=set(args.test_cohort),
                teacher_wsi_root=args.teacher_wsi_root, tile_input_root=args.tile_input_root, device=device,
                min_slides=args.min_slides, use_wandb=use_wandb,
            )
        _write_results(results, experiment_run, args)
        if use_wandb:
            import wandb

            wandb.finish()
        return 0

    results: list[dict] = []
    for task in tasks:
        prepared = []
        for model_name, student in models:
            if student is None:
                embed_fn = _baseline_embedding
                cache_root = args.teacher_wsi_root
            else:
                embed_fn = lambda cohort_dir, slide_id, _s=student: _pruned_embedding(
                    cohort_dir, slide_id, student=_s, device=device
                )
                cache_root = args.tile_input_root

            X, y, groups, keys, n_found, n_missing = collect_embeddings(
                task, cache_root=cache_root, embed_fn=embed_fn, min_slides=args.min_slides
            )
            prepared.append((model_name, X, y, groups, keys, n_found, n_missing))
        paired = _paired_indices([item[4] for item in prepared])
        reference_labels = None
        reference_groups = None
        for (model_name, X, y, groups, _keys, n_found, n_missing), indices in zip(prepared, paired):
            X, y, groups = X[indices], y[indices], groups[indices]
            if reference_labels is None:
                reference_labels, reference_groups = y, groups
            elif not (np.array_equal(reference_labels, y) and np.array_equal(reference_groups, groups)):
                raise ValueError(f"Paired labels or patient groups disagree for {task.cohort}/{task.task}")
            row = {
                "cohort": task.cohort,
                "task": task.task,
                "model": model_name,
                "n_found": n_found,
                "n_missing": n_missing,
                "n_paired": len(indices),
            }
            if len(indices) < args.min_slides:
                row["status"] = "skipped"
                row["reason"] = f"only {len(indices)} slides shared by all models (need >= {args.min_slides})"
                print(f"[skip] {task.cohort}/{task.task} model={model_name}: {row['reason']}")
            elif len(set(y)) < 2:
                row["status"] = "skipped"
                row["reason"] = f"only 1 class present in {n_found} labeled slides"
                print(f"[skip] {task.cohort}/{task.task} model={model_name}: {row['reason']}")
            else:
                metrics = evaluate_linear_probe(X, y, groups, folds=args.folds, seed=args.seed)
                row.update(metrics)
                print(f"[eval] {task.cohort}/{task.task} model={model_name}: {metrics}")
            results.append(row)
            if use_wandb:
                import wandb

                wandb.log({f"{task.cohort}/{task.task}/{model_name}/{k}": v for k, v in row.items() if isinstance(v, (int, float))})

    _write_results(results, experiment_run, args)

    if use_wandb:
        import wandb

        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
