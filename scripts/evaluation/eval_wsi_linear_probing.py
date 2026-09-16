#!/usr/bin/env python
"""Linear-probing evaluation of WSI-FM slide embeddings on labeled downstream
datasets from the configured downstream benchmark bank (see docs/pipeline.md).

Compares the frozen baseline WSI-FM (TITAN) against one or more WSI-EAF
Stage-2 checkpoints (`scripts/training/finetune_wsi_titan_pruned.py`,
`checkpoints/wsi_eaf_pruned/<pair>/<run>/`) by training a plain linear head
(logistic regression, k-fold cross-validated) on each model's slide
embeddings for each labeled task. No backbone/forecaster weights are updated
here -- linear probing only, mirroring the tile-EAF Stage-3 evaluation
(`scripts/training/train_multi_thunder_classifier.py --adaptation linear_probing`) one
level up.

Two embedding sources per task:

- Baseline (frozen TITAN): read the pre-cached `slide_embedding` straight out
  of the `eaf.wsi.fm_output.v1` cache (`scripts/features/wsi_eaf_infer_wsi_fm.py`
  output, `--wsi-eaf-root`) -- no model forward pass at all.
- Each `--pruned-checkpoint`: read per-slide `coords`/`tile_embeddings` from
  the matching Tile-EAF cache (`--tile-eaf-root`) and run them live through
  `PrunedLoRATitanEncoder` (forecaster-guided pruning + LoRA), exactly as
  `finetune_wsi_titan_pruned.py` does during training -- there is no
  precomputed pruned slide_embedding cache, pruning is cheap enough to run at
  eval time and this avoids maintaining yet another cache namespace per
  checkpoint.

Task/dataset discovery is automatic (`discover_tasks`): any
`<labels-root>/<cohort>/labels/<task>.csv` (two columns: `slide_id,label`) is
a candidate task, but a task is only evaluated once matching embeddings are
actually found for at least `--min-slides` slides on both sides (label +
embedding) -- so this script runs correctly today against a mostly-empty
downstream tree (reports 0 usable tasks) and picks up more cohorts
automatically as EAGLE/Patho-Bench downloads complete, with no code change.
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
from src.models.wsi.dense_forecaster import WSIDenseForecaster, WSIDenseForecasterALiBi  # noqa: E402
from src.models.wsi.pruned_titan import PrunedLoRATitanEncoder  # noqa: E402
from src.utils import set_seed  # noqa: E402
from src.wsi_pipeline.io import read_wsi_output_record  # noqa: E402
from src.wsi_pipeline.wsi_models.titan import TitanAdapter  # noqa: E402


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


def _load_wsi_forecaster(checkpoint_path: Path, *, device: torch.device):
    """Mirrors finetune_wsi_titan_pruned.py::_load_forecaster exactly (kept as an
    independent copy since scripts/ is not an importable package)."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fargs = payload["args"]
    common = dict(embed_dim=768, hidden=fargs["hidden"], n_heads=fargs["n_heads"], n_layers=fargs["n_layers"], dropout=0.0)
    if fargs.get("architecture") == "dense_alibi":
        forecaster = WSIDenseForecasterALiBi(**common)
    else:
        forecaster = WSIDenseForecaster(**common)
    forecaster.load_state_dict(payload["model"])
    forecaster = forecaster.to(device).eval()
    source_layer = fargs.get("titan_hidden_layer")
    if source_layer is None:
        raise ValueError(f"{checkpoint_path} was not trained with --input-source titan_hidden")
    return forecaster, int(source_layer)


def _load_pruned_titan(
    pruned_checkpoint: Path, *, device: torch.device, hf_token: str | None
) -> tuple[PrunedLoRATitanEncoder, dict]:
    payload = torch.load(pruned_checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("args", {})
    forecaster_ckpt = config.get("forecaster_checkpoint")
    if not forecaster_ckpt:
        raise ValueError(f"{pruned_checkpoint} does not record a forecaster_checkpoint")
    forecaster, prune_layer = _load_wsi_forecaster(Path(forecaster_ckpt), device=device)
    titan_model = TitanAdapter(token=hf_token).model
    student = PrunedLoRATitanEncoder(
        titan_model,
        forecaster,
        prune_layer=int(config.get("prune_layer", prune_layer)),
        keep_ratio=float(config["keep_ratio"]),
        patch_size_level0=int(config.get("patch_size_level0", 512)),
        lora_r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
    ).to(device)
    student.load_trainable_state_dict(payload["model"])
    student.eval()
    meta = {
        "run_name": payload.get("run_name", pruned_checkpoint.parent.name),
        "prune_layer": student.prune_layer,
        "keep_ratio": student.keep_ratio,
    }
    return student, meta


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Canonical $EAF_WSI_ROOT; defaults to the env var")
    parser.add_argument(
        "--labels-root", type=Path, default=None,
        help="Defaults to <data-root>/datasets/downstream/wsi_level; scanned for <cohort>/labels/*.csv",
    )
    parser.add_argument(
        "--wsi-eaf-root", type=Path, required=True,
        help="Baseline WSI-FM output cache root, one subdirectory per cohort "
        "(<root>/<cohort>/<slide_id>.npyd, eaf.wsi.fm_output.v1 schema; legacy .h5 accepted)",
    )
    parser.add_argument(
        "--tile-eaf-root", type=Path, default=None,
        help="Tile-EAF cache root feeding --pruned-checkpoint (required if any given), "
        "one subdirectory per cohort matching --wsi-eaf-root's",
    )
    parser.add_argument(
        "--pruned-checkpoint", type=Path, action="append", default=[],
        help="Repeatable: a WSI-EAF Stage-2 best_<run>.pt (finetune_wsi_titan_pruned.py). "
        "Evaluated in addition to the always-included frozen baseline.",
    )
    parser.add_argument("--min-slides", type=int, default=20, help="Skip a task if fewer labeled+cached slides are found")
    parser.add_argument("--task", action="append", help="Evaluate only this task name; repeat to select more")
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

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = StoreLayout.from_root(args.data_root)

    labels_root = args.labels_root or (layout.datasets / "downstream" / "wsi_level")
    tasks = discover_tasks(labels_root)
    if args.task:
        tasks = [task for task in tasks if task.task in args.task]
    print(f"discovered {len(tasks)} labeled task(s) under {labels_root}")
    if not tasks:
        print("nothing to evaluate -- no <cohort>/labels/*.csv found yet (expected while EAGLE downloads are in progress)")
        return 0

    if args.pruned_checkpoint and args.tile_eaf_root is None:
        raise SystemExit("--tile-eaf-root is required when --pruned-checkpoint is given")

    models: list[tuple[str, object]] = [("baseline", None)]
    for ckpt in args.pruned_checkpoint:
        student, meta = _load_pruned_titan(ckpt, device=device, hf_token=args.hf_token)
        models.append((meta["run_name"], student))
        print(f"loaded pruned checkpoint: {meta}")

    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=vars(args))

    results: list[dict] = []
    for task in tasks:
        prepared = []
        for model_name, student in models:
            if student is None:
                embed_fn = _baseline_embedding
                cache_root = args.wsi_eaf_root
            else:
                embed_fn = lambda cohort_dir, slide_id, _s=student: _pruned_embedding(
                    cohort_dir, slide_id, student=_s, device=device
                )
                cache_root = args.tile_eaf_root

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

    run_name = str(int(time.time()))
    output_csv = args.output_csv or (layout.results / "wsi_eaf" / "evaluation" / run_name / "results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in results for key in row})
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} rows to {output_csv}")
    from src.wsi_pipeline.experiment_results import publish_run_summary
    publish_run_summary(
        family="wsi_eaf", stage="evaluation", run_name=run_name, args=args,
        summary={"output_csv": str(output_csv), "result_rows": len(results)},
    )

    if use_wandb:
        import wandb

        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
