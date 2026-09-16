#!/usr/bin/env python
"""Paired full-vs-tile_eaf downstream evaluation for one CONCH v1.5 setting.

Implements docs/continuity.md step 3 ("finish paired Tile-EAF evaluation for
the completed CONCH settings, including end-to-end inference latency and
retained-tile ratio") at dev scale: a patient-disjoint linear-probe
comparison on one TCGA task, using the trained forecaster + LoRA-distilled
adapter checkpoints already marked complete in the experiment catalog.

Scope and limits (read before citing a number from this script's output):

* This is the internal TCGA development-scale evaluation described in
  docs/experimental_protocols.md section E01, not the full protocol. CPTAC
  external test, the full candidate endpoint panel, five development
  partitions, and cross-backbone/aggregator comparisons are NOT implemented
  here -- those need CPTAC imagery (E3, external constraint) and much more
  compute to run at the specified panel scale.
* Mean-pooling is the tile-baseline aggregator named in the protocol, used
  here because it needs no additional trained WSI aggregator. It is not the
  TITAN WSI-EAF aggregator path (that is W01/W02, evaluated separately).
* "Retained-tile ratio" here is the forecaster's keep_ratio of ViT PATCH
  TOKENS pruned inside one 512px tile's forward pass (Tile-EAF), not a count
  of which whole tiles get sampled from the WSI -- both arms see the same
  tiles per slide so the comparison isolates the encoder, not the sampler.
* Do not report the resulting AUROC delta as a paper claim without rerunning
  at the full protocol scale (larger --max-slides/--tiles-per-slide, the full
  C-grid via nested CV, and the five development partitions).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thunder.models.pretrained_models import get_model_from_name
from src.data.wsi_tile_stream import SlideRecord, WSITileDataset, inspect_coordinate_file
from src.models import AttentionForecaster, ThunderBackboneAdapter
from src.models.online_tile_eaf import PrunedLoRAEncoder, unwrap_checkpoint_state
from src.utils import set_seed


def _stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _autocast(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5, cwd=repo_root,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, check=True, timeout=5, cwd=repo_root,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"code_revision": revision, "code_dirty": dirty}


def _load_joined_manifest(
    *, slides_csv: Path, coords_csv: Path, labels_csv: Path,
    default_patch_size: int, max_slides: int, seed: int,
) -> tuple[list[SlideRecord], dict[str, int], dict[str, str]]:
    with slides_csv.open(newline="", encoding="utf-8-sig") as handle:
        slide_rows = {row["slide_id"]: row for row in csv.DictReader(handle)}
    with coords_csv.open(newline="", encoding="utf-8-sig") as handle:
        coord_rows = {row["slide_id"]: row for row in csv.DictReader(handle)}
    with labels_csv.open(newline="", encoding="utf-8-sig") as handle:
        label_rows = {row["slide_id"]: row["label"] for row in csv.DictReader(handle)}

    common = sorted(set(slide_rows) & set(coord_rows) & set(label_rows))
    coverage = {
        "labels_total": len(label_rows),
        "slides_total": len(slide_rows),
        "coords_total": len(coord_rows),
        "intersection": len(common),
    }
    # Deterministic, label-stratified cap: interleave classes after a stable
    # per-slide hash ordering so a small --max-slides keeps both classes
    # represented instead of truncating to whichever class sorts first.
    by_label: dict[str, list[str]] = {}
    for slide_id in common:
        by_label.setdefault(label_rows[slide_id], []).append(slide_id)
    for label, ids in by_label.items():
        ids.sort(key=lambda s: _stable_unit_interval(s, seed))
    ordered: list[str] = []
    cursors = {label: 0 for label in by_label}
    while len(ordered) < len(common):
        progressed = False
        for label, ids in by_label.items():
            cursor = cursors[label]
            if cursor < len(ids):
                ordered.append(ids[cursor])
                cursors[label] = cursor + 1
                progressed = True
        if not progressed:
            break
    selected = ordered[:max_slides] if max_slides > 0 else ordered

    records: list[SlideRecord] = []
    skipped: list[str] = []
    labels: dict[str, int] = {}
    for slide_id in selected:
        slide_row = slide_rows[slide_id]
        raw_path = Path(slide_row["wsi_path"]).expanduser().resolve()
        coords_path = Path(coord_rows[slide_id]["path"]).expanduser().resolve()
        if not raw_path.is_file() or not coords_path.is_file():
            skipped.append(slide_id)
            continue
        n_coords, patch_level, patch_size, window = inspect_coordinate_file(
            coords_path, default_patch_size=default_patch_size,
        )
        if n_coords <= 0:
            skipped.append(slide_id)
            continue
        records.append(
            SlideRecord(
                slide_id=slide_id, case_id=slide_row.get("case_id", slide_id),
                cohort="tcga", raw_path=raw_path, coords_path=coords_path, split="",
                coord_count=n_coords, patch_level=patch_level, patch_size=patch_size,
                coordinate_window_size=window,
            )
        )
        labels[slide_id] = int(label_rows[slide_id])
    return records, labels, {"coverage": coverage, "skipped": skipped}


def _coord_indices_for(record: SlideRecord, tiles_per_slide: int) -> list[int]:
    step = max(1, record.coord_count // max(tiles_per_slide, 1))
    return list(range(0, record.coord_count, step))[:tiles_per_slide]


def _read_tiles(record: SlideRecord, transform: Any, *, resize_to: int | None, tiles_per_slide: int) -> torch.Tensor:
    """CPU/IO-only: open one slide, read its scheduled tile crops, return a stacked tensor.

    Runs on a worker thread. OpenSlide's ``read_region`` releases the GIL for
    the actual decode, so several of these overlap real disk/CPU work across
    threads instead of just interleaving Python bytecode.
    """
    coord_indices = _coord_indices_for(record, tiles_per_slide)
    dataset = WSITileDataset(
        [record], transform, augment=False, slide_cache_size=1,
        coordinate_cache_size=1, resize_to=resize_to,
    )
    tiles = [dataset[(0, coord_index)][0] for coord_index in coord_indices]
    return torch.stack(tiles)


def _gpu_forward(
    student: PrunedLoRAEncoder, images_full_batch: torch.Tensor, *,
    batch_size: int, device: torch.device, amp_dtype: str,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """GPU-only: full-forward and pruned-forward embeddings for one slide's tiles."""
    full_chunks: list[torch.Tensor] = []
    eaf_chunks: list[torch.Tensor] = []
    full_ms = 0.0
    eaf_ms = 0.0
    with torch.no_grad(), _autocast(device, amp_dtype):
        for start in range(0, images_full_batch.shape[0], batch_size):
            chunk = images_full_batch[start : start + batch_size].to(device, non_blocking=True)
            _sync(device)
            t0 = time.perf_counter()
            full_out = student.full_teacher_embedding(chunk)
            _sync(device)
            full_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            eaf_out = student(chunk)
            _sync(device)
            eaf_ms += (time.perf_counter() - t0) * 1000.0

            full_chunks.append(full_out.float().cpu())
            eaf_chunks.append(eaf_out.float().cpu())
    full_embedding = torch.cat(full_chunks, dim=0).mean(dim=0).numpy()
    eaf_embedding = torch.cat(eaf_chunks, dim=0).mean(dim=0).numpy()
    return full_embedding, eaf_embedding, full_ms, eaf_ms, images_full_batch.shape[0]


def _fit_eval_arm(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, train_mask: np.ndarray,
    test_mask: np.ndarray, c_grid: list[float], cv_folds: int, seed: int,
) -> dict[str, Any]:
    X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_groups = len(set(groups_train))
    folds = max(2, min(cv_folds, n_groups))
    best_c, best_score = c_grid[0], -np.inf
    if n_groups >= 2 and len(set(y_train)) > 1:
        splitter = GroupKFold(n_splits=folds)
        for c in c_grid:
            fold_scores = []
            for train_idx, val_idx in splitter.split(X_train_scaled, y_train, groups_train):
                if len(set(y_train[train_idx])) < 2 or len(set(y_train[val_idx])) < 2:
                    continue
                model = LogisticRegression(C=c, max_iter=2000, random_state=seed)
                model.fit(X_train_scaled[train_idx], y_train[train_idx])
                probs = model.predict_proba(X_train_scaled[val_idx])[:, 1]
                fold_scores.append(roc_auc_score(y_train[val_idx], probs))
            if fold_scores:
                mean_score = statistics.mean(fold_scores)
                if mean_score > best_score:
                    best_score, best_c = mean_score, c

    final_model = LogisticRegression(C=best_c, max_iter=2000, random_state=seed)
    final_model.fit(X_train_scaled, y_train)
    test_probs = final_model.predict_proba(X_test_scaled)[:, 1]
    test_pred = (test_probs >= 0.5).astype(int)
    return {
        "selected_c": best_c,
        "cv_auroc": None if best_score == -np.inf else best_score,
        "test_auroc": roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else None,
        "test_auprc": average_precision_score(y_test, test_probs) if len(set(y_test)) > 1 else None,
        "test_balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        "test_probs": test_probs.tolist(),
    }


def _bootstrap_patient_auroc_diff(
    probs_a: np.ndarray, probs_b: np.ndarray, y: np.ndarray, case_ids: np.ndarray,
    *, n_reps: int, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_cases = np.unique(case_ids)
    if len(unique_cases) < 2 or len(set(y)) < 2:
        return {"n_reps": 0, "note": "insufficient test cases/classes for bootstrap"}
    diffs: list[float] = []
    for _ in range(n_reps):
        sampled_cases = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        mask_indices = np.concatenate([np.flatnonzero(case_ids == case) for case in sampled_cases])
        y_sample = y[mask_indices]
        if len(set(y_sample)) < 2:
            continue
        auroc_a = roc_auc_score(y_sample, probs_a[mask_indices])
        auroc_b = roc_auc_score(y_sample, probs_b[mask_indices])
        diffs.append(auroc_a - auroc_b)
    if not diffs:
        return {"n_reps": 0, "note": "no valid bootstrap replicate had both classes"}
    diffs_sorted = sorted(diffs)
    lo = diffs_sorted[int(0.025 * (len(diffs_sorted) - 1))]
    hi = diffs_sorted[int(0.975 * (len(diffs_sorted) - 1))]
    return {
        "n_reps": len(diffs),
        "mean_diff_full_minus_eaf": statistics.mean(diffs),
        "ci95_low": lo,
        "ci95_high": hi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired full vs tile_eaf downstream linear probe (dev scale)")
    parser.add_argument("--model-name", default="titan")
    parser.add_argument("--forecaster-ckpt", required=True)
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--prune-layer", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--slides-manifest", required=True, help="e.g. TCGA-BRCA_titan_slides.csv")
    parser.add_argument("--coords-registry", required=True, help="e.g. TCGA-BRCA_titan_coords_registry.csv")
    parser.add_argument("--labels", required=True, help="e.g. TCGA-BRCA/labels/n_status.csv")
    parser.add_argument("--task-name", required=True, help="e.g. tcga_brca/n_status")
    parser.add_argument("--default-patch-size", type=int, default=512)

    parser.add_argument("--max-slides", type=int, default=120, help="Dev-scale cap; 0 = use full intersection")
    parser.add_argument("--tiles-per-slide", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0, help="Unused, kept for CLI compatibility")
    parser.add_argument(
        "--io-workers", type=int, default=8,
        help="Threads reading WSI tiles concurrently, overlapped with GPU forward on other slides",
    )
    parser.add_argument(
        "--prefetch-depth", type=int, default=4,
        help="How many slides' tile reads to keep in flight ahead of the GPU consumer",
    )
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")

    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=17, help="Protocol default split seed")
    parser.add_argument("--seed", type=int, default=42, help="Protocol default EAF confirmation seed")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.001, 0.01, 0.1, 0.5, 1.0, 10.0])
    parser.add_argument("--bootstrap-reps", type=int, default=2000)

    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This evaluation requires a CUDA device")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        args.amp_dtype = "fp16"

    backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    backbone = backbone.to(device)
    adapter = ThunderBackboneAdapter(backbone, transform=transform)
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim, hidden=args.hidden, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    forecaster.load_state_dict(
        unwrap_checkpoint_state(torch.load(args.forecaster_ckpt, map_location="cpu")), strict=True
    )
    student = PrunedLoRAEncoder(
        backbone, adapter, forecaster, prune_layer=args.prune_layer, keep_ratio=args.keep_ratio,
    ).to(device)
    student.eval()
    student.load_trainable_state_dict(
        unwrap_checkpoint_state(torch.load(args.lora_checkpoint, map_location="cpu"))
    )

    records, labels, provenance = _load_joined_manifest(
        slides_csv=Path(args.slides_manifest), coords_csv=Path(args.coords_registry),
        labels_csv=Path(args.labels), default_patch_size=args.default_patch_size,
        max_slides=args.max_slides, seed=args.split_seed,
    )
    if len(records) < 20:
        raise RuntimeError(f"Only {len(records)} usable slides after joining manifests/labels; too few to fit a probe")

    case_ids = np.array([record.case_id for record in records])
    is_test = np.array([
        _stable_unit_interval(f"tcga:{cid}", args.split_seed) < args.test_fraction
        for cid in case_ids
    ])

    full_embeddings, eaf_embeddings = [], []
    full_ms_total, eaf_ms_total, n_tiles_total = 0.0, 0.0, 0
    per_slide_latency_ms: list[float] = []
    encode_failures: list[str] = []

    # Double-buffered pipeline: while the GPU runs full+pruned forward on
    # slide i's already-read tiles, up to --prefetch-depth further slides are
    # being opened/read concurrently on IO threads (OpenSlide releases the
    # GIL during read_region). This is what turned the 26m51s single-threaded
    # 300-slide pilot run into something GPU-bound instead of open()-bound.
    depth = max(1, args.prefetch_depth)
    with ThreadPoolExecutor(max_workers=max(1, args.io_workers)) as io_pool:
        pending: deque[tuple[SlideRecord, Any]] = deque()
        for record in records[:depth]:
            pending.append((record, io_pool.submit(
                _read_tiles, record, transform,
                resize_to=adapter.input_size, tiles_per_slide=args.tiles_per_slide,
            )))
        next_index = depth

        while pending:
            record, future = pending.popleft()
            if next_index < len(records):
                next_record = records[next_index]
                pending.append((next_record, io_pool.submit(
                    _read_tiles, next_record, transform,
                    resize_to=adapter.input_size, tiles_per_slide=args.tiles_per_slide,
                )))
                next_index += 1
            try:
                images_full_batch = future.result()
                full_emb, eaf_emb, full_ms, eaf_ms, n_tiles = _gpu_forward(
                    student, images_full_batch, batch_size=args.batch_size,
                    device=device, amp_dtype=args.amp_dtype,
                )
            except Exception as exc:  # noqa: BLE001 - record and continue, per E01 failure-reporting rule
                encode_failures.append(f"{record.slide_id}: {exc!r}")
                continue
            full_embeddings.append(full_emb)
            eaf_embeddings.append(eaf_emb)
            full_ms_total += full_ms
            eaf_ms_total += eaf_ms
            n_tiles_total += n_tiles
            per_slide_latency_ms.append(eaf_ms)

    kept_slide_ids = [r.slide_id for r in records if not any(r.slide_id in f for f in encode_failures)]
    kept_mask = np.array([sid in set(kept_slide_ids) for sid in [r.slide_id for r in records]])
    y = np.array([labels[r.slide_id] for r, keep in zip(records, kept_mask) if keep])
    groups = case_ids[kept_mask]
    train_mask = ~is_test[kept_mask]
    test_mask = is_test[kept_mask]
    X_full = np.stack(full_embeddings)
    X_eaf = np.stack(eaf_embeddings)

    if train_mask.sum() < 10 or test_mask.sum() < 10:
        raise RuntimeError(
            f"Split too small after encoding: train={train_mask.sum()} test={test_mask.sum()}"
        )

    result_full = _fit_eval_arm(X_full, y, groups, train_mask, test_mask, args.c_grid, args.cv_folds, args.seed)
    result_eaf = _fit_eval_arm(X_eaf, y, groups, train_mask, test_mask, args.c_grid, args.cv_folds, args.seed)

    bootstrap = _bootstrap_patient_auroc_diff(
        np.array(result_full["test_probs"]), np.array(result_eaf["test_probs"]),
        y[test_mask], groups[test_mask], n_reps=args.bootstrap_reps, seed=args.seed,
    )
    result_full.pop("test_probs")
    result_eaf.pop("test_probs")

    run_name = args.run_name or f"{args.task_name.replace('/', '_')}_conch_v15_src{args.prune_layer:02d}_keep{int(round(args.keep_ratio*100))}pct"
    repo_root = Path(__file__).resolve().parents[2]
    summary = {
        "schema": "eaf.paired_downstream_eval.v1",
        "run_name": run_name,
        "task_name": args.task_name,
        "scale": "dev_pilot_not_final_protocol",
        "model_name": args.model_name,
        "prune_layer": args.prune_layer,
        "keep_ratio": args.keep_ratio,
        "retained_tile_token_ratio": args.keep_ratio,
        "forecaster_checkpoint": args.forecaster_ckpt,
        "lora_checkpoint": args.lora_checkpoint,
        "provenance": provenance,
        "n_slides_requested": len(records),
        "n_slides_encoded": int(kept_mask.sum()),
        "encode_failures": encode_failures,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "tiles_per_slide": args.tiles_per_slide,
        "mean_tiles_per_slide": n_tiles_total / max(kept_mask.sum(), 1),
        "latency": {
            "full_forward_ms_per_slide_mean": full_ms_total / max(kept_mask.sum(), 1),
            "eaf_forward_ms_per_slide_mean": eaf_ms_total / max(kept_mask.sum(), 1),
            "eaf_forward_ms_per_slide_median": statistics.median(per_slide_latency_ms) if per_slide_latency_ms else None,
            "speedup_full_over_eaf": full_ms_total / max(eaf_ms_total, 1e-9),
        },
        "full": result_full,
        "tile_eaf": result_eaf,
        "auroc_diff_full_minus_eaf_bootstrap": bootstrap,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        **_git_state(repo_root),
    }

    root_default = Path("/data2/home/vcivale/data/WSI") / "results" / "wsi_eaf" / "evaluation" if not args.output else None
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root_default / f"{run_name}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Summary written to {output_path}")
    print(f"Slides: requested={len(records)} encoded={int(kept_mask.sum())} failures={len(encode_failures)}")
    print(f"Train/test: {int(train_mask.sum())}/{int(test_mask.sum())}")
    print(f"FULL   AUROC={result_full['test_auroc']} AUPRC={result_full['test_auprc']}")
    print(f"TILE_EAF AUROC={result_eaf['test_auroc']} AUPRC={result_eaf['test_auprc']}")
    print(f"AUROC diff (full - eaf), bootstrap 95% CI: {bootstrap}")
    print(f"Latency speedup (full/eaf forward): {summary['latency']['speedup_full_over_eaf']:.2f}x")


if __name__ == "__main__":
    main()
