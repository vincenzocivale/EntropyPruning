"""Canonical result records for registry-declared EAF experiments."""
from __future__ import annotations

import json
from typing import Any

from .experiment_registry import ExperimentRun

PROVENANCE_INPUT_KEYS = (
    "manifest",
    "wsi_manifest",
    "target_cache_index",
    "tile_cache_dir",
    "tile_input_root",
    "source_wsi_root",
    "teacher_wsi_root",
    "labels_root",
    "base_data_folder",
)
PROVENANCE_CHECKPOINT_KEYS = (
    "teacher_checkpoint",
    "forecaster_ckpt",
    "forecaster_checkpoint",
    "pruned_checkpoint",
    "resume",
)


def publish_run_summary(*, run: ExperimentRun, args: Any, summary: dict[str, Any]):
    config = vars(args).copy() if hasattr(args, "__dict__") else dict(args)
    payload = dict(summary)
    history = payload.pop("history", None)
    artifacts: dict[str, str] = {}

    if isinstance(history, list):
        history_path = run.result_dir / "history.json"
        tmp = history_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(history, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        tmp.replace(history_path)
        artifacts["history"] = str(history_path)

    for key in ("checkpoint", "output_csv", "profile_json"):
        value = payload.get(key) or config.get(key)
        if value:
            artifacts[key] = str(value)

    record = {
        "schema": "eaf.experiment_result.v2",
        "experiment_id": run.experiment_id,
        "variant_id": run.variant_id,
        "run_key": run.run_key,
        "run_name": run.run_name,
        "family": run.family,
        "stage": run.stage,
        "seed": run.seed,
        "dataset_id": run.dataset_id,
        "tile_encoder": run.tile_encoder,
        "wsi_encoder": run.wsi_encoder,
        "registry_sha256": run.registry_sha256,
        "inputs": {
            key: str(config[key])
            for key in PROVENANCE_INPUT_KEYS
            if config.get(key) not in (None, "")
        },
        "input_checkpoints": {
            key: ([str(item) for item in config[key]] if isinstance(config[key], (list, tuple)) else str(config[key]))
            for key in PROVENANCE_CHECKPOINT_KEYS
            if config.get(key) not in (None, "", [])
        },
        "artifacts": artifacts,
        "summary": payload,
        "config": config,
    }
    path = run.result_dir / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
