"""Uniform small metadata records for newly completed EAF runs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def result_root() -> Path | None:
    """Return the canonical results root when configured."""
    root = os.environ.get("EAF_WSI_ROOT")
    return Path(root).expanduser().resolve() / "results" if root else None


def publish_run_summary(
    *, family: str, stage: str, run_name: str, args: Any, summary: dict[str, Any],
) -> Path | None:
    """Mirror lightweight run metadata under results; leave checkpoints in place."""
    config = vars(args).copy() if hasattr(args, "__dict__") else dict(args)
    root = result_root()
    if root is None and config.get("data_root"):
        root = Path(config["data_root"]).expanduser().resolve() / "results"
    if root is None:
        return None
    try:
        repo_root = Path(__file__).resolve().parents[2]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5, cwd=repo_root,
        ).stdout.strip()
        code_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, check=True, timeout=5, cwd=repo_root,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision = None
        code_dirty = None
    history = summary.get("history")
    latest_epoch = history[-1] if isinstance(history, list) and history else {}
    timings = summary.get("profile") or summary.get("timings")
    if timings is None:
        timings = {key: value for key, value in latest_epoch.items() if key in ("train", "val")}
        timings = timings or {key: summary[key] for key in ("train_metrics", "val_metrics") if key in summary}
    record = {
        "schema": "eaf.experiment_result.v1", "family": family, "stage": stage,
        "run_name": run_name, "model": summary.get("tile_encoder") or config.get("model_name"),
        "model_revision": summary.get("model_revision") or config.get("model_revision"),
        "dataset": config.get("dataset") or config.get("manifest") or config.get("wsi_manifest") or config.get("tile_eaf_root"),
        "split": {key: config.get(key) for key in ("split_seed", "val_fraction", "val_seed")},
        "input_cache": {key: str(config[key]) for key in ("target_cache_index", "tile_eaf_root", "wsi_eaf_root") if config.get(key)},
        "early_layer": summary.get("prune_layer") if stage == "distillation" else config.get("source_layer") if config.get("source_layer") is not None else config.get("titan_hidden_layer"),
        "keep_ratio": config.get("keep_ratio"), "seed": config.get("seed"),
        "code_revision": revision, "code_dirty": code_dirty,
        "checkpoint": summary.get("checkpoint"),
        "metrics": {key: value for key, value in summary.items() if key.startswith(("best_", "final_val_")) and isinstance(value, (int, float, dict))},
        "artifacts": {key: str(summary[key]) for key in ("output_csv",) if summary.get(key)},
        "timings": timings,
        "epochs_completed": summary.get("epochs_completed"),
        "config": config,
    }
    path = root / family / stage / run_name / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
