"""Read-only discovery of current and legacy EAF experiment artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

MODELS = ("conch_v15", "uni2h", "virchow2", "hoptimus1", "provgigapath")
STAGES = ("cache", "forecaster", "distillation", "evaluation", "quality", "cost")


def _files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern)) if root.is_dir() else []


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"parse_error": "not an object"}
    except (OSError, ValueError) as exc:
        return {"parse_error": str(exc)}


def _resolve_reference(value: str, *, summary: Path, roots: tuple[Path, ...]) -> Path | None:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [summary.parent / path, *(root / path for root in roots)]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _model(name: str) -> str | None:
    normalized = name.lower().replace("-", "").replace("_", "")
    if "conch" in normalized or normalized.startswith("titan_src"):
        return "conch_v15"
    return next((model for model in MODELS[1:] if model.replace("_", "") in normalized), None)


def _normalized(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "")


def _stage(path: Path) -> str:
    parts = set(path.parts)
    if "pruned_finetuned" in parts or "wsi_eaf_pruned" in parts or "pruned_titan_layer0" in parts:
        return "distillation"
    if "multi_thunder" in parts:
        return "evaluation"
    return "forecaster"


def audit_experiments(data_root: Path, repo_root: Path) -> dict[str, Any]:
    """Inventory metadata and paths without loading checkpoint tensors or cache payloads."""
    data_root, repo_root = data_root.resolve(), repo_root.resolve()
    roots = (data_root, repo_root)
    runs: list[dict[str, Any]] = []
    for origin, root in (("canonical", data_root), ("legacy_repo", repo_root)):
        grouped: dict[Path, dict[str, list[Path]]] = {}
        for path in _files(root / "checkpoints", "**/*"):
            if path.is_file() and (path.suffix == ".pt" or path.name.startswith("summary_") and path.suffix == ".json" or path.name == "results.json"):
                bucket = grouped.setdefault(path.parent, {"checkpoints": [], "summaries": []})
                bucket["checkpoints" if path.suffix == ".pt" else "summaries"].append(path)
        for directory, group in sorted(grouped.items()):
            summaries = [_json(path) for path in group["summaries"]]
            summary = next((item for item in summaries if "parse_error" not in item), summaries[0] if summaries else {})
            best = [p for p in group["checkpoints"] if p.name.startswith("best")]
            references: dict[str, str] = {}
            missing: list[str] = []
            config = summary.get("config", summary.get("args", {}))
            if not isinstance(config, dict):
                config = {}
            for key in ("checkpoint", "forecaster_checkpoint", "forecaster_ckpt", "tile_eaf_root", "wsi_eaf_root", "target_cache_index"):
                value = summary.get(key) or config.get(key)
                if isinstance(value, str) and value:
                    resolved = _resolve_reference(value, summary=group["summaries"][0], roots=roots)
                    references[key] = str(resolved) if resolved else value
                    if resolved is None:
                        missing.append(key)
            if not group["summaries"]:
                missing.append("summary")
            if not best:
                missing.append("best_checkpoint")
            if "parse_error" in summary:
                missing.append("summary_parse_error")
            name = str(summary.get("run_name") or directory.name)
            model = _model(str(summary.get("tile_encoder") or config.get("model_name") or directory))
            if model is None and "wsi_eaf" in directory.parts:
                model = "titan"
            metrics = {key: value for key, value in summary.items() if key.startswith(("best_", "final_val_")) and isinstance(value, (int, float))}
            history = summary.get("history")
            if isinstance(history, list):
                val_rho = [row.get("val", {}).get("rho") for row in history if isinstance(row, dict)]
                val_rho = [float(value) for value in val_rho if isinstance(value, (int, float))]
                if val_rho:
                    metrics["max_val_rho"] = max(val_rho)
                epoch_seconds = [
                    float(row["train"]["tiles"]) / float(row["train"]["tiles_per_second"])
                    for row in history if isinstance(row, dict) and isinstance(row.get("train"), dict)
                    and row["train"].get("tiles_per_second", 0) > 0
                    and isinstance(row["train"].get("tiles"), (int, float))
                ]
                if epoch_seconds:
                    metrics["mean_train_epoch_seconds"] = sum(epoch_seconds) / len(epoch_seconds)
            complete_epochs = summary.get("epochs_completed")
            planned_epochs = config.get("epochs")
            if isinstance(complete_epochs, int) and isinstance(planned_epochs, int) and complete_epochs < planned_epochs and not summary.get("stopped_early"):
                missing.append("planned_epochs")
            status = "non_comparable" if "CONTAMINATED" in str(directory).upper() else "complete" if not missing else "partial" if best else "missing_data"
            runs.append({"id": f"{origin}:{directory.relative_to(root)}", "name": name,
                         "origin": origin, "directory": str(directory), "model": model,
                         "stage": _stage(directory), "status": status,
                         "checkpoints": [str(path) for path in group["checkpoints"]],
                         "summaries": [str(path) for path in group["summaries"]],
                         "config": config, "metrics": metrics, "references": references,
                         "missing": sorted(set(missing))})
    labels = _files(data_root / "datasets" / "downstream", "**/labels/*.csv")
    cache_metadata = _files(data_root / "caches", "**/metadata.json") + _files(data_root / "caches", "**/manifest.csv")
    cache_metadata += _files(data_root / "datasets", "**/tile_cache_index.csv")
    results = _files(data_root / "results", "**/*.csv") + _files(data_root / "results", "**/results.json")
    results += _files(data_root / "results", "**/summary.json")
    results = [path for path in results if "experiment_catalog" not in path.parts]
    logs = _files(data_root / "logs", "**/*.log") + _files(repo_root / "logs", "**/*.log")
    profiles = _files(data_root / "logs", "**/*profile*.json")
    for run in runs:
        run["logs"] = [str(path) for path in logs if run["name"] in path.name]
    baseline = repo_root / "eagle_baseline_results.csv"
    if baseline.is_file():
        results.append(baseline)
    baseline_rows = 0
    if baseline.is_file():
        with baseline.open(newline="", encoding="utf-8") as handle:
            baseline_rows = sum(1 for _ in csv.DictReader(handle))
    result_tables: list[dict[str, Any]] = []
    for path in sorted(set(results)):
        if path.suffix != ".csv":
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error):
            continue
        result_tables.append({
            "path": str(path), "rows": len(rows),
            "statuses": dict(sorted(Counter(row.get("status", "") for row in rows).items())),
            "models": sorted({row["model"] for row in rows if row.get("model")}),
            "cv_units": sorted({row["cv_unit"] for row in rows if row.get("cv_unit")}),
        })
    matrix = []
    for family, models in (("tile", MODELS), ("wsi", ("titan",))):
        for model in models:
            matched = [run for run in runs if run["model"] == model and ("/wsi_eaf/" in run["id"] or "/wsi_eaf_pruned/" in run["id"] if family == "wsi" else "/wsi_eaf/" not in run["id"] and "/wsi_eaf_pruned/" not in run["id"])]
            cache_count = sum(_normalized(model) in _normalized(str(path)) for path in cache_metadata)
            cells = {stage: {"status": "missing_data", "evidence": []} for stage in STAGES}
            if cache_count:
                cells["cache"] = {"status": "present", "evidence": [str(p) for p in cache_metadata if _normalized(model) in _normalized(str(p))][:10]}
            for stage in ("forecaster", "distillation", "evaluation"):
                selected = [run for run in matched if run["stage"] == stage]
                if selected:
                    cells[stage] = {"status": "complete" if any(run["status"] == "complete" for run in selected) else "partial", "evidence": [run["id"] for run in selected]}
            quality = [run for run in matched if "max_val_rho" in run["metrics"]]
            if quality:
                cells["quality"] = {"status": "partial", "evidence": [run["id"] for run in quality]}
            cost = [run for run in matched if "mean_train_epoch_seconds" in run["metrics"]]
            model_profiles = [path for path in profiles if _normalized(model) in _normalized(str(path))]
            if cost or model_profiles:
                cells["cost"] = {"status": "partial", "evidence": [run["id"] for run in cost] + [str(path) for path in model_profiles]}
            if family == "wsi" and labels:
                cells["evaluation"]["labels_available"] = len(labels)
                evaluation_tables = [table for table in result_tables if "wsi_eaf/evaluation" in table["path"]]
                if (baseline_rows or evaluation_tables) and cells["evaluation"]["status"] == "missing_data":
                    cells["evaluation"] = {"status": "partial", "evidence": [table["path"] for table in evaluation_tables] + ([str(baseline)] if baseline_rows else []),
                                             "labels_available": len(labels), "baseline_rows": baseline_rows}
            matrix.append({"family": family, "model": model, "stages": cells})
    return {"schema": "eaf.experiment_catalog.v1", "roots": {"canonical": str(data_root), "legacy_repo": str(repo_root)},
            "counts": {"runs": len(runs), "status": dict(sorted(Counter(run["status"] for run in runs).items())),
                       "label_files": len(labels), "cache_metadata_files": len(cache_metadata), "result_files": len(results), "log_files": len(logs), "profile_files": len(profiles), "baseline_rows": baseline_rows},
            "runs": runs, "labels": [str(path) for path in labels], "caches": [str(path) for path in cache_metadata],
            "results": [str(path) for path in sorted(results)], "result_tables": result_tables,
            "logs": [str(path) for path in logs],
            "profiles": [str(path) for path in profiles], "matrix": matrix}


def write_catalog(catalog: dict[str, Any], output_dir: Path) -> Path:
    """Write a deterministic index; rerunning with unchanged sources yields identical bytes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "catalog.json"
    temporary = output_dir / "catalog.json.tmp"
    temporary.write_text(json.dumps(catalog, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
