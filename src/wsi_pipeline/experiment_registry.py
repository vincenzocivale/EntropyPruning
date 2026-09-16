"""Versioned experiment registry and deterministic runtime paths."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_RELATIVE_PATH = Path("configs/experiments/registry.toml")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_registry_path() -> Path:
    return repository_root() / REGISTRY_RELATIVE_PATH


def load_registry(path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    registry_path = Path(path).expanduser().resolve() if path else default_registry_path()
    raw = registry_path.read_bytes()
    registry = tomllib.loads(raw.decode("utf-8"))
    if int(registry.get("version", 0)) != 1:
        raise ValueError(f"Unsupported experiment registry version in {registry_path}")
    return registry, registry_path, hashlib.sha256(raw).hexdigest()


def _data_root(args: Any) -> Path:
    value = getattr(args, "data_root", None) or os.environ.get("EAF_WSI_ROOT")
    if not value:
        raise RuntimeError(
            "Canonical experiment storage requires $EAF_WSI_ROOT or --data-root. "
            "Paper runs must not use repository-relative outputs."
        )
    return Path(value).expanduser().resolve()


def _git_state() -> tuple[str | None, bool | None]:
    root = repository_root()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, text=True, capture_output=True, check=True, timeout=5,
        ).stdout.strip())
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _normal(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, float):
        return round(value, 12)
    return value


@dataclass(frozen=True)
class ExperimentRun:
    experiment_id: str
    variant_id: str
    family: str
    stage: str
    seed: int
    dataset_id: str | None
    tile_encoder: str | None
    wsi_encoder: str | None
    data_root: Path
    checkpoint_dir: Path
    result_dir: Path
    log_dir: Path
    derived_cache_dir: Path
    registry_path: Path
    registry_sha256: str
    declaration: dict[str, Any]
    variant: dict[str, Any]

    @property
    def run_name(self) -> str:
        return f"{self.experiment_id}__{self.variant_id}__seed{self.seed}"

    @property
    def run_key(self) -> str:
        return f"{self.experiment_id}/{self.variant_id}/seed_{self.seed}"

    @property
    def tile_pruned_cache_dir(self) -> Path:
        return self.derived_cache_dir / "tile_pruned"

    @property
    def wsi_source_cache_dir(self) -> Path:
        return self.derived_cache_dir / "wsi_source"


def add_experiment_arguments(parser: Any) -> None:
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--experiment-registry", type=Path, default=None)


def _validate_model_identity(entry: dict[str, Any], config: dict[str, Any]) -> None:
    declared_tile = entry.get("tile_encoder")
    actual_tile = config.get("tile_encoder") or config.get("model_name")
    if declared_tile and actual_tile and str(declared_tile) != str(actual_tile):
        compatible = {("conch_v15", "titan"), ("titan", "conch_v15")}
        if (str(declared_tile), str(actual_tile)) not in compatible:
            raise ValueError(
                f"Registry declares tile_encoder={declared_tile!r}, but CLI requests {actual_tile!r}"
            )
    declared_wsi = entry.get("wsi_encoder")
    actual_wsi = config.get("wsi_encoder")
    if declared_wsi and actual_wsi and str(declared_wsi) != str(actual_wsi):
        raise ValueError(
            f"Registry declares wsi_encoder={declared_wsi!r}, but CLI requests {actual_wsi!r}"
        )


def _validate_variant(variant: dict[str, Any], config: dict[str, Any]) -> None:
    mismatches: list[str] = []
    for key, expected in variant.items():
        if key in {"description", "status", "notes"} or key not in config or config[key] is None:
            continue
        if _normal(config[key]) != _normal(expected):
            mismatches.append(f"{key}: registry={expected!r}, cli={config[key]!r}")
    if mismatches:
        raise ValueError(
            "CLI/config does not match the predeclared experiment variant:\n  "
            + "\n  ".join(mismatches)
        )


def prepare_experiment_run(args: Any, *, family: str, stage: str) -> ExperimentRun:
    registry, registry_path, digest = load_registry(getattr(args, "experiment_registry", None))
    experiments = registry.get("experiments", {})
    experiment_id = str(args.experiment_id)
    variant_id = str(args.variant_id)
    if experiment_id not in experiments:
        raise KeyError(
            f"Unknown experiment_id={experiment_id!r}. Declare it in {registry_path} and commit first."
        )
    entry = dict(experiments[experiment_id])
    status = str(entry.get("status", "blocked"))
    if status != "ready":
        raise RuntimeError(
            f"Experiment {experiment_id!r} is status={status!r}; refusing launch. "
            f"Blockers: {entry.get('blockers', [])}"
        )
    if entry.get("family") != family or entry.get("stage") != stage:
        raise ValueError(
            f"{experiment_id!r} is declared as {entry.get('family')}/{entry.get('stage')}, "
            f"not {family}/{stage}"
        )
    variants = entry.get("variants", {})
    if variant_id not in variants:
        raise KeyError(f"Unknown variant_id={variant_id!r}; declared={sorted(variants)}")
    variant = dict(variants[variant_id])
    if variant.get("status", "ready") != "ready":
        raise RuntimeError(f"Variant {experiment_id}/{variant_id} is not ready")

    config = vars(args).copy() if hasattr(args, "__dict__") else dict(args)
    _validate_model_identity(entry, config)
    _validate_variant(variant, config)

    root = _data_root(args)
    seed = int(config.get("seed", 0))
    suffix = Path(experiment_id) / variant_id / f"seed_{seed}"
    run = ExperimentRun(
        experiment_id=experiment_id,
        variant_id=variant_id,
        family=family,
        stage=stage,
        seed=seed,
        dataset_id=entry.get("dataset_id"),
        tile_encoder=entry.get("tile_encoder"),
        wsi_encoder=entry.get("wsi_encoder"),
        data_root=root,
        checkpoint_dir=root / "checkpoints" / family / stage / suffix,
        result_dir=root / "results" / family / stage / suffix,
        log_dir=root / "logs" / family / stage / suffix,
        derived_cache_dir=root / "caches" / "experiments" / suffix,
        registry_path=registry_path,
        registry_sha256=digest,
        declaration=entry,
        variant=variant,
    )

    requested_name = config.get("run_name")
    if requested_name not in (None, "", run.run_name):
        raise ValueError(f"--run-name is no longer free-form. Canonical name: {run.run_name}")
    requested_output = config.get("output_dir")
    if requested_output:
        requested = Path(requested_output).expanduser().resolve()
        if requested != run.checkpoint_dir.resolve():
            raise ValueError(f"--output-dir must equal canonical path: {run.checkpoint_dir}")

    for path in (run.checkpoint_dir, run.result_dir, run.log_dir, run.derived_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    revision, dirty = _git_state()
    record = {
        "schema": "eaf.experiment_run.v2",
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
        "registry": {"path": str(run.registry_path), "sha256": run.registry_sha256},
        "code_revision": revision,
        "code_dirty": dirty,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "checkpoint_dir": str(run.checkpoint_dir),
            "result_dir": str(run.result_dir),
            "log_dir": str(run.log_dir),
            "derived_cache_dir": str(run.derived_cache_dir),
            "tile_pruned_cache_dir": str(run.tile_pruned_cache_dir),
            "wsi_source_cache_dir": str(run.wsi_source_cache_dir),
        },
        "declared_experiment": {k: v for k, v in entry.items() if k != "variants"},
        "declared_variant": variant,
        "config": config,
    }
    manifest = run.result_dir / "run.json"
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(manifest)
    return run
