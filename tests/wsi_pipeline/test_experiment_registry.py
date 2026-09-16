from pathlib import Path
from types import SimpleNamespace

import pytest

from src.wsi_pipeline.experiment_registry import prepare_experiment_run


def _registry(path: Path, status: str = "ready") -> None:
    path.write_text(
        f"""
version = 1

[experiments.demo]
family = "tile_eaf"
stage = "forecaster"
status = "{status}"
dataset_id = "histai_core_v1"
tile_encoder = "conch_v15"

[experiments.demo.variants.src02]
source_layer = 2
model_name = "conch_v15"
""".strip() + "\n"
    )


def _args(registry: Path, source_layer: int = 2):
    return SimpleNamespace(
        experiment_id="demo", variant_id="src02", experiment_registry=registry,
        seed=17, source_layer=source_layer, model_name="conch_v15", data_root=None,
        run_name=None, output_dir=None,
    )


def test_deterministic_paths(tmp_path, monkeypatch):
    registry = tmp_path / "registry.toml"
    _registry(registry)
    root = tmp_path / "data"
    monkeypatch.setenv("EAF_WSI_ROOT", str(root))
    run = prepare_experiment_run(_args(registry), family="tile_eaf", stage="forecaster")
    assert run.run_name == "demo__src02__seed17"
    assert run.checkpoint_dir == root / "checkpoints/tile_eaf/forecaster/demo/src02/seed_17"
    assert run.result_dir == root / "results/tile_eaf/forecaster/demo/src02/seed_17"
    assert (run.result_dir / "run.json").is_file()


def test_blocked_refuses_launch(tmp_path, monkeypatch):
    registry = tmp_path / "registry.toml"
    _registry(registry, status="blocked")
    monkeypatch.setenv("EAF_WSI_ROOT", str(tmp_path / "data"))
    with pytest.raises(RuntimeError, match="blocked"):
        prepare_experiment_run(_args(registry), family="tile_eaf", stage="forecaster")


def test_variant_mismatch_refuses_launch(tmp_path, monkeypatch):
    registry = tmp_path / "registry.toml"
    _registry(registry)
    monkeypatch.setenv("EAF_WSI_ROOT", str(tmp_path / "data"))
    with pytest.raises(ValueError, match="does not match"):
        prepare_experiment_run(_args(registry, source_layer=3), family="tile_eaf", stage="forecaster")
