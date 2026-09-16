"""Result metadata is saved separately from model weights."""

import argparse
import json

from src.wsi_pipeline.experiment_registry import ExperimentRun
from src.wsi_pipeline.experiment_results import publish_run_summary


def _make_run(tmp_path) -> ExperimentRun:
    root = tmp_path
    suffix = "exp1/variantA/seed_17"
    return ExperimentRun(
        experiment_id="exp1",
        variant_id="variantA",
        family="tile_eaf",
        stage="forecaster",
        seed=17,
        dataset_id="histai_core_v1",
        tile_encoder="conch_v15",
        wsi_encoder=None,
        data_root=root,
        checkpoint_dir=root / "checkpoints" / "tile_eaf" / "forecaster" / suffix,
        result_dir=root / "results" / "tile_eaf" / "forecaster" / suffix,
        log_dir=root / "logs" / "tile_eaf" / "forecaster" / suffix,
        derived_cache_dir=root / "caches" / "experiments" / suffix,
        registry_path=tmp_path / "registry.toml",
        registry_sha256="deadbeef",
        declaration={},
        variant={},
    )


def test_publish_run_summary_uses_canonical_results(tmp_path, monkeypatch):
    monkeypatch.setenv("EAF_WSI_ROOT", str(tmp_path))
    run = _make_run(tmp_path)
    run.result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    path = publish_run_summary(
        run=run,
        args=argparse.Namespace(model_name="conch_v15", source_layer=2, seed=17,
                                target_cache_index=tmp_path / "cache.csv", epochs=20),
        summary={"checkpoint": str(checkpoint), "best_val_kl": 0.1,
                 "epochs_completed": 20},
    )
    assert path == run.result_dir / "summary.json"
    record = json.loads(path.read_text())
    assert record["summary"] == {"checkpoint": str(checkpoint), "best_val_kl": 0.1,
                                  "epochs_completed": 20}
    assert record["family"] == "tile_eaf"
    assert record["stage"] == "forecaster"
    assert record["experiment_id"] == "exp1"
    assert record["variant_id"] == "variantA"
    assert record["artifacts"]["checkpoint"] == str(checkpoint)
