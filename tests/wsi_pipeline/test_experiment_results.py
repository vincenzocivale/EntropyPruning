"""Result metadata is saved separately from model weights."""

import argparse
import json

from src.wsi_pipeline.experiment_results import publish_run_summary


def test_publish_run_summary_uses_canonical_results(tmp_path, monkeypatch):
    monkeypatch.setenv("EAF_WSI_ROOT", str(tmp_path))
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    path = publish_run_summary(
        family="tile_eaf", stage="forecaster", run_name="run",
        args=argparse.Namespace(model_name="conch_v15", source_layer=2, seed=17,
                                target_cache_index=tmp_path / "cache.csv", epochs=20),
        summary={"checkpoint": str(checkpoint), "best_val_kl": 0.1,
                 "epochs_completed": 20},
    )
    assert path == tmp_path / "results" / "tile_eaf" / "forecaster" / "run" / "summary.json"
    record = json.loads(path.read_text())
    assert record["metrics"] == {"best_val_kl": 0.1}
    assert record["early_layer"] == 2
    assert record["checkpoint"] == str(checkpoint)
    assert isinstance(record["code_dirty"], bool)
