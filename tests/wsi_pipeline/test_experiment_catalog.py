"""Experiment inventory contracts."""

import json

from src.wsi_pipeline.experiment_catalog import audit_experiments, write_catalog


def test_audit_canonical_legacy_and_broken_references(tmp_path):
    data = tmp_path / "data"
    repo = tmp_path / "repo"
    current = data / "checkpoints" / "tile_eaf" / "conch_v15" / "run"
    current.mkdir(parents=True)
    (current / "best_run.pt").write_bytes(b"opaque checkpoint")
    (current / "summary_run.json").write_text(json.dumps({
        "run_name": "run", "checkpoint": str(current / "best_run.pt"),
        "forecaster_checkpoint": "/missing/forecaster.pt", "best_val_kl": 0.1,
        "history": [{"train": {"tiles": 100, "tiles_per_second": 10},
                     "val": {"rho": 0.8}}],
    }))
    old = repo / "checkpoints" / "wsi_eaf" / "hidden_layer0_final"
    old.mkdir(parents=True)
    (old / "best_wsi_landmark_forecaster.pt").write_bytes(b"opaque")
    labels = data / "datasets" / "downstream" / "wsi_level" / "cohort" / "labels"
    labels.mkdir(parents=True)
    (labels / "task.csv").write_text("slide_id,label\na,1\n")
    (repo / "eagle_baseline_results.csv").write_text("task,status\na,ok\n")
    result_csv = data / "results" / "wsi_eaf" / "evaluation" / "run" / "results.csv"
    result_csv.parent.mkdir(parents=True)
    result_csv.write_text("task,model,status,cv_unit\na,baseline,ok,patient\n")

    catalog = audit_experiments(data, repo)
    assert catalog["counts"]["runs"] == 2
    assert catalog["counts"]["label_files"] == 1
    assert catalog["counts"]["baseline_rows"] == 1
    assert any(table["cv_units"] == ["patient"] for table in catalog["result_tables"])
    assert catalog["runs"][0]["status"] == "partial"
    assert catalog["runs"][0]["missing"] == ["forecaster_checkpoint"]
    assert catalog["runs"][0]["metrics"]["mean_train_epoch_seconds"] == 10
    assert catalog["matrix"][0]["stages"]["quality"]["status"] == "partial"
    assert catalog["runs"][1]["origin"] == "legacy_repo"
    assert "summary" in catalog["runs"][1]["missing"]
    output = data / "results" / "experiment_catalog"
    path = write_catalog(catalog, output)
    first = path.read_bytes()
    write_catalog(audit_experiments(data, repo), output)
    assert path.read_bytes() == first
