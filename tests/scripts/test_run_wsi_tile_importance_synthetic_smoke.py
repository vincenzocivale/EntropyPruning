import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("h5py")


def test_run_wsi_tile_importance_synthetic_smoke_script(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workdir = tmp_path / "wsi_importance_smoke"

    env = os.environ.copy()
    env.update(
        {
            "WORKDIR": str(workdir),
            "EARLY_FEATURE_DIM": "8",
            "LATE_FEATURE_DIM": "16",
            "HIDDEN_DIM": "16",
            "N_SLIDES": "8",
            "MIN_TILES": "4",
            "MAX_TILES": "8",
            "N_CLASSES": "2",
            "NOISE_STD": "0.1",
            "BATCH_SIZE": "2",
            "FORECASTER_EPOCHS": "1",
            "KEEP_RATIOS": "0.10 1.0",
            "PRUNED_KEEP_RATIO": "0.10",
            "DEVICE": "cpu",
            "SEED": "123",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_wsi_tile_importance_synthetic_smoke.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "synthetic tile-importance smoke failed\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )

    assert "[wsi-importance-smoke] done" in result.stdout

    expected_files = [
        workdir / "features_layer2.h5",
        workdir / "features_late.h5",
        workdir / "features_synthetic_importance.h5",
        workdir / "features_late_pruned_keep_0.10.h5",
        workdir / "splits/train.txt",
        workdir / "splits/val.txt",
        workdir / "splits/test.txt",
        workdir / "splits/split_summary.json",
        workdir / "checkpoints/importance_forecaster/best_wsi_tile_importance_forecaster.pt",
        workdir / "checkpoints/importance_forecaster/training_summary.json",
        workdir / "results/wsi_importance_pruning.csv",
        workdir / "reports/paired_validation.json",
        workdir / "reports/pruned_store_validation.json",
    ]

    for path in expected_files:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_run_wsi_tile_importance_synthetic_smoke_script_produces_valid_paired_report(
    tmp_path,
) -> None:
    import json

    repo_root = Path(__file__).resolve().parents[2]
    workdir = tmp_path / "wsi_importance_smoke_report"

    env = os.environ.copy()
    env.update(
        {
            "WORKDIR": str(workdir),
            "N_SLIDES": "6",
            "MIN_TILES": "4",
            "MAX_TILES": "6",
            "FORECASTER_EPOCHS": "1",
            "BATCH_SIZE": "2",
            "KEEP_RATIOS": "1.0",
            "PRUNED_KEEP_RATIO": "0.5",
            "DEVICE": "cpu",
            "SEED": "7",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_wsi_tile_importance_synthetic_smoke.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads((workdir / "reports" / "paired_validation.json").read_text())
    assert report["valid"] is True
    assert report["n_slides"] == 6
    assert report["target_coverage"] == 1.0
    assert report["coords_coverage"] == 1.0

    pruned_report = json.loads((workdir / "reports" / "pruned_store_validation.json").read_text())
    assert pruned_report["valid"] is True
