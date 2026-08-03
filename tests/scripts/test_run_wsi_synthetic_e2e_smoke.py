import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("h5py")
pytest.importorskip("matplotlib")


def test_run_wsi_synthetic_e2e_smoke_script(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workdir = tmp_path / "wsi_e2e"

    env = os.environ.copy()
    env.update(
        {
            "WORKDIR": str(workdir),
            "FEATURE_DIM": "8",
            "HIDDEN_DIM": "16",
            "N_SLIDES": "6",
            "MIN_TILES": "4",
            "MAX_TILES": "6",
            "N_CLASSES": "2",
            "BATCH_SIZE": "2",
            "ABMIL_EPOCHS": "1",
            "FORECASTER_EPOCHS": "1",
            "KEEP_RATIOS": "0.5 1.0",
            "PRUNED_KEEP_RATIO": "0.5",
            "DEVICE": "cpu",
            "SEED": "123",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_wsi_synthetic_e2e_smoke.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "synthetic E2E smoke failed\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )

    assert "[wsi-e2e] done" in result.stdout

    expected_files = [
        workdir / "features_raw.h5",
        workdir / "features_abmil_attention.h5",
        workdir / "features_pruned_keep_0.5.h5",
        workdir / "splits/train.txt",
        workdir / "splits/val.txt",
        workdir / "splits/test.txt",
        workdir / "splits/split_summary.json",
        workdir / "checkpoints/abmil/best_abmil_classifier.pt",
        workdir / "checkpoints/forecaster/best_wsi_tile_attention_forecaster.pt",
        workdir / "results/wsi_forecaster_pruning.csv",
        workdir / "results/wsi_abmil_pruning_agreement.csv",
        workdir / "reports/pruning/summary.json",
        workdir / "reports/pruning/attention_mass_retained.png",
        workdir / "reports/pruning/prediction_agreement.png",
    ]

    for path in expected_files:
        assert path.exists(), path
        assert path.stat().st_size > 0, path
