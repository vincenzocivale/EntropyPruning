import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("torch")


def _run_python(code: str, *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_run_wsi_unsupervised_fm_experiment_template_smoke(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workdir = tmp_path / "unsup_fm"
    early_dir = workdir / "early"
    late_dir = workdir / "late"
    target_dir = workdir / "targets"
    coords_dir = workdir / "coords"
    output_root = workdir / "output"

    env = os.environ.copy()
    env["WORKDIR"] = str(workdir)

    _run_python(
        """
from pathlib import Path
import numpy as np
import torch

workdir = Path(r"%s")
early_dir = workdir / "early"
late_dir = workdir / "late"
target_dir = workdir / "targets"
coords_dir = workdir / "coords"

for path in (early_dir, late_dir, target_dir, coords_dir):
    path.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
for index in range(6):
    slide_id = f"slide_{index:03d}"
    n_tiles = 5 + (index %% 2)
    coords = np.stack([np.arange(n_tiles), np.full(n_tiles, index)], axis=1).astype(np.int64)
    early = rng.normal(size=(n_tiles, 8)).astype(np.float32)
    late = rng.normal(size=(n_tiles, 12)).astype(np.float32)
    raw_scores = late[:, 0] + 0.25 * late[:, 1]
    target = np.exp(raw_scores - raw_scores.max()).astype(np.float32)

    torch.save(torch.from_numpy(early), early_dir / f"{slide_id}.pt")
    np.save(late_dir / f"{slide_id}.npy", late)
    np.save(target_dir / f"{slide_id}.npy", target)
    np.save(coords_dir / f"{slide_id}.npy", coords)
"""
        % str(workdir),
        cwd=repo_root,
        env=env,
    )

    run_env = os.environ.copy()
    run_env.update(
        {
            "PYTHON_BIN": sys.executable,
            "EARLY_FEATURES_DIR": str(early_dir),
            "LATE_FEATURES_DIR": str(late_dir),
            "TARGETS_DIR": str(target_dir),
            "COORDS_DIR": str(coords_dir),
            "OUTPUT_ROOT": str(output_root),
            "INPUT_FEATURE_DIM": "8",
            "LATE_FEATURE_DIM": "12",
            "TARGET_SOURCE": "synthetic_wsi_fm",
            "TARGET_NORMALIZE": "sum",
            "DEVICE": "cpu",
            "EPOCHS": "1",
            "BATCH_SIZE": "2",
            "TOP_K": "2",
            "KEEP_RATIOS": "0.5 1.0",
            "PRUNED_KEEP_RATIO": "0.5",
            "OVERWRITE": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_wsi_unsupervised_fm_experiment_template.sh"],
        cwd=repo_root,
        env=run_env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "[wsi-unsupervised-template] done" in result.stdout
    assert (output_root / "stores" / "features_layer2.h5").exists()
    assert (output_root / "stores" / "features_late.h5").exists()
    assert (output_root / "stores" / "features_wsi_importance.h5").exists()
    assert (output_root / "splits" / "train.txt").exists()
    assert (
        output_root
        / "checkpoints"
        / "exp001_importance_forecaster"
        / "best_wsi_tile_importance_forecaster.pt"
    ).exists()
    assert (output_root / "results" / "exp001_importance_pruning.csv").exists()
    assert (output_root / "features_late_pruned_keep_0.5.h5").exists()
