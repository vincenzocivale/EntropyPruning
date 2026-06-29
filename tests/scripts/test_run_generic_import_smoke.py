import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("h5py")


def test_run_generic_import_smoke_script(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workdir = tmp_path / "generic_smoke"

    env = os.environ.copy()
    env.update(
        {
            "WORKDIR": str(workdir),
            "FEATURE_DIM": "8",
            "N_SLIDES": "5",
            "MIN_TILES": "4",
            "SEED": "123",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_generic_import_smoke.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "[generic-smoke] done" in result.stdout

    expected_files = [
        workdir / "labels.csv",
        workdir / "manifest_generic.csv",
        workdir / "features_generic_eaf.h5",
        workdir / "features/slide_000.pt",
        workdir / "features/slide_001.npy",
        workdir / "features/slide_002.npz",
        workdir / "coords/slide_000.pt",
        workdir / "coords/slide_001.npy",
        workdir / "coords/slide_002.npz",
    ]

    for path in expected_files:
        assert path.exists(), path
        assert path.stat().st_size > 0, path
