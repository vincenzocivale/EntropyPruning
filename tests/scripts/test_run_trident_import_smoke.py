import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("h5py")


def test_run_trident_import_smoke_script(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workdir = tmp_path / "trident_smoke"

    env = os.environ.copy()
    env.update(
        {
            "WORKDIR": str(workdir),
            "FEATURE_DIM": "8",
            "N_SLIDES": "3",
            "MIN_TILES": "4",
            "SEED": "123",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_trident_import_smoke.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "[trident-smoke] done" in result.stdout

    expected_files = [
        workdir / "labels.csv",
        workdir / "manifest_trident.csv",
        workdir / "features_trident_eaf.h5",
        workdir / "trident_processed/20x_256px_0px_overlap/features_uni_v1/slide_000.h5",
        workdir / "trident_processed/20x_256px_0px_overlap/patches/slide_000.h5",
    ]

    for path in expected_files:
        assert path.exists(), path
        assert path.stat().st_size > 0, path
