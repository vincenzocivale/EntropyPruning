import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore


def _run_script(*args: str, repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/create_synthetic_wsi_feature_store.py", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def test_create_synthetic_wsi_feature_store_cli_creates_valid_store(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "synthetic.h5"

    result = _run_script(
        "--output",
        str(output),
        "--n-slides",
        "6",
        "--feature-dim",
        "8",
        "--min-tiles",
        "4",
        "--max-tiles",
        "7",
        "--n-classes",
        "3",
        "--seed",
        "123",
        repo_root=repo_root,
    )

    assert result.returncode == 0, result.stderr

    summary = json.loads(result.stdout)
    assert summary["output"] == str(output)
    assert summary["n_slides"] == 6
    assert summary["feature_dim"] == 8
    assert 4 <= summary["min_tiles"] <= summary["max_tiles"] <= 7
    assert output.exists()

    store = H5WSIFeatureStore(output)
    assert len(store.slide_ids()) == 6

    first = store.read("synthetic_slide_00000")
    assert first.tile_features.shape[1] == 8
    assert first.attention is not None
    assert first.attention.shape == (first.n_tiles,)
    assert first.coords is not None
    assert first.coords.shape == (first.n_tiles, 2)
    assert first.label == 0
    assert first.metadata is not None
    assert first.metadata["source"] == "synthetic"


def test_create_synthetic_wsi_feature_store_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "synthetic.h5"
    output.write_text("already here")

    result = _run_script(
        "--output",
        str(output),
        repo_root=repo_root,
    )

    assert result.returncode != 0
    assert "output already exists" in result.stderr


def test_create_synthetic_wsi_feature_store_cli_overwrites_when_requested(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "synthetic.h5"
    output.write_text("already here")

    result = _run_script(
        "--output",
        str(output),
        "--n-slides",
        "2",
        "--feature-dim",
        "4",
        "--min-tiles",
        "3",
        "--max-tiles",
        "3",
        "--overwrite",
        repo_root=repo_root,
    )

    assert result.returncode == 0, result.stderr

    store = H5WSIFeatureStore(output)
    assert store.slide_ids() == ("synthetic_slide_00000", "synthetic_slide_00001")


def test_create_synthetic_wsi_feature_store_cli_output_passes_validator(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "synthetic.h5"

    create_result = _run_script(
        "--output",
        str(output),
        "--n-slides",
        "4",
        "--feature-dim",
        "8",
        "--min-tiles",
        "3",
        "--max-tiles",
        "5",
        repo_root=repo_root,
    )
    assert create_result.returncode == 0, create_result.stderr

    validate_result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_wsi_feature_store.py",
            "--feature-store",
            str(output),
            "--feature-dim",
            "8",
            "--require-attention",
            "--require-coords",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert validate_result.returncode == 0, validate_result.stderr
    summary = json.loads(validate_result.stdout)
    assert summary["valid"] is True
    assert summary["n_slides"] == 4
