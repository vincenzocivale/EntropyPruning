import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag


def _make_bag(
    slide_id: str,
    n_tiles: int = 4,
    feature_dim: int = 8,
    *,
    with_attention: bool = True,
    with_coords: bool = True,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim)
    attention = torch.rand(n_tiles) + 0.1 if with_attention else None
    coords = torch.zeros(n_tiles, 2, dtype=torch.long) if with_coords else None

    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=coords,
        label=1,
        attention=attention,
        metadata={"source": "synthetic"},
    )


def _run_validator(*args: str, repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/validate_wsi_feature_store.py", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def test_validate_wsi_feature_store_cli_accepts_valid_store(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = tmp_path / "features.h5"

    store = H5WSIFeatureStore(path)
    store.write(_make_bag("slide_001", n_tiles=4, feature_dim=8))
    store.write(_make_bag("slide_002", n_tiles=7, feature_dim=8))

    result = _run_validator(
        "--feature-store",
        str(path),
        "--feature-dim",
        "8",
        "--require-attention",
        "--require-coords",
        repo_root=repo_root,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)

    assert summary["valid"] is True
    assert summary["n_slides"] == 2
    assert summary["n_tiles_total"] == 11
    assert summary["min_tiles"] == 4
    assert summary["max_tiles"] == 7
    assert summary["feature_dims"] == [8]
    assert summary["n_with_attention"] == 2
    assert summary["n_with_coords"] == 2
    assert summary["n_errors"] == 0


def test_validate_wsi_feature_store_cli_rejects_missing_attention_when_required(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = tmp_path / "features.h5"

    store = H5WSIFeatureStore(path)
    store.write(_make_bag("slide_001", with_attention=False))

    result = _run_validator(
        "--feature-store",
        str(path),
        "--feature-dim",
        "8",
        "--require-attention",
        repo_root=repo_root,
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)

    assert summary["valid"] is False
    assert summary["n_errors"] == 1
    assert summary["errors"][0]["slide_id"] == "slide_001"
    assert "attention is required" in summary["errors"][0]["error"]


def test_validate_wsi_feature_store_cli_rejects_feature_dim_mismatch(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = tmp_path / "features.h5"

    store = H5WSIFeatureStore(path)
    store.write(_make_bag("slide_001", feature_dim=8))

    result = _run_validator(
        "--feature-store",
        str(path),
        "--feature-dim",
        "16",
        repo_root=repo_root,
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)

    assert summary["valid"] is False
    assert summary["n_errors"] == 1
    assert "feature_dim mismatch" in summary["errors"][0]["error"]


def test_validate_wsi_feature_store_cli_rejects_empty_store(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = tmp_path / "empty.h5"
    H5WSIFeatureStore(path)

    result = _run_validator(
        "--feature-store",
        str(path),
        repo_root=repo_root,
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)

    assert summary["valid"] is False
    assert summary["n_slides"] == 0
    assert summary["n_errors"] == 1
    assert "no slides" in summary["errors"][0]["error"]
