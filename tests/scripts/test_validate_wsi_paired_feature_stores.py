import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag


def _bag(
    slide_id: str,
    *,
    n_tiles: int = 4,
    feature_dim: int = 8,
    coords: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=coords,
        label=1,
        attention=attention,
        metadata={"source": "synthetic"},
    )


def _run_validator(*args: str, repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/validate_wsi_paired_feature_stores.py", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def test_validate_wsi_paired_feature_stores_accepts_legacy_single_store(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = tmp_path / "features.h5"

    coords = torch.zeros(4, 2, dtype=torch.long)
    store = H5WSIFeatureStore(path)
    store.write(_bag("slide_001", coords=coords, attention=torch.rand(4) + 0.1))
    store.write(_bag("slide_002", coords=coords, attention=torch.rand(4) + 0.1))

    result = _run_validator(
        "--input-feature-store",
        str(path),
        "--target-feature-store",
        str(path),
        "--input-feature-dim",
        "8",
        "--alignment-mode",
        "index",
        "--require-coords",
        "--require-attention",
        repo_root=repo_root,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)

    assert summary["valid"] is True
    assert summary["n_slides"] == 2
    assert summary["n_tiles_min"] == 4
    assert summary["n_tiles_max"] == 4
    assert summary["target_coverage"] == 1.0
    assert summary["coords_coverage"] == 1.0
    assert summary["n_errors"] == 0


def test_validate_wsi_paired_feature_stores_accepts_index_aligned_pair(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = tmp_path / "input.h5"
    target_path = tmp_path / "target.h5"
    coords = torch.zeros(4, 2, dtype=torch.long)

    input_store = H5WSIFeatureStore(input_path)
    input_store.write(_bag("slide_001", feature_dim=16, coords=coords))

    target_store = H5WSIFeatureStore(target_path)
    target_store.write(
        _bag("slide_001", feature_dim=32, coords=coords, attention=torch.rand(4) + 0.1)
    )

    result = _run_validator(
        "--input-feature-store",
        str(input_path),
        "--target-feature-store",
        str(target_path),
        "--input-feature-dim",
        "16",
        "--alignment-mode",
        "index",
        repo_root=repo_root,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)

    assert summary["valid"] is True
    assert summary["input_feature_dims"] == [16]


def test_validate_wsi_paired_feature_stores_rejects_tile_count_mismatch(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = tmp_path / "input.h5"
    target_path = tmp_path / "target.h5"

    input_store = H5WSIFeatureStore(input_path)
    input_store.write(_bag("slide_001", n_tiles=4))

    target_store = H5WSIFeatureStore(target_path)
    target_store.write(_bag("slide_001", n_tiles=5, attention=torch.rand(5) + 0.1))

    result = _run_validator(
        "--input-feature-store",
        str(input_path),
        "--target-feature-store",
        str(target_path),
        "--alignment-mode",
        "index",
        repo_root=repo_root,
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)

    assert summary["valid"] is False
    assert any("pairing failed" in entry["error"] for entry in summary["mismatch_examples"])


def test_validate_wsi_paired_feature_stores_rejects_missing_target_slide(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = tmp_path / "input.h5"
    target_path = tmp_path / "target.h5"

    input_store = H5WSIFeatureStore(input_path)
    input_store.write(_bag("slide_001"))
    input_store.write(_bag("slide_002"))

    target_store = H5WSIFeatureStore(target_path)
    target_store.write(_bag("slide_001", attention=torch.rand(4) + 0.1))

    result = _run_validator(
        "--input-feature-store",
        str(input_path),
        "--target-feature-store",
        str(target_path),
        repo_root=repo_root,
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)

    assert summary["valid"] is False
    assert summary["n_slides_only_in_input"] == 1
    assert any(
        "missing from target store" in entry["error"] for entry in summary["mismatch_examples"]
    )


def test_validate_wsi_paired_feature_stores_rejects_missing_target_attention(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = tmp_path / "input.h5"
    target_path = tmp_path / "target.h5"

    input_store = H5WSIFeatureStore(input_path)
    input_store.write(_bag("slide_001"))

    target_store = H5WSIFeatureStore(target_path)
    target_store.write(_bag("slide_001", attention=None))

    result = _run_validator(
        "--input-feature-store",
        str(input_path),
        "--target-feature-store",
        str(target_path),
        "--require-attention",
        repo_root=repo_root,
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)

    assert summary["valid"] is False
    assert summary["target_coverage"] == 0.0
