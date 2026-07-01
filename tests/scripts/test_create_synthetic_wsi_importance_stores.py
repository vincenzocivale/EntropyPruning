import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore


def _run_script(*args: str, repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/create_synthetic_wsi_importance_stores.py", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def _default_args(tmp_path: Path, **overrides: str) -> list[str]:
    args = {
        "--early-output": str(tmp_path / "early.h5"),
        "--late-output": str(tmp_path / "late.h5"),
        "--importance-output": str(tmp_path / "importance.h5"),
        "--n-slides": "6",
        "--early-feature-dim": "8",
        "--late-feature-dim": "16",
        "--min-tiles": "4",
        "--max-tiles": "7",
        "--n-classes": "2",
        "--noise-std": "0.1",
        "--seed": "123",
    }
    args.update(overrides)
    flat: list[str] = []
    for key, value in args.items():
        flat.extend([key, value])
    return flat


def test_create_synthetic_wsi_importance_stores_cli_creates_aligned_stores(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = _run_script(*_default_args(tmp_path), repo_root=repo_root)
    assert result.returncode == 0, result.stderr

    summary = json.loads(result.stdout)
    assert summary["n_slides"] == 6
    assert summary["early_feature_dim"] == 8
    assert summary["late_feature_dim"] == 16
    assert 4 <= summary["min_tiles"] <= summary["max_tiles"] <= 7

    early_store = H5WSIFeatureStore(tmp_path / "early.h5")
    late_store = H5WSIFeatureStore(tmp_path / "late.h5")
    importance_store = H5WSIFeatureStore(tmp_path / "importance.h5")

    assert early_store.slide_ids() == late_store.slide_ids() == importance_store.slide_ids()
    assert len(early_store.slide_ids()) == 6

    for slide_id in early_store.slide_ids():
        early_bag = early_store.read(slide_id)
        late_bag = late_store.read(slide_id)
        importance_bag = importance_store.read(slide_id)

        assert early_bag.tile_features.shape[1] == 8
        assert late_bag.tile_features.shape[1] == 16
        assert importance_bag.tile_features.shape[1] == 1

        assert early_bag.n_tiles == late_bag.n_tiles == importance_bag.n_tiles
        assert torch.equal(early_bag.coords, late_bag.coords)
        assert torch.equal(early_bag.coords, importance_bag.coords)
        assert early_bag.label == late_bag.label == importance_bag.label

        assert importance_bag.attention is not None
        assert importance_bag.attention.shape == (importance_bag.n_tiles,)
        assert torch.isfinite(importance_bag.attention).all()
        assert (importance_bag.attention >= 0).all()
        assert torch.allclose(
            importance_bag.attention.sum(), torch.tensor(1.0), atol=1e-5
        )
        assert importance_bag.metadata["target_source"] == "synthetic_importance_from_late"
        assert importance_bag.metadata["target_type"] == "tile_importance"


def test_create_synthetic_wsi_importance_stores_target_correlates_with_late_features(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = _run_script(
        *_default_args(tmp_path, **{"--noise-std": "0.0", "--min-tiles": "16", "--max-tiles": "16"}),
        repo_root=repo_root,
    )
    assert result.returncode == 0, result.stderr

    late_store = H5WSIFeatureStore(tmp_path / "late.h5")
    importance_store = H5WSIFeatureStore(tmp_path / "importance.h5")

    direction = torch.linspace(-1.0, 1.0, 16)
    for slide_id in late_store.slide_ids():
        late_bag = late_store.read(slide_id)
        importance_bag = importance_store.read(slide_id)

        expected = torch.softmax(late_bag.tile_features @ direction, dim=0)
        assert torch.allclose(importance_bag.attention, expected, atol=1e-5)


def test_create_synthetic_wsi_importance_stores_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "early.h5").write_text("already here")

    result = _run_script(*_default_args(tmp_path), repo_root=repo_root)

    assert result.returncode != 0
    assert "already exist" in result.stderr


def test_create_synthetic_wsi_importance_stores_cli_overwrites_when_requested(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "early.h5").write_text("already here")

    result = _run_script(
        *_default_args(tmp_path), "--overwrite", repo_root=repo_root
    )

    assert result.returncode == 0, result.stderr


def test_create_synthetic_wsi_importance_stores_cli_rejects_duplicate_output_paths(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    shared = str(tmp_path / "shared.h5")

    result = _run_script(
        *_default_args(tmp_path, **{"--late-output": shared, "--early-output": shared}),
        repo_root=repo_root,
    )

    assert result.returncode != 0
    assert "distinct paths" in result.stderr


def test_create_synthetic_wsi_importance_stores_output_passes_paired_validator(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    create_result = _run_script(*_default_args(tmp_path), repo_root=repo_root)
    assert create_result.returncode == 0, create_result.stderr

    validate_result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_wsi_paired_feature_stores.py",
            "--input-feature-store",
            str(tmp_path / "early.h5"),
            "--target-feature-store",
            str(tmp_path / "importance.h5"),
            "--input-feature-dim",
            "8",
            "--alignment-mode",
            "coords",
            "--require-coords",
            "--require-attention",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert validate_result.returncode == 0, validate_result.stderr
    summary = json.loads(validate_result.stdout)
    assert summary["valid"] is True
    assert summary["n_slides"] == 6
