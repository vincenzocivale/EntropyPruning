import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

import h5py

from src.data.wsi import H5WSIFeatureStore


def _write_h5_dataset(path: Path, name: str, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(name, data=tensor.numpy())


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["slide_id", "features_path", "coords_path", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_import_trident_feature_store_cli_imports_manifest(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    feature_dir = tmp_path / "trident_processed/20x_256px_0px_overlap/features_uni_v1"
    coord_dir = tmp_path / "trident_processed/20x_256px_0px_overlap/patches"

    _write_h5_dataset(
        feature_dir / "slide_001.h5",
        "features",
        torch.randn(4, 8),
    )
    _write_h5_dataset(
        coord_dir / "slide_001.h5",
        "coords",
        torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long),
    )

    _write_h5_dataset(
        feature_dir / "slide_002.h5",
        "features",
        torch.randn(3, 8),
    )
    _write_h5_dataset(
        coord_dir / "slide_002.h5",
        "coords",
        torch.tensor([[0, 0], [1, 0], [2, 0]], dtype=torch.long),
    )

    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "trident_processed/20x_256px_0px_overlap/features_uni_v1/slide_001.h5",
                "coords_path": "trident_processed/20x_256px_0px_overlap/patches/slide_001.h5",
                "label": "0",
            },
            {
                "slide_id": "slide_002",
                "features_path": "trident_processed/20x_256px_0px_overlap/features_uni_v1/slide_002.h5",
                "coords_path": "trident_processed/20x_256px_0px_overlap/patches/slide_002.h5",
                "label": "1",
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_trident_feature_store.py",
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_path),
            "--feature-dim",
            "8",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"event": "done"' in result.stdout

    store = H5WSIFeatureStore(output_path)
    assert store.slide_ids() == ("slide_001", "slide_002")

    bag = store.read("slide_001")
    assert bag.tile_features.shape == (4, 8)
    assert bag.coords is not None
    assert bag.coords.shape == (4, 2)
    assert bag.label == 0
    assert bag.metadata is not None
    assert bag.metadata["source"] == "trident"
    assert "trident_features_path" in bag.metadata
    assert "trident_coords_path" in bag.metadata


def test_import_trident_feature_store_cli_supports_explicit_dataset_names(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    _write_h5_dataset(tmp_path / "features/slide_001.h5", "my_features", torch.randn(4, 8))
    _write_h5_dataset(
        tmp_path / "coords/slide_001.h5",
        "my_coords",
        torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long),
    )

    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "features/slide_001.h5",
                "coords_path": "coords/slide_001.h5",
                "label": "1",
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/import_trident_feature_store.py",
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_path),
            "--feature-dim",
            "8",
            "--feature-dataset",
            "my_features",
            "--coords-dataset",
            "my_coords",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    store = H5WSIFeatureStore(output_path)
    bag = store.read("slide_001")
    assert bag.tile_features.shape == (4, 8)
    assert bag.coords is not None
    assert bag.coords.shape == (4, 2)
    assert bag.label == 1


def test_import_trident_feature_store_cli_squeezes_singleton_multilayer_features(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    _write_h5_dataset(
        tmp_path / "features/slide_001.h5",
        "features",
        torch.randn(4, 1, 8),
    )
    _write_h5_dataset(
        tmp_path / "coords/slide_001.h5",
        "coords",
        torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long),
    )

    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "features/slide_001.h5",
                "coords_path": "coords/slide_001.h5",
                "label": "1",
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/import_trident_feature_store.py",
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_path),
            "--feature-dim",
            "8",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    store = H5WSIFeatureStore(output_path)
    bag = store.read("slide_001")
    assert bag.tile_features.shape == (4, 8)
    assert bag.coords is not None
    assert bag.coords.shape == (4, 2)


def test_import_trident_feature_store_cli_rejects_feature_dim_mismatch(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    _write_h5_dataset(tmp_path / "features/slide_001.h5", "features", torch.randn(4, 7))
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "features/slide_001.h5",
                "coords_path": "",
                "label": "",
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_trident_feature_store.py",
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_path),
            "--feature-dim",
            "8",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "expected 8" in result.stderr


def test_import_trident_feature_store_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    _write_h5_dataset(tmp_path / "features/slide_001.h5", "features", torch.randn(4, 8))
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "features/slide_001.h5",
                "coords_path": "",
                "label": "",
            },
        ],
    )
    output_path.write_text("already here")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_trident_feature_store.py",
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output feature store already exists" in result.stderr


def test_import_trident_feature_store_cli_output_passes_validator(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    _write_h5_dataset(tmp_path / "features/slide_001.h5", "features", torch.randn(4, 8))
    _write_h5_dataset(
        tmp_path / "coords/slide_001.h5",
        "coords",
        torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long),
    )
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "features/slide_001.h5",
                "coords_path": "coords/slide_001.h5",
                "label": "0",
            },
        ],
    )

    import_result = subprocess.run(
        [
            sys.executable,
            "scripts/import_trident_feature_store.py",
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_path),
            "--feature-dim",
            "8",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr

    validate_result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_wsi_feature_store.py",
            "--feature-store",
            str(output_path),
            "--feature-dim",
            "8",
            "--require-coords",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert validate_result.returncode == 0, validate_result.stderr
    summary = json.loads(validate_result.stdout)
    assert summary["valid"] is True
    assert summary["n_slides"] == 1
