import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.wsi import H5WSIFeatureStore


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["slide_id", "features_path", "coords_path", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_import_generic_feature_store_cli_imports_pt_npy_and_npz(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    torch.save(torch.randn(4, 8), tmp_path / "slide_001.pt")
    np.save(tmp_path / "slide_002.npy", np.random.randn(3, 8).astype("float32"))
    np.savez(tmp_path / "slide_003.npz", features=np.random.randn(5, 8).astype("float32"))

    torch.save(torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]), tmp_path / "slide_001_coords.pt")
    np.save(tmp_path / "slide_002_coords.npy", np.array([[0, 0], [1, 0], [2, 0]]))
    np.savez(
        tmp_path / "slide_003_coords.npz",
        coords=np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),
    )

    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "slide_001.pt",
                "coords_path": "slide_001_coords.pt",
                "label": "0",
            },
            {
                "slide_id": "slide_002",
                "features_path": "slide_002.npy",
                "coords_path": "slide_002_coords.npy",
                "label": "1",
            },
            {
                "slide_id": "slide_003",
                "features_path": "slide_003.npz",
                "coords_path": "slide_003_coords.npz",
                "label": "2",
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_generic_feature_store.py",
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
    assert store.slide_ids() == ("slide_001", "slide_002", "slide_003")

    first = store.read("slide_001")
    assert first.tile_features.shape == (4, 8)
    assert first.coords is not None
    assert first.coords.shape == (4, 2)
    assert first.label == 0
    assert first.metadata is not None
    assert first.metadata["source"] == "generic_feature_file"

    third = store.read("slide_003")
    assert third.tile_features.shape == (5, 8)
    assert third.coords is not None
    assert third.coords.shape == (5, 2)
    assert third.label == 2


def test_import_generic_feature_store_cli_supports_explicit_keys(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    torch.save(
        {
            "not_this": torch.randn(2, 2),
            "my_features": torch.randn(4, 8),
        },
        tmp_path / "slide_001.pt",
    )
    np.savez(
        tmp_path / "slide_001_coords.npz",
        my_coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
    )

    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "slide_001.pt",
                "coords_path": "slide_001_coords.npz",
                "label": "1",
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/import_generic_feature_store.py",
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_path),
            "--feature-dim",
            "8",
            "--feature-key",
            "my_features",
            "--coords-key",
            "my_coords",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    bag = H5WSIFeatureStore(output_path).read("slide_001")
    assert bag.tile_features.shape == (4, 8)
    assert bag.coords is not None
    assert bag.coords.shape == (4, 2)
    assert bag.label == 1


def test_import_generic_feature_store_cli_rejects_feature_dim_mismatch(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    torch.save(torch.randn(4, 7), tmp_path / "slide_001.pt")
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "slide_001.pt",
                "coords_path": "",
                "label": "",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_generic_feature_store.py",
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


def test_import_generic_feature_store_cli_rejects_coords_length_mismatch(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    torch.save(torch.randn(4, 8), tmp_path / "slide_001.pt")
    torch.save(torch.tensor([[0, 0], [1, 0], [0, 1]]), tmp_path / "slide_001_coords.pt")

    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "slide_001.pt",
                "coords_path": "slide_001_coords.pt",
                "label": "",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_generic_feature_store.py",
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
    assert "coordinates" in result.stderr


def test_import_generic_feature_store_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    torch.save(torch.randn(4, 8), tmp_path / "slide_001.pt")
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "slide_001.pt",
                "coords_path": "",
                "label": "",
            }
        ],
    )
    output_path.write_text("already here")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_generic_feature_store.py",
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


def test_import_generic_feature_store_cli_output_passes_validator(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.h5"

    torch.save(torch.randn(4, 8), tmp_path / "slide_001.pt")
    torch.save(torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]), tmp_path / "slide_001_coords.pt")

    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "features_path": "slide_001.pt",
                "coords_path": "slide_001_coords.pt",
                "label": "0",
            }
        ],
    )

    import_result = subprocess.run(
        [
            sys.executable,
            "scripts/import_generic_feature_store.py",
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
