import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slide_id",
                "target_path",
                "coords_path",
                "label",
                "target_source",
                "target_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _run(args: list[str], *, repo_root: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/import_wsi_importance_targets.py", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def test_import_wsi_importance_targets_from_h5_and_npy(tmp_path) -> None:
    import h5py

    repo_root = Path(__file__).resolve().parents[2]

    with h5py.File(tmp_path / "slide_001_target.h5", "w") as handle:
        handle.create_dataset("attention", data=np.array([0.1, 0.6, 0.3], dtype=np.float32))
    with h5py.File(tmp_path / "slide_001_coords.h5", "w") as handle:
        handle.create_dataset(
            "coords", data=np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
        )
    np.save(tmp_path / "slide_002_target.npy", np.array([0.4, 0.6], dtype=np.float32))
    np.save(tmp_path / "slide_002_coords.npy", np.array([[0, 0], [1, 1]], dtype=np.int64))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "target_path": "slide_001_target.h5",
                "coords_path": "slide_001_coords.h5",
                "label": "1",
                "target_source": "gigapath_wsi_fm",
                "target_type": "tile_importance",
            },
            {
                "slide_id": "slide_002",
                "target_path": "slide_002_target.npy",
                "coords_path": "slide_002_coords.npy",
                "label": "0",
                "target_source": "gigapath_wsi_fm",
                "target_type": "tile_importance",
            },
        ],
    )

    output_store = tmp_path / "target_store.h5"

    result = _run(
        [
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_store),
            "--require-coords",
            "--created-by",
            "test_suite",
            "--patch-encoder",
            "uni_v1",
        ],
        repo_root=repo_root,
    )

    assert '"event": "done"' in result.stdout
    assert output_store.exists()

    store = H5WSIFeatureStore(output_store)
    assert set(store.slide_ids()) == {"slide_001", "slide_002"}

    bag = store.read("slide_001")
    assert bag.attention is not None
    assert bag.attention.shape == (3,)
    assert bag.coords is not None
    assert bag.metadata["target_source"] == "gigapath_wsi_fm"
    assert bag.metadata["target_type"] == "tile_importance"
    assert bag.metadata["created_by"] == "test_suite"
    assert bag.metadata["patch_encoder"] == "uni_v1"


def test_import_wsi_importance_targets_validates_with_require_attention(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    np.save(tmp_path / "slide_001_target.npy", np.array([0.5, 0.5], dtype=np.float32))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "target_path": "slide_001_target.npy",
                "coords_path": "",
                "label": "",
                "target_source": "",
                "target_type": "",
            }
        ],
    )

    output_store = tmp_path / "target_store.h5"
    _run(
        ["--manifest", str(manifest_path), "--output-feature-store", str(output_store)],
        repo_root=repo_root,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_wsi_feature_store.py",
            "--feature-store",
            str(output_store),
            "--require-attention",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '"valid": true' in result.stdout


def test_import_wsi_importance_targets_custom_target_key(tmp_path) -> None:
    import h5py

    repo_root = Path(__file__).resolve().parents[2]
    with h5py.File(tmp_path / "slide_001_target.h5", "w") as handle:
        handle.create_dataset(
            "my_custom_score", data=np.array([0.3, 0.7], dtype=np.float32)
        )

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "target_path": "slide_001_target.h5",
                "coords_path": "",
                "label": "",
                "target_source": "",
                "target_type": "",
            }
        ],
    )

    output_store = tmp_path / "target_store.h5"
    _run(
        [
            "--manifest",
            str(manifest_path),
            "--output-feature-store",
            str(output_store),
            "--target-key",
            "my_custom_score",
        ],
        repo_root=repo_root,
    )

    store = H5WSIFeatureStore(output_store)
    bag = store.read("slide_001")
    assert bag.attention.shape == (2,)


def test_import_wsi_importance_targets_coords_mismatch_fails(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    np.save(tmp_path / "slide_001_target.npy", np.array([0.2, 0.3, 0.5], dtype=np.float32))
    np.save(tmp_path / "slide_001_coords.npy", np.array([[0, 0], [1, 1]], dtype=np.int64))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "target_path": "slide_001_target.npy",
                "coords_path": "slide_001_coords.npy",
                "label": "",
                "target_source": "",
                "target_type": "",
            }
        ],
    )

    output_store = tmp_path / "target_store.h5"
    result = _run(
        ["--manifest", str(manifest_path), "--output-feature-store", str(output_store)],
        repo_root=repo_root,
        check=False,
    )

    assert result.returncode != 0
    assert "coordinates" in result.stderr


def test_import_wsi_importance_targets_rejects_existing_store_without_overwrite(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    np.save(tmp_path / "slide_001_target.npy", np.array([0.5, 0.5], dtype=np.float32))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "slide_id": "slide_001",
                "target_path": "slide_001_target.npy",
                "coords_path": "",
                "label": "",
                "target_source": "",
                "target_type": "",
            }
        ],
    )

    output_store = tmp_path / "target_store.h5"
    _run(
        ["--manifest", str(manifest_path), "--output-feature-store", str(output_store)],
        repo_root=repo_root,
    )

    result = _run(
        ["--manifest", str(manifest_path), "--output-feature-store", str(output_store)],
        repo_root=repo_root,
        check=False,
    )

    assert result.returncode != 0
    assert "already exists" in result.stderr
