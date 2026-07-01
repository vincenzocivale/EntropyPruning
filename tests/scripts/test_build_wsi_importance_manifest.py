import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _run(args: list[str], *, repo_root: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/build_wsi_importance_manifest.py", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def _make_targets_and_coords(tmp_path: Path) -> tuple[Path, Path]:
    targets_dir = tmp_path / "targets"
    coords_dir = tmp_path / "coords"
    targets_dir.mkdir()
    coords_dir.mkdir()

    h5py = pytest.importorskip("h5py")
    with h5py.File(targets_dir / "slide_001.h5", "w") as handle:
        handle.create_dataset("attention", data=np.array([0.2, 0.8], dtype=np.float32))
    with h5py.File(coords_dir / "slide_001.h5", "w") as handle:
        handle.create_dataset("coords", data=np.array([[0, 0], [0, 1]], dtype=np.int64))

    np.save(targets_dir / "slide_002.npy", np.array([0.5, 0.5], dtype=np.float32))
    np.save(coords_dir / "slide_002.npy", np.array([[0, 0], [1, 0]], dtype=np.int64))

    return targets_dir, coords_dir


def test_build_wsi_importance_manifest_writes_expected_rows(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets_dir, coords_dir = _make_targets_and_coords(tmp_path)
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("slide_id,label\nslide_001,1\nslide_002,0\n")
    output_manifest = tmp_path / "manifest.csv"

    _run(
        [
            "--targets-dir",
            str(targets_dir),
            "--coords-dir",
            str(coords_dir),
            "--labels-csv",
            str(labels_csv),
            "--output-manifest",
            str(output_manifest),
            "--target-glob",
            "*",
            "--target-source",
            "gigapath_wsi_fm",
            "--require-coords",
            "--require-labels",
        ],
        repo_root=repo_root,
    )

    with output_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["slide_id"] for row in rows} == {"slide_001", "slide_002"}
    for row in rows:
        assert row["target_source"] == "gigapath_wsi_fm"
        assert row["target_type"] == "tile_importance"
        assert row["coords_path"]
        assert row["label"]


def test_build_wsi_importance_manifest_glob_filters_by_suffix(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets_dir, coords_dir = _make_targets_and_coords(tmp_path)
    output_manifest = tmp_path / "manifest_h5_only.csv"

    _run(
        [
            "--targets-dir",
            str(targets_dir),
            "--coords-dir",
            str(coords_dir),
            "--output-manifest",
            str(output_manifest),
            "--target-glob",
            "*.h5",
        ],
        repo_root=repo_root,
    )

    with output_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["slide_id"] for row in rows] == ["slide_001"]


def test_build_wsi_importance_manifest_require_coords_fails_on_missing(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets_dir = tmp_path / "targets"
    targets_dir.mkdir()
    np.save(targets_dir / "slide_no_coords.npy", np.array([0.5, 0.5], dtype=np.float32))
    coords_dir = tmp_path / "coords"
    coords_dir.mkdir()
    output_manifest = tmp_path / "manifest.csv"

    result = _run(
        [
            "--targets-dir",
            str(targets_dir),
            "--coords-dir",
            str(coords_dir),
            "--output-manifest",
            str(output_manifest),
            "--target-glob",
            "*.npy",
            "--require-coords",
        ],
        repo_root=repo_root,
        check=False,
    )

    assert result.returncode != 0
    assert "missing coords" in result.stderr


def test_build_wsi_importance_manifest_rejects_existing_output_without_overwrite(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets_dir, coords_dir = _make_targets_and_coords(tmp_path)
    output_manifest = tmp_path / "manifest.csv"
    output_manifest.write_text("existing content")

    result = _run(
        [
            "--targets-dir",
            str(targets_dir),
            "--coords-dir",
            str(coords_dir),
            "--output-manifest",
            str(output_manifest),
            "--target-glob",
            "*",
        ],
        repo_root=repo_root,
        check=False,
    )

    assert result.returncode != 0
    assert "already exists" in result.stderr
