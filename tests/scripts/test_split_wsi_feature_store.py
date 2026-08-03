import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag


def _make_bag(slide_id: str, *, label: int | None) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(4, 8),
        coords=torch.zeros(4, 2, dtype=torch.long),
        label=label,
        attention=None,
        metadata={"source": "unit_test"},
    )


def _write_store(path: Path, *, n_slides: int = 10, with_labels: bool = True) -> None:
    store = H5WSIFeatureStore(path)
    for index in range(n_slides):
        store.write(
            _make_bag(
                f"slide_{index:03d}",
                label=index % 2 if with_labels else None,
            )
        )


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_split_wsi_feature_store_cli_writes_train_val_test_files(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    output_dir = tmp_path / "splits"

    _write_store(store_path, n_slides=10, with_labels=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/split_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
            "--output-dir",
            str(output_dir),
            "--train-ratio",
            "0.6",
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.2",
            "--seed",
            "123",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)

    train_ids = _read_ids(output_dir / "train.txt")
    val_ids = _read_ids(output_dir / "val.txt")
    test_ids = _read_ids(output_dir / "test.txt")

    assert len(train_ids) == 6
    assert len(val_ids) == 2
    assert len(test_ids) == 2
    assert len(set(train_ids + val_ids + test_ids)) == 10

    assert summary["n_slides"] == 10
    assert summary["splits"]["train"]["n_slides"] == 6
    assert (output_dir / "split_summary.json").exists()
    assert json.loads((output_dir / "split_summary.json").read_text()) == summary


def test_split_wsi_feature_store_cli_is_reproducible_for_same_seed(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    output_a = tmp_path / "splits_a"
    output_b = tmp_path / "splits_b"

    _write_store(store_path, n_slides=12, with_labels=True)

    base_command = [
        sys.executable,
        "scripts/split_wsi_feature_store.py",
        "--feature-store",
        str(store_path),
        "--train-ratio",
        "0.5",
        "--val-ratio",
        "0.25",
        "--test-ratio",
        "0.25",
        "--seed",
        "999",
    ]

    subprocess.run(
        [*base_command, "--output-dir", str(output_a)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*base_command, "--output-dir", str(output_b)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_a / "train.txt").read_text() == (output_b / "train.txt").read_text()
    assert (output_a / "val.txt").read_text() == (output_b / "val.txt").read_text()
    assert (output_a / "test.txt").read_text() == (output_b / "test.txt").read_text()


def test_split_wsi_feature_store_cli_supports_label_stratification(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    output_dir = tmp_path / "splits"

    _write_store(store_path, n_slides=12, with_labels=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/split_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
            "--output-dir",
            str(output_dir),
            "--train-ratio",
            "0.5",
            "--val-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
            "--seed",
            "123",
            "--stratify-label",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_dir / "split_summary.json").read_text())

    assert summary["stratify_label"] is True
    assert summary["splits"]["train"]["label_distribution"] == {"0": 3, "1": 3}
    assert summary["splits"]["val"]["label_distribution"] == {"0": 2, "1": 2}
    assert summary["splits"]["test"]["label_distribution"] == {"0": 1, "1": 1}


def test_split_wsi_feature_store_cli_rejects_stratification_without_labels(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    output_dir = tmp_path / "splits"

    _write_store(store_path, n_slides=6, with_labels=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/split_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
            "--output-dir",
            str(output_dir),
            "--stratify-label",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "has no label" in result.stderr


def test_split_wsi_feature_store_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    output_dir = tmp_path / "splits"

    _write_store(store_path, n_slides=6, with_labels=True)
    output_dir.mkdir()
    (output_dir / "train.txt").write_text("already here")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/split_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "split output files already exist" in result.stderr


def test_split_wsi_feature_store_cli_overwrites_when_requested(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    output_dir = tmp_path / "splits"

    _write_store(store_path, n_slides=6, with_labels=True)
    output_dir.mkdir()
    (output_dir / "train.txt").write_text("already here")

    subprocess.run(
        [
            sys.executable,
            "scripts/split_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "already here" not in (output_dir / "train.txt").read_text()


def test_split_wsi_feature_store_cli_rejects_bad_ratios(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    output_dir = tmp_path / "splits"

    _write_store(store_path, n_slides=6, with_labels=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/split_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
            "--output-dir",
            str(output_dir),
            "--train-ratio",
            "0.6",
            "--val-ratio",
            "0.3",
            "--test-ratio",
            "0.3",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must sum to 1.0" in result.stderr
