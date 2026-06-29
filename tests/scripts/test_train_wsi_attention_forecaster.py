import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag
from src.models.wsi import load_wsi_tile_attention_forecaster_checkpoint


def _make_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int,
    generator: torch.Generator,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim, generator=generator)
    direction = torch.linspace(-1.0, 1.0, feature_dim)
    attention = torch.softmax(tile_features @ direction, dim=0)

    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=1,
        attention=attention,
        metadata={"source": "synthetic"},
    )


def test_train_wsi_attention_forecaster_cli_smoke(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    feature_store_path = tmp_path / "features.h5"
    output_dir = tmp_path / "out"

    generator = torch.Generator().manual_seed(123)
    store = H5WSIFeatureStore(feature_store_path)
    for index in range(6):
        store.write(
            _make_bag(
                slide_id=f"slide_{index:03d}",
                n_tiles=4 + (index % 3),
                feature_dim=8,
                generator=generator,
            )
        )

    command = [
        sys.executable,
        "scripts/train_wsi_attention_forecaster.py",
        "--feature-store",
        str(feature_store_path),
        "--output-dir",
        str(output_dir),
        "--feature-dim",
        "8",
        "--hidden-dim",
        "16",
        "--n-heads",
        "4",
        "--n-layers",
        "1",
        "--dropout",
        "0.0",
        "--epochs",
        "2",
        "--batch-size",
        "2",
        "--lr",
        "1e-3",
        "--weight-decay",
        "0.0",
        "--top-k",
        "2",
        "--seed",
        "0",
        "--device",
        "cpu",
    ]

    result = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"event": "done"' in result.stdout

    checkpoint_path = output_dir / "best_wsi_tile_attention_forecaster.pt"
    summary_path = output_dir / "training_summary.json"

    assert checkpoint_path.exists()
    assert summary_path.exists()

    checkpoint = load_wsi_tile_attention_forecaster_checkpoint(checkpoint_path)
    assert checkpoint.config.feature_dim == 8
    assert checkpoint.config.hidden_dim == 16
    assert checkpoint.epoch in {1, 2}
    assert "val_loss" in checkpoint.metrics

    summary = json.loads(summary_path.read_text())
    assert summary["best_epoch"] in {1, 2}
    assert len(summary["history"]) == 2
    assert len(summary["split"]["train_slide_ids"]) == 5
    assert len(summary["split"]["val_slide_ids"]) == 1


def test_train_wsi_attention_forecaster_cli_supports_split_dir(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    feature_store_path = tmp_path / "features_split_dir.h5"
    output_dir = tmp_path / "out_split_dir"
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    generator = torch.Generator().manual_seed(321)
    store = H5WSIFeatureStore(feature_store_path)
    for index in range(6):
        store.write(
            _make_bag(
                slide_id=f"split_slide_{index:03d}",
                n_tiles=4 + (index % 2),
                feature_dim=8,
                generator=generator,
            )
        )

    (split_dir / "train.txt").write_text(
        "split_slide_000\nsplit_slide_001\nsplit_slide_002\nsplit_slide_003\n"
    )
    (split_dir / "val.txt").write_text("split_slide_004\nsplit_slide_005\n")
    (split_dir / "test.txt").write_text("")

    subprocess.run(
        [
            sys.executable,
            "scripts/train_wsi_attention_forecaster.py",
            "--feature-store",
            str(feature_store_path),
            "--output-dir",
            str(output_dir),
            "--feature-dim",
            "8",
            "--hidden-dim",
            "16",
            "--n-heads",
            "4",
            "--n-layers",
            "1",
            "--dropout",
            "0.0",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--top-k",
            "2",
            "--device",
            "cpu",
            "--split-dir",
            str(split_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["split"]["train_slide_ids"] == [
        "split_slide_000",
        "split_slide_001",
        "split_slide_002",
        "split_slide_003",
    ]
    assert summary["split"]["val_slide_ids"] == [
        "split_slide_004",
        "split_slide_005",
    ]


def test_train_wsi_attention_forecaster_cli_rejects_split_dir_with_explicit_split_files(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    feature_store_path = tmp_path / "features_conflict.h5"
    output_dir = tmp_path / "out_conflict"
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    train_ids_path = tmp_path / "train.txt"
    val_ids_path = tmp_path / "val.txt"

    generator = torch.Generator().manual_seed(321)
    store = H5WSIFeatureStore(feature_store_path)
    for index in range(4):
        store.write(
            _make_bag(
                slide_id=f"conflict_slide_{index:03d}",
                n_tiles=4,
                feature_dim=8,
                generator=generator,
            )
        )

    (split_dir / "train.txt").write_text("conflict_slide_000\nconflict_slide_001\n")
    (split_dir / "val.txt").write_text("conflict_slide_002\n")
    train_ids_path.write_text("conflict_slide_000\nconflict_slide_001\n")
    val_ids_path.write_text("conflict_slide_002\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_wsi_attention_forecaster.py",
            "--feature-store",
            str(feature_store_path),
            "--output-dir",
            str(output_dir),
            "--feature-dim",
            "8",
            "--hidden-dim",
            "16",
            "--n-heads",
            "4",
            "--n-layers",
            "1",
            "--epochs",
            "1",
            "--top-k",
            "2",
            "--device",
            "cpu",
            "--split-dir",
            str(split_dir),
            "--train-slide-ids-file",
            str(train_ids_path),
            "--val-slide-ids-file",
            str(val_ids_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_train_wsi_attention_forecaster_cli_supports_explicit_split_files(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    feature_store_path = tmp_path / "features.h5"
    output_dir = tmp_path / "out"
    train_ids_path = tmp_path / "train.txt"
    val_ids_path = tmp_path / "val.txt"

    generator = torch.Generator().manual_seed(123)
    store = H5WSIFeatureStore(feature_store_path)
    for index in range(4):
        store.write(
            _make_bag(
                slide_id=f"slide_{index:03d}",
                n_tiles=4 + (index % 2),
                feature_dim=8,
                generator=generator,
            )
        )

    train_ids_path.write_text("slide_000\nslide_001\nslide_002\n")
    val_ids_path.write_text("slide_003\n")

    command = [
        sys.executable,
        "scripts/train_wsi_attention_forecaster.py",
        "--feature-store",
        str(feature_store_path),
        "--output-dir",
        str(output_dir),
        "--feature-dim",
        "8",
        "--hidden-dim",
        "16",
        "--n-heads",
        "4",
        "--n-layers",
        "1",
        "--dropout",
        "0.0",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--top-k",
        "2",
        "--device",
        "cpu",
        "--train-slide-ids-file",
        str(train_ids_path),
        "--val-slide-ids-file",
        str(val_ids_path),
    ]

    subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["split"]["train_slide_ids"] == ["slide_000", "slide_001", "slide_002"]
    assert summary["split"]["val_slide_ids"] == ["slide_003"]
