import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag
from src.models.wsi import (
    WSI_TILE_IMPORTANCE_LOSS_TYPES,
    load_wsi_tile_importance_forecaster_checkpoint,
)


def _make_input_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int,
    generator: torch.Generator,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim, generator=generator)
    coords = torch.stack(
        [torch.arange(n_tiles), torch.arange(n_tiles) * 2], dim=1
    ).to(torch.long)
    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=coords,
        label=1,
        metadata={"source": "synthetic_input"},
    )


def _make_target_bag(
    input_bag: WSIBag,
    *,
    permute: bool = False,
) -> WSIBag:
    direction = torch.linspace(-1.0, 1.0, input_bag.feature_dim)
    target = torch.softmax(input_bag.tile_features @ direction, dim=0)
    coords = input_bag.coords

    if permute:
        order = torch.randperm(input_bag.n_tiles)
        target = target[order]
        coords = coords[order]

    return WSIBag(
        slide_id=input_bag.slide_id,
        tile_features=target.unsqueeze(1),
        coords=coords,
        attention=target,
        metadata={"target_source": "synthetic_abmil"},
    )


def _build_paired_stores(
    tmp_path: Path,
    *,
    n_slides: int = 6,
    permute_target: bool = False,
    seed: int = 123,
) -> tuple[Path, Path]:
    input_path = tmp_path / "input.h5"
    target_path = tmp_path / "target.h5"

    generator = torch.Generator().manual_seed(seed)
    input_store = H5WSIFeatureStore(input_path)
    target_store = H5WSIFeatureStore(target_path)

    for index in range(n_slides):
        input_bag = _make_input_bag(
            slide_id=f"slide_{index:03d}",
            n_tiles=4 + (index % 3),
            feature_dim=8,
            generator=generator,
        )
        input_store.write(input_bag)
        target_store.write(_make_target_bag(input_bag, permute=permute_target))

    return input_path, target_path


def _run(command: list[str], *, repo_root: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def _base_command(*, input_path: Path, target_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/train_wsi_importance_forecaster.py",
        "--input-feature-store",
        str(input_path),
        "--target-feature-store",
        str(target_path),
        "--output-dir",
        str(output_dir),
        "--input-feature-dim",
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
        "--top-k",
        "2",
        "--seed",
        "0",
        "--device",
        "cpu",
    ]


def test_train_wsi_importance_forecaster_cli_smoke_with_paired_stores(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path)
    output_dir = tmp_path / "out"

    result = _run(
        _base_command(input_path=input_path, target_path=target_path, output_dir=output_dir),
        repo_root=repo_root,
    )

    assert '"event": "done"' in result.stdout

    checkpoint_path = output_dir / "best_wsi_tile_importance_forecaster.pt"
    summary_path = output_dir / "training_summary.json"
    assert checkpoint_path.exists()
    assert summary_path.exists()

    checkpoint = load_wsi_tile_importance_forecaster_checkpoint(checkpoint_path)
    assert checkpoint.config.feature_dim == 8
    assert checkpoint.metadata["model_type"] == "WSITileImportanceForecaster"
    assert checkpoint.metadata["loss"] == "kl"
    assert checkpoint.metadata["target_source"] == "synthetic_abmil"
    assert checkpoint.metadata["input_feature_store"] == str(input_path)
    assert checkpoint.metadata["target_feature_store"] == str(target_path)
    assert checkpoint.metadata["alignment_mode"] == "index"

    summary = json.loads(summary_path.read_text())
    assert summary["best_epoch"] in {1, 2}
    assert summary["target_source"] == "synthetic_abmil"
    assert summary["seed"] == 0
    assert "target_entropy_mean" in summary
    assert summary["target_entropy_mean"] is not None
    assert len(summary["split"]["train_slide_ids"]) == 5
    assert len(summary["split"]["val_slide_ids"]) == 1


def test_train_wsi_importance_forecaster_cli_supports_coords_alignment(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path, permute_target=True)
    output_dir = tmp_path / "out_coords"

    command = _base_command(
        input_path=input_path, target_path=target_path, output_dir=output_dir
    ) + ["--alignment-mode", "coords", "--require-coords"]

    result = _run(command, repo_root=repo_root)

    assert '"event": "done"' in result.stdout
    checkpoint = load_wsi_tile_importance_forecaster_checkpoint(
        output_dir / "best_wsi_tile_importance_forecaster.pt"
    )
    assert checkpoint.metadata["alignment_mode"] == "coords"


@pytest.mark.parametrize("loss", WSI_TILE_IMPORTANCE_LOSS_TYPES)
def test_train_wsi_importance_forecaster_cli_supports_all_loss_types(tmp_path, loss) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path, seed=7)
    output_dir = tmp_path / f"out_{loss.replace('+', '_')}"

    command = _base_command(
        input_path=input_path, target_path=target_path, output_dir=output_dir
    ) + ["--epochs", "1", "--loss", loss]

    result = _run(command, repo_root=repo_root)

    assert '"event": "done"' in result.stdout
    checkpoint = load_wsi_tile_importance_forecaster_checkpoint(
        output_dir / "best_wsi_tile_importance_forecaster.pt"
    )
    assert checkpoint.metadata["loss"] == loss


def test_train_wsi_importance_forecaster_cli_supports_legacy_feature_store_alias(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fused_path = tmp_path / "fused.h5"
    output_dir = tmp_path / "out_fused"

    generator = torch.Generator().manual_seed(11)
    store = H5WSIFeatureStore(fused_path)
    for index in range(4):
        n_tiles = 4 + index
        tile_features = torch.randn(n_tiles, 8, generator=generator)
        direction = torch.linspace(-1.0, 1.0, 8)
        attention = torch.softmax(tile_features @ direction, dim=0)
        store.write(
            WSIBag(
                slide_id=f"fused_{index:03d}",
                tile_features=tile_features,
                attention=attention,
                label=1,
            )
        )

    command = [
        sys.executable,
        "scripts/train_wsi_importance_forecaster.py",
        "--feature-store",
        str(fused_path),
        "--output-dir",
        str(output_dir),
        "--input-feature-dim",
        "8",
        "--hidden-dim",
        "16",
        "--n-heads",
        "4",
        "--n-layers",
        "1",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--top-k",
        "2",
        "--device",
        "cpu",
    ]

    result = _run(command, repo_root=repo_root)

    assert '"event": "done"' in result.stdout
    checkpoint_path = output_dir / "best_wsi_tile_importance_forecaster.pt"
    assert checkpoint_path.exists()

    checkpoint = load_wsi_tile_importance_forecaster_checkpoint(checkpoint_path)
    assert checkpoint.metadata["input_feature_store"] == str(fused_path)
    assert checkpoint.metadata["target_feature_store"] == str(fused_path)


def test_train_wsi_importance_forecaster_cli_rejects_conflicting_store_args(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path, n_slides=2)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_wsi_importance_forecaster.py",
            "--feature-store",
            str(input_path),
            "--input-feature-store",
            str(input_path),
            "--target-feature-store",
            str(target_path),
            "--output-dir",
            str(tmp_path / "out_bad"),
            "--input-feature-dim",
            "8",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_train_wsi_importance_forecaster_cli_rejects_missing_store_args(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_wsi_importance_forecaster.py",
            "--output-dir",
            str(tmp_path / "out_bad"),
            "--input-feature-dim",
            "8",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must be provided" in result.stderr


def test_train_wsi_importance_forecaster_cli_supports_split_dir(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path, n_slides=6, seed=321)
    output_dir = tmp_path / "out_split_dir"
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    (split_dir / "train.txt").write_text(
        "slide_000\nslide_001\nslide_002\nslide_003\n"
    )
    (split_dir / "val.txt").write_text("slide_004\nslide_005\n")

    command = _base_command(
        input_path=input_path, target_path=target_path, output_dir=output_dir
    ) + ["--epochs", "1", "--split-dir", str(split_dir)]

    _run(command, repo_root=repo_root)

    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["split"]["train_slide_ids"] == [
        "slide_000",
        "slide_001",
        "slide_002",
        "slide_003",
    ]
    assert summary["split"]["val_slide_ids"] == ["slide_004", "slide_005"]
    assert summary["split"]["split_files"]["train"] == str(split_dir / "train.txt")
    assert summary["split"]["split_files"]["val"] == str(split_dir / "val.txt")


def test_train_wsi_importance_forecaster_cli_target_smoothing_rescues_all_zero_bag(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = tmp_path / "input_zero.h5"
    target_path = tmp_path / "target_zero.h5"
    output_dir = tmp_path / "out_zero"

    generator = torch.Generator().manual_seed(5)
    input_store = H5WSIFeatureStore(input_path)
    target_store = H5WSIFeatureStore(target_path)
    for index in range(4):
        n_tiles = 4
        tile_features = torch.randn(n_tiles, 8, generator=generator)
        input_store.write(WSIBag(slide_id=f"zero_{index:03d}", tile_features=tile_features))
        target_store.write(
            WSIBag(
                slide_id=f"zero_{index:03d}",
                tile_features=torch.zeros(n_tiles, 1),
                attention=torch.zeros(n_tiles),
            )
        )

    command = [
        sys.executable,
        "scripts/train_wsi_importance_forecaster.py",
        "--input-feature-store",
        str(input_path),
        "--target-feature-store",
        str(target_path),
        "--output-dir",
        str(output_dir),
        "--input-feature-dim",
        "8",
        "--hidden-dim",
        "16",
        "--n-heads",
        "4",
        "--n-layers",
        "1",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--top-k",
        "2",
        "--device",
        "cpu",
    ]

    failing = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    assert failing.returncode != 0

    smoothed = subprocess.run(
        command + ["--target-smoothing", "1.0"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert smoothed.returncode == 0
    assert '"event": "done"' in smoothed.stdout
