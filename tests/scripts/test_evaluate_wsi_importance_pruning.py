import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag
from src.models.wsi import (
    ABMILClassifierConfig,
    WSITileImportanceForecasterConfig,
    save_abmil_classifier_checkpoint,
    save_wsi_tile_importance_forecaster_checkpoint,
)


def _is_probability_like(value: float, eps: float = 1e-6) -> bool:
    return -eps <= value <= 1.0 + eps


def _make_input_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int,
    label: int,
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
        label=label,
        metadata={"source": "synthetic_input"},
    )


def _make_target_bag(input_bag: WSIBag, *, permute: bool = False) -> WSIBag:
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
            n_tiles=6 + (index % 3),
            feature_dim=8,
            label=index % 2,
            generator=generator,
        )
        input_store.write(input_bag)
        target_store.write(_make_target_bag(input_bag, permute=permute_target))

    return input_path, target_path


def _make_forecaster_checkpoint(path: Path) -> None:
    config = WSITileImportanceForecasterConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    model = config.build()

    save_wsi_tile_importance_forecaster_checkpoint(
        path,
        model=model,
        config=config,
        loss="kl",
        target_source="synthetic_abmil",
        epoch=1,
        metrics={"val_loss": 1.0},
    )


def _make_abmil_checkpoint(path: Path) -> None:
    config = ABMILClassifierConfig(
        feature_dim=8,
        hidden_dim=16,
        n_classes=2,
        dropout=0.0,
        gated=True,
    )
    model = config.build()

    save_abmil_classifier_checkpoint(
        path,
        model=model,
        config=config,
        epoch=1,
        metrics={"val_loss": 1.0},
        metadata={"source": "unit_test"},
    )


def _run(command: list[str], *, repo_root: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=repo_root, check=check, capture_output=True, text=True)


def _base_command(*, input_path: Path, target_path: Path, checkpoint_path: Path, output_csv: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_wsi_importance_pruning.py",
        "--input-feature-store",
        str(input_path),
        "--target-feature-store",
        str(target_path),
        "--forecaster-checkpoint",
        str(checkpoint_path),
        "--keep-ratios",
        "0.25",
        "0.5",
        "1.0",
        "--output-csv",
        str(output_csv),
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ]


def test_evaluate_wsi_importance_pruning_cli_writes_csv_with_paired_stores(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path)
    checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "pruning.csv"

    _make_forecaster_checkpoint(checkpoint_path)

    result = _run(
        _base_command(
            input_path=input_path,
            target_path=target_path,
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
        ),
        repo_root=repo_root,
    )

    assert '"event": "done"' in result.stdout
    assert output_csv.exists()

    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 3
    assert [float(row["keep_ratio"]) for row in rows] == [0.25, 0.5, 1.0]
    assert all(int(row["n_slides"]) == 6 for row in rows)

    for row in rows:
        assert _is_probability_like(float(row["mean_topk_overlap"]))
        assert _is_probability_like(float(row["mean_ndcg_at_k"]))
        assert _is_probability_like(float(row["mean_target_importance_mass_retained"]))
        assert _is_probability_like(float(row["mean_oracle_importance_mass_retained"]))
        assert _is_probability_like(float(row["mean_relative_importance_mass_retained"]))
        assert "full_accuracy" not in row


def test_evaluate_wsi_importance_pruning_cli_supports_coords_alignment(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path, permute_target=True)
    checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "pruning.csv"

    _make_forecaster_checkpoint(checkpoint_path)

    command = _base_command(
        input_path=input_path,
        target_path=target_path,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
    ) + ["--alignment-mode", "coords", "--require-coords"]

    result = _run(command, repo_root=repo_root)

    assert '"event": "done"' in result.stdout
    assert output_csv.exists()


def test_evaluate_wsi_importance_pruning_cli_supports_legacy_feature_store_alias(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fused_path = tmp_path / "fused.h5"
    checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "pruning.csv"

    generator = torch.Generator().manual_seed(11)
    store = H5WSIFeatureStore(fused_path)
    for index in range(4):
        n_tiles = 6 + index
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

    _make_forecaster_checkpoint(checkpoint_path)

    command = [
        sys.executable,
        "scripts/evaluate_wsi_importance_pruning.py",
        "--feature-store",
        str(fused_path),
        "--forecaster-checkpoint",
        str(checkpoint_path),
        "--keep-ratios",
        "0.5",
        "--output-csv",
        str(output_csv),
        "--device",
        "cpu",
    ]

    result = _run(command, repo_root=repo_root)

    assert '"event": "done"' in result.stdout
    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 1
    assert int(rows[0]["n_slides"]) == 4


def test_evaluate_wsi_importance_pruning_cli_with_abmil_checkpoint_adds_agreement_columns(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path)
    forecaster_checkpoint_path = tmp_path / "forecaster.pt"
    abmil_checkpoint_path = tmp_path / "abmil.pt"
    output_csv = tmp_path / "pruning.csv"

    _make_forecaster_checkpoint(forecaster_checkpoint_path)
    _make_abmil_checkpoint(abmil_checkpoint_path)

    command = _base_command(
        input_path=input_path,
        target_path=target_path,
        checkpoint_path=forecaster_checkpoint_path,
        output_csv=output_csv,
    ) + ["--abmil-checkpoint", str(abmil_checkpoint_path)]

    result = _run(command, repo_root=repo_root)

    assert '"event": "done"' in result.stdout
    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 3

    for row in rows:
        assert _is_probability_like(float(row["full_accuracy"]))
        assert _is_probability_like(float(row["pruned_accuracy"]))
        assert _is_probability_like(float(row["prediction_agreement"]))
        assert row["mean_logit_cosine_similarity"] != ""
        assert row["mean_prob_kl_full_to_pruned"] != ""


def test_evaluate_wsi_importance_pruning_cli_supports_slide_subset(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path)
    checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "pruning.csv"
    slide_ids_path = tmp_path / "slide_ids.txt"

    _make_forecaster_checkpoint(checkpoint_path)
    slide_ids_path.write_text("slide_001\nslide_003\n")

    command = _base_command(
        input_path=input_path,
        target_path=target_path,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
    ) + ["--slide-ids-file", str(slide_ids_path)]

    _run(command, repo_root=repo_root)

    rows = list(csv.DictReader(output_csv.open()))
    assert all(int(row["n_slides"]) == 2 for row in rows)


def test_evaluate_wsi_importance_pruning_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path, n_slides=2)
    checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "pruning.csv"

    _make_forecaster_checkpoint(checkpoint_path)
    output_csv.write_text("already here")

    result = subprocess.run(
        _base_command(
            input_path=input_path,
            target_path=target_path,
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output CSV already exists" in result.stderr


def test_evaluate_wsi_importance_pruning_cli_rejects_conflicting_store_args(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, target_path = _build_paired_stores(tmp_path, n_slides=2)
    checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "pruning.csv"

    _make_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_wsi_importance_pruning.py",
            "--feature-store",
            str(input_path),
            "--input-feature-store",
            str(input_path),
            "--target-feature-store",
            str(target_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratios",
            "0.5",
            "--output-csv",
            str(output_csv),
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_evaluate_wsi_importance_pruning_cli_rejects_missing_store_args(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "pruning.csv"

    _make_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_wsi_importance_pruning.py",
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratios",
            "0.5",
            "--output-csv",
            str(output_csv),
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must be provided" in result.stderr
