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
    WSITileAttentionForecasterConfig,
    save_abmil_classifier_checkpoint,
    save_wsi_tile_attention_forecaster_checkpoint,
)


def _is_probability_like(value: float, eps: float = 1e-6) -> bool:
    return -eps <= value <= 1.0 + eps


def _make_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int,
    label: int,
    generator: torch.Generator,
    *,
    with_attention: bool = True,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim, generator=generator)
    direction = torch.linspace(-1.0, 1.0, feature_dim)
    attention = torch.softmax(tile_features @ direction, dim=0) if with_attention else None

    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=label,
        attention=attention,
        metadata={"source": "synthetic"},
    )


def _make_store(path: Path, *, with_attention: bool = True) -> None:
    generator = torch.Generator().manual_seed(123)
    store = H5WSIFeatureStore(path)

    for index in range(5):
        store.write(
            _make_bag(
                slide_id=f"slide_{index:03d}",
                n_tiles=4 + index,
                feature_dim=8,
                label=index % 2,
                generator=generator,
                with_attention=with_attention,
            )
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


def _make_forecaster_checkpoint(path: Path) -> None:
    config = WSITileAttentionForecasterConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    model = config.build()

    save_wsi_tile_attention_forecaster_checkpoint(
        path,
        model=model,
        config=config,
        epoch=1,
        metrics={"val_loss": 1.0},
        metadata={"source": "unit_test"},
    )


def test_evaluate_wsi_abmil_pruning_agreement_cli_writes_csv(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    abmil_checkpoint_path = tmp_path / "abmil.pt"
    forecaster_checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "agreement.csv"

    _make_store(store_path, with_attention=True)
    _make_abmil_checkpoint(abmil_checkpoint_path)
    _make_forecaster_checkpoint(forecaster_checkpoint_path)

    command = [
        sys.executable,
        "scripts/evaluate_wsi_abmil_pruning_agreement.py",
        "--feature-store",
        str(store_path),
        "--abmil-checkpoint",
        str(abmil_checkpoint_path),
        "--forecaster-checkpoint",
        str(forecaster_checkpoint_path),
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

    result = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"event": "done"' in result.stdout
    assert output_csv.exists()

    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 3
    assert [float(row["keep_ratio"]) for row in rows] == [0.25, 0.5, 1.0]

    for row in rows:
        assert int(row["n_slides"]) == 5
        assert _is_probability_like(float(row["full_accuracy"]))
        assert _is_probability_like(float(row["pruned_accuracy"]))
        assert _is_probability_like(float(row["prediction_agreement"]))
        assert -1.0 <= float(row["mean_logit_cosine_similarity"]) <= 1.0
        assert float(row["mean_prob_kl_full_to_pruned"]) >= -1e-6
        assert _is_probability_like(float(row["mean_attention_mass_retained"]))
        assert _is_probability_like(float(row["mean_oracle_attention_mass_at_k"]))
        assert _is_probability_like(float(row["mean_relative_attention_mass_retained"]))

    full_keep_row = rows[-1]
    assert float(full_keep_row["keep_ratio"]) == 1.0
    assert float(full_keep_row["prediction_agreement"]) == pytest.approx(1.0)
    assert float(full_keep_row["mean_attention_mass_retained"]) == pytest.approx(1.0)
    assert float(full_keep_row["mean_prob_kl_full_to_pruned"]) == pytest.approx(0.0, abs=1e-5)


def test_evaluate_wsi_abmil_pruning_agreement_cli_supports_slide_subset(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    abmil_checkpoint_path = tmp_path / "abmil.pt"
    forecaster_checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "agreement.csv"
    slide_ids_path = tmp_path / "slide_ids.txt"

    _make_store(store_path, with_attention=True)
    _make_abmil_checkpoint(abmil_checkpoint_path)
    _make_forecaster_checkpoint(forecaster_checkpoint_path)
    slide_ids_path.write_text("slide_001\nslide_003\n")

    command = [
        sys.executable,
        "scripts/evaluate_wsi_abmil_pruning_agreement.py",
        "--feature-store",
        str(store_path),
        "--abmil-checkpoint",
        str(abmil_checkpoint_path),
        "--forecaster-checkpoint",
        str(forecaster_checkpoint_path),
        "--keep-ratios",
        "0.5",
        "--output-csv",
        str(output_csv),
        "--slide-ids-file",
        str(slide_ids_path),
        "--device",
        "cpu",
    ]

    subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 1
    assert int(rows[0]["n_slides"]) == 2


def test_evaluate_wsi_abmil_pruning_agreement_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    abmil_checkpoint_path = tmp_path / "abmil.pt"
    forecaster_checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "agreement.csv"

    _make_store(store_path, with_attention=True)
    _make_abmil_checkpoint(abmil_checkpoint_path)
    _make_forecaster_checkpoint(forecaster_checkpoint_path)
    output_csv.write_text("already here")

    command = [
        sys.executable,
        "scripts/evaluate_wsi_abmil_pruning_agreement.py",
        "--feature-store",
        str(store_path),
        "--abmil-checkpoint",
        str(abmil_checkpoint_path),
        "--forecaster-checkpoint",
        str(forecaster_checkpoint_path),
        "--keep-ratios",
        "0.5",
        "--output-csv",
        str(output_csv),
        "--device",
        "cpu",
    ]

    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output CSV already exists" in result.stderr


def test_evaluate_wsi_abmil_pruning_agreement_cli_rejects_missing_attention(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    abmil_checkpoint_path = tmp_path / "abmil.pt"
    forecaster_checkpoint_path = tmp_path / "forecaster.pt"
    output_csv = tmp_path / "agreement.csv"

    _make_store(store_path, with_attention=False)
    _make_abmil_checkpoint(abmil_checkpoint_path)
    _make_forecaster_checkpoint(forecaster_checkpoint_path)

    command = [
        sys.executable,
        "scripts/evaluate_wsi_abmil_pruning_agreement.py",
        "--feature-store",
        str(store_path),
        "--abmil-checkpoint",
        str(abmil_checkpoint_path),
        "--forecaster-checkpoint",
        str(forecaster_checkpoint_path),
        "--keep-ratios",
        "0.5",
        "--output-csv",
        str(output_csv),
        "--device",
        "cpu",
    ]

    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must contain attention targets" in result.stderr
