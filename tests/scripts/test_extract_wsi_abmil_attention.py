import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag
from src.models.wsi import ABMILClassifierConfig, save_abmil_classifier_checkpoint


def _make_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int,
    label: int,
    generator: torch.Generator,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim, generator=generator)

    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=label,
        attention=None,
        metadata={"source": "synthetic"},
    )


def _make_input_store(path: Path, *, n_slides: int = 4, feature_dim: int = 8) -> None:
    generator = torch.Generator().manual_seed(123)
    store = H5WSIFeatureStore(path)

    for index in range(n_slides):
        store.write(
            _make_bag(
                slide_id=f"slide_{index:03d}",
                n_tiles=4 + (index % 3),
                feature_dim=feature_dim,
                label=index % 2,
                generator=generator,
            )
        )


def _make_abmil_checkpoint(path: Path, *, feature_dim: int = 8) -> None:
    config = ABMILClassifierConfig(
        feature_dim=feature_dim,
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


def test_extract_wsi_abmil_attention_cli_writes_attention_targets(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "output.h5"
    checkpoint_path = tmp_path / "abmil.pt"

    _make_input_store(input_store_path, n_slides=4, feature_dim=8)
    _make_abmil_checkpoint(checkpoint_path, feature_dim=8)

    command = [
        sys.executable,
        "scripts/extract_wsi_abmil_attention.py",
        "--input-feature-store",
        str(input_store_path),
        "--output-feature-store",
        str(output_store_path),
        "--abmil-checkpoint",
        str(checkpoint_path),
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

    input_store = H5WSIFeatureStore(input_store_path)
    output_store = H5WSIFeatureStore(output_store_path)

    assert output_store.slide_ids() == input_store.slide_ids()

    for slide_id in output_store.slide_ids():
        original = input_store.read(slide_id)
        extracted = output_store.read(slide_id)

        assert extracted.slide_id == original.slide_id
        assert torch.equal(extracted.tile_features, original.tile_features)
        assert extracted.coords is not None
        assert original.coords is not None
        assert torch.equal(extracted.coords, original.coords)
        assert extracted.label == original.label
        assert extracted.attention is not None
        assert extracted.attention.shape == (original.n_tiles,)
        assert torch.isclose(extracted.attention.sum(), torch.tensor(1.0), atol=1e-6)
        assert (extracted.attention >= 0).all()
        assert extracted.metadata is not None
        assert extracted.metadata["source"] == "synthetic"
        assert extracted.metadata["attention_source"] == "ABMILClassifier"


def test_extract_wsi_abmil_attention_cli_supports_slide_subset(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "output.h5"
    checkpoint_path = tmp_path / "abmil.pt"
    slide_ids_path = tmp_path / "slide_ids.txt"

    _make_input_store(input_store_path, n_slides=4, feature_dim=8)
    _make_abmil_checkpoint(checkpoint_path, feature_dim=8)
    slide_ids_path.write_text("slide_001\nslide_003\n")

    command = [
        sys.executable,
        "scripts/extract_wsi_abmil_attention.py",
        "--input-feature-store",
        str(input_store_path),
        "--output-feature-store",
        str(output_store_path),
        "--abmil-checkpoint",
        str(checkpoint_path),
        "--slide-ids-file",
        str(slide_ids_path),
        "--batch-size",
        "2",
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

    output_store = H5WSIFeatureStore(output_store_path)
    assert output_store.slide_ids() == ("slide_001", "slide_003")


def test_extract_wsi_abmil_attention_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "output.h5"
    checkpoint_path = tmp_path / "abmil.pt"

    _make_input_store(input_store_path, n_slides=2, feature_dim=8)
    _make_abmil_checkpoint(checkpoint_path, feature_dim=8)
    output_store_path.write_text("already here")

    command = [
        sys.executable,
        "scripts/extract_wsi_abmil_attention.py",
        "--input-feature-store",
        str(input_store_path),
        "--output-feature-store",
        str(output_store_path),
        "--abmil-checkpoint",
        str(checkpoint_path),
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
    assert "output feature store already exists" in result.stderr


def test_extract_wsi_abmil_attention_cli_output_passes_validator(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "output.h5"
    checkpoint_path = tmp_path / "abmil.pt"

    _make_input_store(input_store_path, n_slides=4, feature_dim=8)
    _make_abmil_checkpoint(checkpoint_path, feature_dim=8)

    extract_result = subprocess.run(
        [
            sys.executable,
            "scripts/extract_wsi_abmil_attention.py",
            "--input-feature-store",
            str(input_store_path),
            "--output-feature-store",
            str(output_store_path),
            "--abmil-checkpoint",
            str(checkpoint_path),
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert extract_result.returncode == 0, extract_result.stderr

    validate_result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_wsi_feature_store.py",
            "--feature-store",
            str(output_store_path),
            "--feature-dim",
            "8",
            "--require-attention",
            "--require-coords",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert validate_result.returncode == 0, validate_result.stderr
    summary = json.loads(validate_result.stdout)
    assert summary["valid"] is True
    assert summary["n_slides"] == 4
    assert summary["n_with_attention"] == 4
