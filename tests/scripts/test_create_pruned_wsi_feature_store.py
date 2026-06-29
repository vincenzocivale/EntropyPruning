import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag
from src.models.wsi import (
    WSITileAttentionForecasterConfig,
    load_wsi_tile_attention_forecaster_checkpoint,
    save_wsi_tile_attention_forecaster_checkpoint,
)


def _make_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int,
    generator: torch.Generator,
    *,
    with_attention: bool = True,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim, generator=generator)
    direction = torch.linspace(-1.0, 1.0, feature_dim)
    attention = torch.softmax(tile_features @ direction, dim=0) if with_attention else None

    coords = torch.stack(
        [
            torch.arange(n_tiles, dtype=torch.long),
            torch.arange(n_tiles, dtype=torch.long) + 100,
        ],
        dim=1,
    )

    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=coords,
        label=1,
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
                generator=generator,
                with_attention=with_attention,
            )
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


def _expected_selected_indices(
    *,
    checkpoint_path: Path,
    bag: WSIBag,
    keep_ratio: float,
) -> torch.Tensor:
    checkpoint = load_wsi_tile_attention_forecaster_checkpoint(checkpoint_path)
    model = checkpoint.model.eval()

    with torch.no_grad():
        scores = model(bag.tile_features)

    n_keep = max(1, int(torch.ceil(torch.tensor(bag.n_tiles * keep_ratio)).item()))
    n_keep = min(n_keep, bag.n_tiles)
    selected = torch.topk(scores, k=n_keep).indices
    return torch.sort(selected).values


def test_create_pruned_wsi_feature_store_cli_writes_pruned_store(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_store(input_store_path, with_attention=True)
    _make_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--input-feature-store",
            str(input_store_path),
            "--output-feature-store",
            str(output_store_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratio",
            "0.5",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"event": "done"' in result.stdout

    json_objects = []
    decoder = json.JSONDecoder()
    text = result.stdout.strip()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        obj, index = decoder.raw_decode(text, index)
        json_objects.append(obj)

    summary = json_objects[-1]
    assert summary["event"] == "done"
    assert summary["n_slides"] == 5
    assert summary["n_tiles_output_total"] < summary["n_tiles_input_total"]

    input_store = H5WSIFeatureStore(input_store_path)
    output_store = H5WSIFeatureStore(output_store_path)

    assert output_store.slide_ids() == input_store.slide_ids()

    for slide_id in input_store.slide_ids():
        original = input_store.read(slide_id)
        pruned = output_store.read(slide_id)
        selected = _expected_selected_indices(
            checkpoint_path=checkpoint_path,
            bag=original,
            keep_ratio=0.5,
        )

        assert pruned.n_tiles == selected.numel()
        assert torch.allclose(pruned.tile_features, original.tile_features[selected])
        assert pruned.coords is not None
        assert original.coords is not None
        assert torch.equal(pruned.coords, original.coords[selected])
        assert pruned.attention is not None
        assert original.attention is not None
        assert torch.allclose(pruned.attention, original.attention[selected])
        assert pruned.label == original.label
        assert pruned.metadata is not None
        assert pruned.metadata["source"] == "synthetic"
        assert pruned.metadata["pruned_by"] == "WSITileAttentionForecaster"
        assert pruned.metadata["pruning_preserved_original_tile_order"] is True


def test_create_pruned_wsi_feature_store_cli_supports_slide_subset(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"
    slide_ids_path = tmp_path / "slide_ids.txt"

    _make_store(input_store_path, with_attention=True)
    _make_forecaster_checkpoint(checkpoint_path)
    slide_ids_path.write_text("slide_001\nslide_003\n")

    subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--input-feature-store",
            str(input_store_path),
            "--output-feature-store",
            str(output_store_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratio",
            "0.5",
            "--slide-ids-file",
            str(slide_ids_path),
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    output_store = H5WSIFeatureStore(output_store_path)
    assert output_store.slide_ids() == ("slide_001", "slide_003")


def test_create_pruned_wsi_feature_store_cli_preserves_missing_attention(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_store(input_store_path, with_attention=False)
    _make_forecaster_checkpoint(checkpoint_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--input-feature-store",
            str(input_store_path),
            "--output-feature-store",
            str(output_store_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratio",
            "0.5",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    output_store = H5WSIFeatureStore(output_store_path)
    pruned = output_store.read("slide_000")
    assert pruned.attention is None


def test_create_pruned_wsi_feature_store_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_store(input_store_path, with_attention=True)
    _make_forecaster_checkpoint(checkpoint_path)
    output_store_path.write_text("already here")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--input-feature-store",
            str(input_store_path),
            "--output-feature-store",
            str(output_store_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratio",
            "0.5",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output feature store already exists" in result.stderr


def test_create_pruned_wsi_feature_store_cli_rejects_invalid_keep_ratio(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_store(input_store_path, with_attention=True)
    _make_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--input-feature-store",
            str(input_store_path),
            "--output-feature-store",
            str(output_store_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratio",
            "0.0",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--keep-ratio" in result.stderr
