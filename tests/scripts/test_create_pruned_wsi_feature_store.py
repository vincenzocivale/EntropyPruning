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
    WSITileImportanceForecasterConfig,
    load_wsi_tile_attention_forecaster_checkpoint,
    load_wsi_tile_importance_forecaster_checkpoint,
    save_wsi_tile_attention_forecaster_checkpoint,
    save_wsi_tile_importance_forecaster_checkpoint,
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


def _make_input_bag_with_coords(
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
        metadata={"source": "synthetic_selection"},
    )


def _make_materialize_bag(
    input_bag: WSIBag,
    *,
    late_feature_dim: int,
    generator: torch.Generator,
    permute: bool = False,
) -> WSIBag:
    late_features = torch.randn(input_bag.n_tiles, late_feature_dim, generator=generator)
    order = (
        torch.arange(input_bag.n_tiles - 1, -1, -1)
        if permute
        else torch.arange(input_bag.n_tiles)
    )

    return WSIBag(
        slide_id=input_bag.slide_id,
        tile_features=late_features[order],
        coords=input_bag.coords[order],
        label=input_bag.label,
        metadata={"source": "synthetic_materialize"},
    )


def _build_selection_and_materialize_stores(
    tmp_path: Path,
    *,
    n_slides: int = 5,
    late_feature_dim: int = 16,
    permute_materialize: bool = False,
    seed: int = 321,
) -> tuple[Path, Path]:
    selection_path = tmp_path / "selection.h5"
    materialize_path = tmp_path / "materialize.h5"

    generator = torch.Generator().manual_seed(seed)
    selection_store = H5WSIFeatureStore(selection_path)
    materialize_store = H5WSIFeatureStore(materialize_path)

    for index in range(n_slides):
        input_bag = _make_input_bag_with_coords(
            slide_id=f"slide_{index:03d}",
            n_tiles=6 + index,
            feature_dim=8,
            generator=generator,
        )
        selection_store.write(input_bag)
        materialize_store.write(
            _make_materialize_bag(
                input_bag,
                late_feature_dim=late_feature_dim,
                generator=generator,
                permute=permute_materialize,
            )
        )

    return selection_path, materialize_path


def _make_importance_forecaster_checkpoint(
    path: Path,
    *,
    feature_dim: int = 8,
    target_source: str | None = "synthetic_target",
) -> None:
    config = WSITileImportanceForecasterConfig(
        feature_dim=feature_dim,
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
        target_source=target_source,
        epoch=1,
        metrics={"val_loss": 1.0},
    )


def _expected_selected_indices_importance(
    *,
    checkpoint_path: Path,
    bag: WSIBag,
    keep_ratio: float,
) -> torch.Tensor:
    checkpoint = load_wsi_tile_importance_forecaster_checkpoint(checkpoint_path)
    model = checkpoint.model.eval()

    with torch.no_grad():
        scores = model(bag.tile_features)

    n_keep = max(1, int(torch.ceil(torch.tensor(bag.n_tiles * keep_ratio)).item()))
    n_keep = min(n_keep, bag.n_tiles)
    selected = torch.topk(scores, k=n_keep).indices
    return torch.sort(selected).values


def test_create_pruned_wsi_feature_store_cli_selection_mode_same_store(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    selection_store_path = tmp_path / "selection.h5"
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    generator = torch.Generator().manual_seed(7)
    selection_store = H5WSIFeatureStore(selection_store_path)
    for index in range(5):
        selection_store.write(
            _make_input_bag_with_coords(
                slide_id=f"slide_{index:03d}",
                n_tiles=6 + index,
                feature_dim=8,
                generator=generator,
            )
        )

    _make_importance_forecaster_checkpoint(checkpoint_path, target_source="synthetic_target")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--selection-feature-store",
            str(selection_store_path),
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

    assert '"event": "done"' in result.stdout

    output_store = H5WSIFeatureStore(output_store_path)
    for slide_id in selection_store.slide_ids():
        original = selection_store.read(slide_id)
        pruned = output_store.read(slide_id)
        selected = _expected_selected_indices_importance(
            checkpoint_path=checkpoint_path, bag=original, keep_ratio=0.5
        )

        assert pruned.n_tiles == selected.numel()
        assert torch.allclose(pruned.tile_features, original.tile_features[selected])
        assert torch.equal(pruned.coords, original.coords[selected])
        assert pruned.metadata["pruned_by"] == "WSITileImportanceForecaster"
        assert pruned.metadata["target_source"] == "synthetic_target"
        assert pruned.metadata["keep_ratio"] == 0.5
        assert pruned.metadata["n_tiles_original"] == original.n_tiles
        assert pruned.metadata["n_tiles_kept"] == selected.numel()
        assert pruned.metadata["selection_feature_store"] == str(selection_store_path)
        assert pruned.metadata["materialize_feature_store"] == str(selection_store_path)
        assert pruned.metadata["alignment_mode"] == "index"


def test_create_pruned_wsi_feature_store_cli_selection_differs_from_materialize_index_mode(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    selection_path, materialize_path = _build_selection_and_materialize_stores(tmp_path)
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_importance_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--selection-feature-store",
            str(selection_path),
            "--materialize-feature-store",
            str(materialize_path),
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

    assert '"event": "done"' in result.stdout

    selection_store = H5WSIFeatureStore(selection_path)
    materialize_store = H5WSIFeatureStore(materialize_path)
    output_store = H5WSIFeatureStore(output_store_path)

    for slide_id in selection_store.slide_ids():
        selection_bag = selection_store.read(slide_id)
        materialize_bag = materialize_store.read(slide_id)
        pruned = output_store.read(slide_id)

        selected = _expected_selected_indices_importance(
            checkpoint_path=checkpoint_path, bag=selection_bag, keep_ratio=0.5
        )

        assert pruned.n_tiles == selected.numel()
        assert pruned.feature_dim == materialize_bag.feature_dim
        assert torch.allclose(pruned.tile_features, materialize_bag.tile_features[selected])
        assert torch.equal(pruned.coords, materialize_bag.coords[selected])
        assert pruned.metadata["selection_feature_store"] == str(selection_path)
        assert pruned.metadata["materialize_feature_store"] == str(materialize_path)


def test_create_pruned_wsi_feature_store_cli_coords_alignment_reorders_materialize_tiles(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    selection_path, materialize_path = _build_selection_and_materialize_stores(
        tmp_path, permute_materialize=True
    )
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_importance_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--selection-feature-store",
            str(selection_path),
            "--materialize-feature-store",
            str(materialize_path),
            "--output-feature-store",
            str(output_store_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratio",
            "0.5",
            "--alignment-mode",
            "coords",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"event": "done"' in result.stdout

    selection_store = H5WSIFeatureStore(selection_path)
    materialize_store = H5WSIFeatureStore(materialize_path)
    output_store = H5WSIFeatureStore(output_store_path)

    for slide_id in selection_store.slide_ids():
        selection_bag = selection_store.read(slide_id)
        materialize_bag = materialize_store.read(slide_id)
        pruned = output_store.read(slide_id)

        selected = _expected_selected_indices_importance(
            checkpoint_path=checkpoint_path, bag=selection_bag, keep_ratio=0.5
        )
        expected_coords = selection_bag.coords[selected]

        materialize_coord_to_index = {
            tuple(row.tolist()): row_index
            for row_index, row in enumerate(materialize_bag.coords)
        }
        expected_materialize_indices = torch.tensor(
            [materialize_coord_to_index[tuple(row.tolist())] for row in expected_coords]
        )

        assert torch.equal(pruned.coords, expected_coords)
        assert torch.allclose(
            pruned.tile_features,
            materialize_bag.tile_features[expected_materialize_indices],
        )
        assert pruned.metadata["alignment_mode"] == "coords"


def test_create_pruned_wsi_feature_store_cli_index_mode_rejects_reordered_materialize_coords(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    selection_path, materialize_path = _build_selection_and_materialize_stores(
        tmp_path, permute_materialize=True
    )
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_importance_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--selection-feature-store",
            str(selection_path),
            "--materialize-feature-store",
            str(materialize_path),
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
    assert "index alignment requires identical tile order" in result.stderr


def test_create_pruned_wsi_feature_store_cli_new_mode_keep_ratio_floor_keeps_at_least_one_tile(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    selection_store_path = tmp_path / "selection.h5"
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    generator = torch.Generator().manual_seed(42)
    selection_store = H5WSIFeatureStore(selection_store_path)
    selection_store.write(
        _make_input_bag_with_coords(
            slide_id="slide_000", n_tiles=20, feature_dim=8, generator=generator
        )
    )

    _make_importance_forecaster_checkpoint(checkpoint_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--selection-feature-store",
            str(selection_store_path),
            "--output-feature-store",
            str(output_store_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--keep-ratio",
            "0.01",
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
    assert pruned.n_tiles == 1
    assert pruned.metadata["n_tiles_kept"] == 1


def test_create_pruned_wsi_feature_store_cli_rejects_conflicting_store_args(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    selection_path, materialize_path = _build_selection_and_materialize_stores(tmp_path, n_slides=2)
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_importance_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
            "--input-feature-store",
            str(selection_path),
            "--selection-feature-store",
            str(selection_path),
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
    assert "mutually exclusive" in result.stderr


def test_create_pruned_wsi_feature_store_cli_rejects_missing_store_args(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_store_path = tmp_path / "pruned.h5"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_importance_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_pruned_wsi_feature_store.py",
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
    assert "must be provided" in result.stderr
