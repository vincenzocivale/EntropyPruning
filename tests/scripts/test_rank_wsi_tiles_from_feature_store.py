import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag, WSIRankingStore
from src.models.wsi import (
    WSITileImportanceForecasterConfig,
    load_wsi_tile_importance_forecaster_checkpoint,
    save_wsi_tile_importance_forecaster_checkpoint,
)


def _make_bag(slide_id: str, n_tiles: int, feature_dim: int, seed: int) -> WSIBag:
    generator = torch.Generator().manual_seed(seed)
    tile_features = torch.randn(n_tiles, feature_dim, generator=generator)
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
        metadata={"source": "synthetic"},
    )


def _make_store(path: Path) -> None:
    store = H5WSIFeatureStore(path)
    for index, n_tiles in enumerate((4, 6, 5)):
        store.write(_make_bag(f"slide_{index:03d}", n_tiles, 8, 100 + index))


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
        loss="mse",
        metadata={"source": "unit_test"},
    )


def _expected_for_slide(
    checkpoint_path: Path,
    bag: WSIBag,
) -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = load_wsi_tile_importance_forecaster_checkpoint(checkpoint_path)
    model = checkpoint.model.eval()
    with torch.no_grad():
        scores = model(bag.tile_features)
    order = torch.argsort(scores, descending=True, stable=True)
    ranks = torch.empty(bag.n_tiles, dtype=torch.int64)
    ranks[order] = torch.arange(1, bag.n_tiles + 1, dtype=torch.int64)
    return scores, ranks


def test_rank_wsi_tiles_from_feature_store_cli_writes_rankings(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_store_path = tmp_path / "input.h5"
    output_dir = tmp_path / "rankings"
    checkpoint_path = tmp_path / "forecaster.pt"

    _make_store(input_store_path)
    _make_forecaster_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/rank_wsi_tiles_from_feature_store.py",
            "--input-feature-store",
            str(input_store_path),
            "--forecaster-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--keep-ratios",
            "0.25",
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

    decoder = json.JSONDecoder()
    objects = []
    text = result.stdout.strip()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        obj, index = decoder.raw_decode(text, index)
        objects.append(obj)

    summary = objects[-1]
    assert summary["event"] == "done"
    assert summary["n_slides"] == 3
    assert summary["feature_dim"] == 8

    input_store = H5WSIFeatureStore(input_store_path)
    ranking_store = WSIRankingStore(output_dir, file_format="npz")
    assert ranking_store.slide_ids() == input_store.slide_ids()

    for slide_id in input_store.slide_ids():
        bag = input_store.read(slide_id)
        ranking = ranking_store.read(slide_id)
        expected_scores, expected_ranks = _expected_for_slide(checkpoint_path, bag)

        assert ranking.slide_id == slide_id
        assert ranking.coords is not None
        assert torch.equal(ranking.coords, bag.coords)
        assert torch.allclose(ranking.scores, expected_scores.cpu())
        assert torch.equal(ranking.ranks, expected_ranks)
        assert ranking.metadata is not None
        assert ranking.metadata["ranking_checkpoint"] == str(checkpoint_path)
        assert ranking.metadata["ranking_input_feature_store"] == str(input_store_path)
        assert ranking.metadata["ranking_feature_dim"] == 8
        assert ranking.metadata["ranking_ranks_are_1_based"] is True

        selected_025 = ranking.selected_indices["0.25"]
        selected_05 = ranking.selected_indices["0.5"]
        assert selected_025.numel() == 1
        assert selected_05.numel() == max(1, (bag.n_tiles + 1) // 2)
        assert torch.equal(
            ranking.original_order_indices["0.5"],
            torch.sort(selected_05).values,
        )
