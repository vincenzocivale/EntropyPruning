from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.wsi.wsi_forecaster_dataset import (
    WSIForecasterDataset,
    WSIForecasterManifestConfig,
    build_attention_manifest_csv,
)
from src.wsi_pipeline.io import WSIOutputRecord, write_wsi_output_record


def _write_synthetic_wsi_eaf(path: Path, *, n_tiles: int, hidden_dim: int, layer: int = 0) -> None:
    rng = np.random.default_rng(0)
    coords = np.column_stack([np.arange(n_tiles), np.zeros(n_tiles)]).astype(np.int32)
    # [n_layers, n_heads, n_tiles], matching the real TITAN capture shape (see
    # titan_attention.py's global_to_tiles_mass_share) -- ManifestAttentionSource is
    # configured with tile_axis=2, selections={0: target_layer}, reduction="mean".
    attention = rng.random((2, 3, n_tiles)).astype(np.float32)
    attention /= attention.sum(axis=-1, keepdims=True)
    hidden = rng.normal(size=(n_tiles, hidden_dim)).astype(np.float32)
    write_wsi_output_record(
        path,
        WSIOutputRecord(
            slide_id=path.stem,
            slide_embedding=rng.normal(size=(hidden_dim,)).astype(np.float32),
            coords=coords,
            attention={"global_to_tiles_mass_share": attention},
            auxiliary={f"hidden_layer_{layer:03d}": hidden},
        ),
    )


def _make_dataset(tmp_path: Path, *, hidden_layer: int | None) -> WSIForecasterDataset:
    slide_id = "slideA"
    wsi_eaf_path = tmp_path / f"{slide_id}.h5"
    # The file always only has hidden_layer_000; requesting any other layer must
    # raise cleanly (test_missing_hidden_layer_key_raises_clearly below).
    _write_synthetic_wsi_eaf(wsi_eaf_path, n_tiles=6, hidden_dim=4, layer=0)

    manifest = pd.DataFrame(
        [
            {
                "slide_id": slide_id,
                "case_id": slide_id,
                "project": "test",
                "tile_path": str(wsi_eaf_path),  # unused in hidden_layer mode
                "attention_path": str(wsi_eaf_path),
                "split": "train",
            }
        ]
    )
    attention_manifest_path = tmp_path / "attention_manifest.csv"
    build_attention_manifest_csv(manifest, attention_manifest_path)

    config = WSIForecasterManifestConfig(
        tile_eaf_root=tmp_path,
        wsi_eaf_root=tmp_path,
        hidden_layer=hidden_layer,
    )
    return WSIForecasterDataset(manifest, attention_manifest_path=attention_manifest_path, config=config, split="train")


def test_hidden_layer_bag_matches_auxiliary_array(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, hidden_layer=0)
    features, coords, target, slide_id = dataset[0]
    assert slide_id == "slideA"
    assert features.shape == (6, 4)
    assert coords.shape == (6, 2)
    assert torch.isfinite(features).all()
    assert torch.isclose(target.sum(), torch.tensor(1.0), atol=1e-5)


def test_missing_hidden_layer_key_raises_clearly(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, hidden_layer=3)  # file only has hidden_layer_000
    with pytest.raises(KeyError):
        dataset[0]
