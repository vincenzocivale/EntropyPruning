import pytest
import torch
from torch.utils.data import DataLoader

from src.data.wsi import (
    InMemoryWSIBagDataset,
    PaddedWSIBatch,
    WSIBag,
    collate_wsi_bags,
    pad_wsi_bags,
)
from src.models.wsi import WSITileAttentionForecaster, wsi_attention_kl_loss


def _make_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int = 8,
    *,
    with_coords: bool = True,
    with_attention: bool = True,
) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=torch.arange(n_tiles * 2, dtype=torch.long).reshape(n_tiles, 2)
        if with_coords
        else None,
        label=1,
        attention=torch.rand(n_tiles) if with_attention else None,
        metadata={"split": "train"},
    )


def test_pad_wsi_bags_returns_padded_batch() -> None:
    bags = [
        _make_bag("slide_001", n_tiles=4),
        _make_bag("slide_002", n_tiles=7),
    ]

    batch = pad_wsi_bags(bags)

    assert isinstance(batch, PaddedWSIBatch)
    assert batch.slide_ids == ("slide_001", "slide_002")
    assert batch.tile_features.shape == (2, 7, 8)
    assert batch.mask.shape == (2, 7)
    assert batch.mask.tolist() == [
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True],
    ]
    assert batch.coords is not None
    assert batch.coords.shape == (2, 7, 2)
    assert batch.attention is not None
    assert batch.attention.shape == (2, 7)
    assert batch.labels == (1, 1)
    assert batch.metadata == ({"split": "train"}, {"split": "train"})


def test_pad_wsi_bags_preserves_feature_values_on_valid_tiles() -> None:
    bags = [
        _make_bag("slide_001", n_tiles=4),
        _make_bag("slide_002", n_tiles=7),
    ]

    batch = pad_wsi_bags(bags)

    assert torch.equal(batch.tile_features[0, :4], bags[0].tile_features)
    assert torch.equal(batch.tile_features[1, :7], bags[1].tile_features)
    assert batch.tile_features[0, 4:].eq(0.0).all()


def test_pad_wsi_bags_supports_bags_without_coords_or_attention() -> None:
    bags = [
        _make_bag("slide_001", n_tiles=4, with_coords=False, with_attention=False),
        _make_bag("slide_002", n_tiles=7, with_coords=False, with_attention=False),
    ]

    batch = pad_wsi_bags(bags)

    assert batch.coords is None
    assert batch.attention is None
    assert batch.tile_features.shape == (2, 7, 8)


def test_pad_wsi_bags_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        pad_wsi_bags([])


def test_pad_wsi_bags_rejects_mixed_feature_dims() -> None:
    bags = [
        _make_bag("slide_001", n_tiles=4, feature_dim=8),
        _make_bag("slide_002", n_tiles=7, feature_dim=16),
    ]

    with pytest.raises(ValueError, match="same feature_dim"):
        pad_wsi_bags(bags)


def test_pad_wsi_bags_rejects_mixed_coords_presence() -> None:
    bags = [
        _make_bag("slide_001", n_tiles=4, with_coords=True),
        _make_bag("slide_002", n_tiles=7, with_coords=False),
    ]

    with pytest.raises(ValueError, match="all bags must have coords"):
        pad_wsi_bags(bags)


def test_pad_wsi_bags_rejects_mixed_attention_presence() -> None:
    bags = [
        _make_bag("slide_001", n_tiles=4, with_attention=True),
        _make_bag("slide_002", n_tiles=7, with_attention=False),
    ]

    with pytest.raises(ValueError, match="all bags must have attention"):
        pad_wsi_bags(bags)


def test_padded_wsi_batch_to_changes_float_dtype_only() -> None:
    bags = [_make_bag("slide_001", n_tiles=4)]
    batch = pad_wsi_bags(bags)

    moved = batch.to(dtype=torch.float16)

    assert moved.tile_features.dtype == torch.float16
    assert moved.attention is not None
    assert moved.attention.dtype == torch.float16
    assert moved.coords is not None
    assert moved.coords.dtype == torch.long
    assert moved.mask.dtype == torch.bool


def test_padded_wsi_batch_connects_dataloader_to_forecaster_and_loss() -> None:
    dataset = InMemoryWSIBagDataset(
        [
            _make_bag("slide_001", n_tiles=4, feature_dim=8),
            _make_bag("slide_002", n_tiles=7, feature_dim=8),
        ]
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_wsi_bags,
        shuffle=False,
    )
    bags = next(iter(loader))
    batch = pad_wsi_bags(bags)

    model = WSITileAttentionForecaster(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    scores = model(batch.tile_features, mask=batch.mask)

    assert scores.shape == batch.attention.shape
    assert batch.attention is not None

    loss = wsi_attention_kl_loss(scores, batch.attention, mask=batch.mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
