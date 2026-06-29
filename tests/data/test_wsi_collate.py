import pytest
import torch
from torch.utils.data import DataLoader

from src.data.wsi import InMemoryWSIBagDataset, WSIBag, collate_wsi_bags


def _make_bag(slide_id: str, n_tiles: int, feature_dim: int = 8) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=1,
        attention=torch.rand(n_tiles),
    )


def test_collate_wsi_bags_preserves_variable_length_bags() -> None:
    bags = [
        _make_bag("slide_001", n_tiles=4),
        _make_bag("slide_002", n_tiles=7),
    ]

    batch = collate_wsi_bags(bags)

    assert isinstance(batch, list)
    assert len(batch) == 2
    assert batch[0].slide_id == "slide_001"
    assert batch[1].slide_id == "slide_002"
    assert batch[0].tile_features.shape == (4, 8)
    assert batch[1].tile_features.shape == (7, 8)


def test_collate_wsi_bags_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one"):
        collate_wsi_bags([])


def test_collate_wsi_bags_rejects_non_wsi_bag_items() -> None:
    with pytest.raises(TypeError, match="expects WSIBag"):
        collate_wsi_bags([_make_bag("slide_001", n_tiles=4), object()])  # type: ignore[list-item]


def test_dataloader_with_collate_wsi_bags_supports_batch_size_greater_than_one() -> None:
    dataset = InMemoryWSIBagDataset(
        [
            _make_bag("slide_001", n_tiles=4),
            _make_bag("slide_002", n_tiles=7),
            _make_bag("slide_003", n_tiles=5),
        ]
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_wsi_bags,
    )

    first_batch = next(iter(loader))

    assert isinstance(first_batch, list)
    assert len(first_batch) == 2
    assert [bag.slide_id for bag in first_batch] == ["slide_001", "slide_002"]
    assert first_batch[0].n_tiles == 4
    assert first_batch[1].n_tiles == 7
