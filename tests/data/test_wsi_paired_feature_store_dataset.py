import pytest
import torch
from torch.utils.data import DataLoader

from src.data.wsi import (
    InMemoryWSIFeatureStore,
    PairedFeatureStoreWSIBagDataset,
    WSIBag,
    collate_padded_wsi_bags,
)


def _bag(
    slide_id: str,
    *,
    n_tiles: int = 4,
    feature_dim: int = 8,
    coords: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=coords,
        label=1,
        attention=attention if attention is not None else (torch.rand(n_tiles) + 0.1),
    )


def _paired_stores(n_slides: int = 3) -> tuple[InMemoryWSIFeatureStore, InMemoryWSIFeatureStore]:
    input_store = InMemoryWSIFeatureStore()
    target_store = InMemoryWSIFeatureStore()
    for index in range(n_slides):
        n_tiles = 4 + index
        coords = torch.arange(n_tiles * 2, dtype=torch.long).reshape(n_tiles, 2)
        slide_id = f"slide_{index:03d}"
        input_store.write(
            WSIBag(
                slide_id=slide_id,
                tile_features=torch.randn(n_tiles, 8),
                coords=coords,
                label=1,
            )
        )
        target_store.write(
            WSIBag(
                slide_id=slide_id,
                tile_features=torch.zeros(n_tiles, 1),
                coords=coords,
                attention=torch.rand(n_tiles) + 0.1,
            )
        )
    return input_store, target_store


def test_paired_feature_store_dataset_yields_wsi_bags_with_target_as_attention() -> None:
    input_store, target_store = _paired_stores(n_slides=3)

    dataset = PairedFeatureStoreWSIBagDataset(input_store, target_store)

    assert len(dataset) == 3
    bag = dataset[0]
    assert isinstance(bag, WSIBag)
    assert bag.attention is not None
    assert bag.feature_dim == 8


def test_paired_feature_store_dataset_supports_coords_alignment() -> None:
    input_store, target_store = _paired_stores(n_slides=2)

    dataset = PairedFeatureStoreWSIBagDataset(
        input_store,
        target_store,
        alignment_mode="coords",
        require_coords=True,
    )

    for index in range(len(dataset)):
        bag = dataset[index]
        assert bag.attention.shape[0] == bag.n_tiles


def test_paired_feature_store_dataset_works_with_dataloader_collate() -> None:
    input_store, target_store = _paired_stores(n_slides=3)
    dataset = PairedFeatureStoreWSIBagDataset(input_store, target_store)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_padded_wsi_bags)
    batch = next(iter(loader))

    assert batch.tile_features.shape[0] == 2
    assert batch.attention is not None
    assert batch.attention.shape == batch.mask.shape


def test_paired_feature_store_dataset_iterates_all_bags() -> None:
    input_store, target_store = _paired_stores(n_slides=3)
    dataset = PairedFeatureStoreWSIBagDataset(input_store, target_store)

    slide_ids = [bag.slide_id for bag in dataset]
    assert slide_ids == ["slide_000", "slide_001", "slide_002"]


def test_paired_feature_store_dataset_rejects_missing_slide_in_input_store() -> None:
    input_store, target_store = _paired_stores(n_slides=2)

    with pytest.raises(KeyError, match="input feature store"):
        PairedFeatureStoreWSIBagDataset(
            input_store, target_store, slide_ids=["slide_000", "does_not_exist"]
        )


def test_paired_feature_store_dataset_rejects_missing_slide_in_target_store() -> None:
    input_store, target_store = _paired_stores(n_slides=2)
    input_store.write(_bag("slide_999", coords=torch.zeros(4, 2, dtype=torch.long)))

    with pytest.raises(KeyError, match="target feature store"):
        PairedFeatureStoreWSIBagDataset(
            input_store, target_store, slide_ids=["slide_000", "slide_999"]
        )


def test_paired_feature_store_dataset_rejects_non_int_index() -> None:
    input_store, target_store = _paired_stores(n_slides=2)
    dataset = PairedFeatureStoreWSIBagDataset(input_store, target_store)

    with pytest.raises(TypeError, match="index must be an int"):
        dataset["0"]  # type: ignore[index]


def test_paired_feature_store_dataset_legacy_single_store_supported() -> None:
    store, _ = _paired_stores(n_slides=1)
    # legacy case: same store passed as both input and target.
    single_store = InMemoryWSIFeatureStore(
        [
            WSIBag(
                slide_id="fused_slide",
                tile_features=torch.randn(4, 8),
                attention=torch.rand(4) + 0.1,
            )
        ]
    )

    dataset = PairedFeatureStoreWSIBagDataset(single_store, single_store)
    bag = dataset[0]

    assert bag.slide_id == "fused_slide"
    assert bag.attention is not None
