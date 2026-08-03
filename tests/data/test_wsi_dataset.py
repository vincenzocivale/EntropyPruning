import pytest
import torch
from torch.utils.data import DataLoader

from src.data.wsi import InMemoryWSIBagDataset, WSIBag, WSIBagDataset


def _make_bag(slide_id: str, n_tiles: int = 4, feature_dim: int = 8) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=1,
        attention=torch.rand(n_tiles),
    )


def test_in_memory_wsi_bag_dataset_is_a_wsi_bag_dataset() -> None:
    dataset = InMemoryWSIBagDataset([_make_bag("slide_001")])

    assert isinstance(dataset, WSIBagDataset)


def test_in_memory_wsi_bag_dataset_preserves_order() -> None:
    dataset = InMemoryWSIBagDataset(
        [
            _make_bag("slide_001"),
            _make_bag("slide_002"),
            _make_bag("slide_003"),
        ]
    )

    assert len(dataset) == 3
    assert dataset[0].slide_id == "slide_001"
    assert dataset[1].slide_id == "slide_002"
    assert dataset[2].slide_id == "slide_003"


def test_in_memory_wsi_bag_dataset_supports_iteration() -> None:
    dataset = InMemoryWSIBagDataset(
        [
            _make_bag("slide_001"),
            _make_bag("slide_002"),
        ]
    )

    assert [bag.slide_id for bag in dataset] == ["slide_001", "slide_002"]


def test_in_memory_wsi_bag_dataset_rejects_non_wsi_bag_items() -> None:
    with pytest.raises(TypeError, match="expects WSIBag"):
        InMemoryWSIBagDataset([_make_bag("slide_001"), object()])


def test_in_memory_wsi_bag_dataset_rejects_non_integer_index() -> None:
    dataset = InMemoryWSIBagDataset([_make_bag("slide_001")])

    with pytest.raises(TypeError, match="index must be an int"):
        _ = dataset["slide_001"]  # type: ignore[index]


def test_in_memory_wsi_bag_dataset_can_be_used_with_dataloader_without_batching() -> None:
    dataset = InMemoryWSIBagDataset([_make_bag("slide_001"), _make_bag("slide_002")])
    loader = DataLoader(dataset, batch_size=None, shuffle=False)

    first = next(iter(loader))

    assert isinstance(first, WSIBag)
    assert first.slide_id == "slide_001"


from src.data.wsi import FeatureStoreWSIBagDataset, InMemoryWSIFeatureStore, collate_wsi_bags


def test_feature_store_wsi_bag_dataset_uses_all_store_slide_ids_by_default() -> None:
    bags = [_make_bag("slide_001"), _make_bag("slide_002")]
    store = InMemoryWSIFeatureStore(bags)
    dataset = FeatureStoreWSIBagDataset(store)

    assert len(dataset) == 2
    assert dataset.slide_ids == ("slide_001", "slide_002")
    assert dataset[0] is bags[0]
    assert dataset[1] is bags[1]


def test_feature_store_wsi_bag_dataset_preserves_explicit_slide_id_order() -> None:
    bags = [_make_bag("slide_001"), _make_bag("slide_002"), _make_bag("slide_003")]
    store = InMemoryWSIFeatureStore(bags)
    dataset = FeatureStoreWSIBagDataset(store, slide_ids=["slide_003", "slide_001"])

    assert len(dataset) == 2
    assert dataset.slide_ids == ("slide_003", "slide_001")
    assert dataset[0].slide_id == "slide_003"
    assert dataset[1].slide_id == "slide_001"


def test_feature_store_wsi_bag_dataset_supports_iteration() -> None:
    bags = [_make_bag("slide_001"), _make_bag("slide_002")]
    store = InMemoryWSIFeatureStore(bags)
    dataset = FeatureStoreWSIBagDataset(store)

    assert [bag.slide_id for bag in dataset] == ["slide_001", "slide_002"]


def test_feature_store_wsi_bag_dataset_rejects_missing_slide_ids_by_default() -> None:
    store = InMemoryWSIFeatureStore([_make_bag("slide_001")])

    with pytest.raises(KeyError, match="not found"):
        FeatureStoreWSIBagDataset(store, slide_ids=["slide_001", "missing_slide"])


def test_feature_store_wsi_bag_dataset_can_defer_missing_slide_id_validation() -> None:
    store = InMemoryWSIFeatureStore([_make_bag("slide_001")])
    dataset = FeatureStoreWSIBagDataset(
        store,
        slide_ids=["slide_001", "missing_slide"],
        validate_slide_ids=False,
    )

    assert len(dataset) == 2
    assert dataset[0].slide_id == "slide_001"

    with pytest.raises(KeyError, match="not found"):
        _ = dataset[1]


def test_feature_store_wsi_bag_dataset_rejects_invalid_slide_ids() -> None:
    store = InMemoryWSIFeatureStore([_make_bag("slide_001")])

    with pytest.raises(ValueError, match="non-empty strings"):
        FeatureStoreWSIBagDataset(store, slide_ids=["slide_001", ""])


def test_feature_store_wsi_bag_dataset_rejects_non_integer_index() -> None:
    store = InMemoryWSIFeatureStore([_make_bag("slide_001")])
    dataset = FeatureStoreWSIBagDataset(store)

    with pytest.raises(TypeError, match="index must be an int"):
        _ = dataset["slide_001"]  # type: ignore[index]


def test_feature_store_wsi_bag_dataset_works_with_variable_length_collate() -> None:
    store = InMemoryWSIFeatureStore(
        [
            _make_bag("slide_001", n_tiles=4),
            _make_bag("slide_002", n_tiles=7),
        ]
    )
    dataset = FeatureStoreWSIBagDataset(store)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wsi_bags)

    batch = next(iter(loader))

    assert len(batch) == 2
    assert batch[0].slide_id == "slide_001"
    assert batch[1].slide_id == "slide_002"
    assert batch[0].n_tiles == 4
    assert batch[1].n_tiles == 7
