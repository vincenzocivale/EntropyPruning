import pytest
import torch

from src.data.wsi import InMemoryWSIFeatureStore, WSIBag, WSIFeatureStore


def _make_bag(slide_id: str, n_tiles: int = 4, feature_dim: int = 8) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=1,
        attention=torch.rand(n_tiles),
    )


def test_in_memory_wsi_feature_store_is_a_feature_store() -> None:
    store = InMemoryWSIFeatureStore()

    assert isinstance(store, WSIFeatureStore)


def test_in_memory_wsi_feature_store_starts_empty() -> None:
    store = InMemoryWSIFeatureStore()

    assert len(store) == 0
    assert store.slide_ids() == ()
    assert store.exists("slide_001") is False
    assert "slide_001" not in store


def test_in_memory_wsi_feature_store_writes_and_reads_bags() -> None:
    bag = _make_bag("slide_001")
    store = InMemoryWSIFeatureStore()

    store.write(bag)

    assert len(store) == 1
    assert store.exists("slide_001") is True
    assert "slide_001" in store
    assert store.slide_ids() == ("slide_001",)
    assert store.read("slide_001") is bag


def test_in_memory_wsi_feature_store_can_be_initialized_from_bags() -> None:
    bags = [_make_bag("slide_001"), _make_bag("slide_002")]
    store = InMemoryWSIFeatureStore(bags)

    assert len(store) == 2
    assert store.slide_ids() == ("slide_001", "slide_002")
    assert store.read("slide_001") is bags[0]
    assert store.read("slide_002") is bags[1]


def test_in_memory_wsi_feature_store_overwrites_existing_slide_id() -> None:
    first = _make_bag("slide_001", n_tiles=4)
    second = _make_bag("slide_001", n_tiles=7)
    store = InMemoryWSIFeatureStore([first])

    store.write(second)

    assert len(store) == 1
    assert store.read("slide_001") is second
    assert store.read("slide_001").n_tiles == 7


def test_in_memory_wsi_feature_store_rejects_non_bag_write() -> None:
    store = InMemoryWSIFeatureStore()

    with pytest.raises(TypeError, match="expects a WSIBag"):
        store.write(object())  # type: ignore[arg-type]


def test_in_memory_wsi_feature_store_rejects_non_string_slide_id() -> None:
    store = InMemoryWSIFeatureStore()

    with pytest.raises(TypeError, match="slide_id must be a str"):
        store.exists(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="slide_id must be a str"):
        store.read(123)  # type: ignore[arg-type]


def test_in_memory_wsi_feature_store_raises_for_missing_slide_id() -> None:
    store = InMemoryWSIFeatureStore()

    with pytest.raises(KeyError, match="slide_id not found"):
        store.read("missing_slide")
