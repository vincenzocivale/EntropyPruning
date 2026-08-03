import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import FeatureStoreWSIBagDataset, H5WSIFeatureStore, WSIBag


def _make_bag(slide_id: str, n_tiles: int = 4, feature_dim: int = 8) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=torch.arange(n_tiles * 2, dtype=torch.long).reshape(n_tiles, 2),
        label=1,
        attention=torch.rand(n_tiles),
        metadata={"source": "synthetic", "fold": 0},
    )


def test_h5_wsi_feature_store_writes_and_reads_bag(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)
    bag = _make_bag("slide_001")

    store.write(bag)
    loaded = store.read("slide_001")

    assert loaded.slide_id == bag.slide_id
    assert torch.equal(loaded.tile_features, bag.tile_features)
    assert loaded.coords is not None
    assert torch.equal(loaded.coords, bag.coords)
    assert loaded.attention is not None
    assert torch.equal(loaded.attention, bag.attention)
    assert loaded.label == 1
    assert loaded.metadata == {"source": "synthetic", "fold": 0}


def test_h5_wsi_feature_store_preserves_slide_order(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)

    store.write(_make_bag("slide_001"))
    store.write(_make_bag("slide_002"))
    store.write(_make_bag("slide_003"))

    assert store.slide_ids() == ("slide_001", "slide_002", "slide_003")


def test_h5_wsi_feature_store_overwrites_existing_slide(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)

    first = _make_bag("slide_001", n_tiles=4)
    second = _make_bag("slide_001", n_tiles=7)

    store.write(first)
    store.write(second)

    loaded = store.read("slide_001")

    assert store.slide_ids() == ("slide_001",)
    assert loaded.n_tiles == 7
    assert torch.equal(loaded.tile_features, second.tile_features)


def test_h5_wsi_feature_store_supports_tensor_label(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(4, 8),
        label=torch.tensor([1.0, 2.0]),
    )

    store.write(bag)
    loaded = store.read("slide_001")

    assert isinstance(loaded.label, torch.Tensor)
    assert torch.equal(loaded.label, torch.tensor([1.0, 2.0]))


def test_h5_wsi_feature_store_supports_bags_without_optional_fields(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(4, 8),
        coords=None,
        label=None,
        attention=None,
        metadata=None,
    )

    store.write(bag)
    loaded = store.read("slide_001")

    assert loaded.coords is None
    assert loaded.label is None
    assert loaded.attention is None
    assert loaded.metadata is None


def test_h5_wsi_feature_store_supports_slide_ids_with_slashes(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)
    bag = _make_bag("patient_001/slide_A")

    store.write(bag)

    assert store.exists("patient_001/slide_A")
    assert store.read("patient_001/slide_A").slide_id == "patient_001/slide_A"


def test_h5_wsi_feature_store_raises_for_missing_slide(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)

    with pytest.raises(KeyError, match="not found"):
        store.read("missing_slide")


def test_h5_wsi_feature_store_rejects_non_string_slide_id(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)

    with pytest.raises(TypeError, match="slide_id must be a str"):
        store.exists(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="slide_id must be a str"):
        store.read(123)  # type: ignore[arg-type]


def test_h5_wsi_feature_store_rejects_non_json_metadata(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(4, 8),
        metadata={"bad": object()},
    )

    with pytest.raises(TypeError, match="JSON-serializable"):
        store.write(bag)


def test_h5_wsi_feature_store_works_with_feature_store_dataset(tmp_path) -> None:
    path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(path)
    store.write(_make_bag("slide_001"))
    store.write(_make_bag("slide_002"))

    dataset = FeatureStoreWSIBagDataset(store, slide_ids=["slide_002", "slide_001"])

    assert len(dataset) == 2
    assert dataset[0].slide_id == "slide_002"
    assert dataset[1].slide_id == "slide_001"
