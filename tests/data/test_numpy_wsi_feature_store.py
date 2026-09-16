from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.wsi import NumpyWSIFeatureStore, WSIBag


def test_numpy_feature_store_roundtrip_and_overwrite(tmp_path):
    path = tmp_path / "features.npyd"
    store = NumpyWSIFeatureStore(path)
    first = WSIBag(slide_id="slide/one", tile_features=torch.ones((2, 3)),
                   coords=torch.tensor([[0, 1], [2, 3]]), label=1,
                   attention=torch.tensor([0.2, 0.8]), metadata={"model": "test"})
    store.write(first)
    assert store.slide_ids() == ("slide/one",)
    found = NumpyWSIFeatureStore(path, read_only=True).read("slide/one")
    assert torch.equal(found.tile_features, first.tile_features)
    assert torch.equal(found.coords, first.coords)
    assert found.label == 1
    assert found.metadata == {"model": "test"}
    store.write(WSIBag(slide_id="slide/one", tile_features=torch.zeros((1, 3))))
    assert store.slide_ids() == ("slide/one",)
    assert store.read("slide/one").tile_features.shape == (1, 3)
