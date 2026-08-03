from __future__ import annotations

import pytest
import torch

from src.data.wsi.bag import WSIBag
from src.data.wsi.h5_feature_store import H5WSIFeatureStore


def test_schema_v1_store_remains_readable_and_read_only(tmp_path) -> None:
    path = tmp_path / "legacy.h5"
    writable = H5WSIFeatureStore(path)
    original = WSIBag(
        slide_id="legacy-slide",
        tile_features=torch.randn(4, 3),
        coords=torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]),
        label=7,
        attention=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        metadata={"producer": "old-wsi-pipeline"},
    )
    writable.write(original)

    store = H5WSIFeatureStore(path, read_only=True)
    restored = store.read("legacy-slide")
    assert torch.equal(restored.tile_features, original.tile_features)
    assert torch.equal(restored.attention, original.attention)
    assert restored.label == 7
    assert restored.metadata == original.metadata
    with pytest.raises(PermissionError):
        store.write(original)
