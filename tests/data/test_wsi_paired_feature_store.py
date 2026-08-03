import pytest
import torch

from src.data.wsi import (
    InMemoryWSIFeatureStore,
    PairedWSIBag,
    WSIBag,
    load_paired_wsi_bag,
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
        metadata={"source": "test"},
    )


def test_load_paired_wsi_bag_legacy_single_store_still_valid() -> None:
    coords = torch.arange(8, dtype=torch.long).reshape(4, 2)
    store = InMemoryWSIFeatureStore([_bag("slide_001", coords=coords)])
    direct = store.read("slide_001")

    paired = load_paired_wsi_bag(store, store, "slide_001", alignment_mode="index")

    assert paired.slide_id == "slide_001"
    assert torch.equal(paired.input_features, direct.tile_features)
    assert torch.equal(paired.target_importance, direct.attention)
    assert torch.equal(paired.coords, direct.coords)
    assert paired.label == 1


def test_load_paired_wsi_bag_index_aligned_separate_stores() -> None:
    coords = torch.arange(8, dtype=torch.long).reshape(4, 2)
    input_features = torch.randn(4, 8)
    attention = torch.rand(4) + 0.1

    input_bag = WSIBag(
        slide_id="slide_001",
        tile_features=input_features,
        coords=coords,
        label=1,
    )
    target_bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(4, 32),  # different (late) feature dim, irrelevant here
        coords=coords,
        attention=attention,
    )

    input_store = InMemoryWSIFeatureStore([input_bag])
    target_store = InMemoryWSIFeatureStore([target_bag])

    paired = load_paired_wsi_bag(
        input_store, target_store, "slide_001", alignment_mode="index"
    )

    assert torch.equal(paired.input_features, input_features)
    assert torch.equal(paired.target_importance, attention)
    assert paired.target_metadata["target_type"] == "tile_importance"


def test_load_paired_wsi_bag_coords_aligned_with_permuted_order() -> None:
    input_coords = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
    permuted_order = [2, 0, 3, 1]
    target_coords = input_coords[permuted_order]

    input_features = torch.randn(4, 8)
    attention = torch.rand(4) + 0.1

    input_bag = WSIBag(
        slide_id="slide_001", tile_features=input_features, coords=input_coords
    )
    target_bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(4, 8),
        coords=target_coords,
        attention=attention[permuted_order],
    )

    input_store = InMemoryWSIFeatureStore([input_bag])
    target_store = InMemoryWSIFeatureStore([target_bag])

    paired = load_paired_wsi_bag(
        input_store,
        target_store,
        "slide_001",
        alignment_mode="coords",
        require_coords=True,
    )

    assert torch.equal(paired.input_features, input_features)
    assert torch.allclose(paired.target_importance, attention)
    assert torch.equal(paired.coords, input_coords)


def test_load_paired_wsi_bag_index_mismatch_tile_count_fails() -> None:
    input_store = InMemoryWSIFeatureStore([_bag("slide_001", n_tiles=4)])
    target_store = InMemoryWSIFeatureStore([_bag("slide_001", n_tiles=5)])

    with pytest.raises(ValueError, match="equal tile counts"):
        load_paired_wsi_bag(input_store, target_store, "slide_001", alignment_mode="index")


def test_load_paired_wsi_bag_index_mismatch_coords_fails() -> None:
    coords_a = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
    coords_b = torch.tensor([[9, 9], [0, 1], [1, 0], [1, 1]], dtype=torch.long)

    input_store = InMemoryWSIFeatureStore([_bag("slide_001", coords=coords_a)])
    target_store = InMemoryWSIFeatureStore([_bag("slide_001", coords=coords_b)])

    with pytest.raises(ValueError, match="identical tile order"):
        load_paired_wsi_bag(input_store, target_store, "slide_001", alignment_mode="index")


def test_load_paired_wsi_bag_coords_duplicate_fails() -> None:
    input_coords = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
    target_coords_with_duplicate = torch.tensor(
        [[0, 0], [0, 0], [1, 0], [1, 1]], dtype=torch.long
    )

    input_store = InMemoryWSIFeatureStore([_bag("slide_001", coords=input_coords)])
    target_store = InMemoryWSIFeatureStore([_bag("slide_001", coords=target_coords_with_duplicate)])

    with pytest.raises(ValueError, match="duplicate tile coordinates"):
        load_paired_wsi_bag(input_store, target_store, "slide_001", alignment_mode="coords")


def test_load_paired_wsi_bag_missing_slide_raises_key_error() -> None:
    input_store = InMemoryWSIFeatureStore([_bag("slide_001")])
    target_store = InMemoryWSIFeatureStore([_bag("slide_002")])

    with pytest.raises(KeyError):
        load_paired_wsi_bag(input_store, target_store, "slide_001", alignment_mode="index")


def test_load_paired_wsi_bag_missing_target_attention_raises() -> None:
    input_store = InMemoryWSIFeatureStore([_bag("slide_001")])
    target_store = InMemoryWSIFeatureStore(
        [WSIBag(slide_id="slide_001", tile_features=torch.randn(4, 8))]
    )

    with pytest.raises(ValueError, match="no attention/importance target"):
        load_paired_wsi_bag(input_store, target_store, "slide_001", alignment_mode="index")


def test_load_paired_wsi_bag_require_coords_missing_raises() -> None:
    input_store = InMemoryWSIFeatureStore([_bag("slide_001", coords=None)])
    target_store = InMemoryWSIFeatureStore(
        [_bag("slide_001", coords=torch.zeros(4, 2, dtype=torch.long))]
    )

    with pytest.raises(ValueError, match="require_coords"):
        load_paired_wsi_bag(
            input_store,
            target_store,
            "slide_001",
            alignment_mode="index",
            require_coords=True,
        )


def test_paired_wsi_bag_rejects_non_finite_target_importance() -> None:
    target_importance = torch.rand(4)
    target_importance[0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        PairedWSIBag(
            slide_id="slide_001",
            input_features=torch.randn(4, 8),
            target_importance=target_importance,
        )


def test_paired_wsi_bag_rejects_target_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        PairedWSIBag(
            slide_id="slide_001",
            input_features=torch.randn(4, 8),
            target_importance=torch.rand(3),
        )


def test_paired_wsi_bag_to_wsi_bag_round_trips_target_as_attention() -> None:
    input_features = torch.randn(4, 8)
    target_importance = torch.rand(4) + 0.1

    paired = PairedWSIBag(
        slide_id="slide_001",
        input_features=input_features,
        target_importance=target_importance,
    )
    bag = paired.to_wsi_bag()

    assert isinstance(bag, WSIBag)
    assert torch.equal(bag.tile_features, input_features)
    assert torch.equal(bag.attention, target_importance)
