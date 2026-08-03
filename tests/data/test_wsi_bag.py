import pytest
import torch

from src.data.wsi import WSIBag


def test_wsi_bag_valid_minimal() -> None:
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(8, 16),
    )

    assert bag.slide_id == "slide_001"
    assert bag.n_tiles == 8
    assert bag.feature_dim == 16
    assert bag.coords is None
    assert bag.attention is None


def test_wsi_bag_valid_with_coords_attention_and_label() -> None:
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(8, 16),
        coords=torch.arange(16).reshape(8, 2),
        label=torch.tensor(1),
        attention=torch.rand(8),
        metadata={"source": "synthetic"},
    )

    assert bag.n_tiles == 8
    assert bag.coords is not None
    assert bag.coords.shape == (8, 2)
    assert bag.attention is not None
    assert bag.attention.shape == (8,)
    assert bag.metadata == {"source": "synthetic"}


def test_wsi_bag_rejects_empty_slide_id() -> None:
    with pytest.raises(ValueError, match="slide_id"):
        WSIBag(slide_id="", tile_features=torch.randn(8, 16))


def test_wsi_bag_rejects_non_2d_tile_features() -> None:
    with pytest.raises(ValueError, match="tile_features must have shape"):
        WSIBag(slide_id="slide_001", tile_features=torch.randn(8, 16, 2))


def test_wsi_bag_rejects_integer_tile_features() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        WSIBag(slide_id="slide_001", tile_features=torch.ones(8, 16, dtype=torch.long))


def test_wsi_bag_rejects_coords_with_wrong_number_of_rows() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        WSIBag(
            slide_id="slide_001",
            tile_features=torch.randn(8, 16),
            coords=torch.zeros(7, 2),
        )


def test_wsi_bag_rejects_coords_with_wrong_width() -> None:
    with pytest.raises(ValueError, match="second dimension"):
        WSIBag(
            slide_id="slide_001",
            tile_features=torch.randn(8, 16),
            coords=torch.zeros(8, 3),
        )


def test_wsi_bag_rejects_attention_with_wrong_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        WSIBag(
            slide_id="slide_001",
            tile_features=torch.randn(8, 16),
            attention=torch.rand(7),
        )


def test_wsi_bag_rejects_non_finite_attention() -> None:
    attention = torch.rand(8)
    attention[0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        WSIBag(
            slide_id="slide_001",
            tile_features=torch.randn(8, 16),
            attention=attention,
        )


def test_wsi_bag_to_changes_float_dtype_only() -> None:
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(8, 16, dtype=torch.float32),
        coords=torch.arange(16, dtype=torch.long).reshape(8, 2),
        label=torch.tensor(1),
        attention=torch.rand(8, dtype=torch.float32),
    )

    moved = bag.to(dtype=torch.float16)

    assert moved.tile_features.dtype == torch.float16
    assert moved.attention is not None
    assert moved.attention.dtype == torch.float16
    assert moved.coords is not None
    assert moved.coords.dtype == torch.long
    assert isinstance(moved.label, torch.Tensor)
    assert moved.label.dtype == torch.long
