import pytest
import torch

from src.models.wsi import WSITileAttentionForecaster, wsi_attention_kl_loss


def test_wsi_tile_attention_forecaster_accepts_single_bag() -> None:
    model = WSITileAttentionForecaster(
        feature_dim=16,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    tile_features = torch.randn(7, 16)

    scores = model(tile_features)

    assert scores.shape == (7,)
    assert torch.is_floating_point(scores)


def test_wsi_tile_attention_forecaster_accepts_batched_bags() -> None:
    model = WSITileAttentionForecaster(
        feature_dim=16,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    tile_features = torch.randn(3, 7, 16)

    scores = model(tile_features)

    assert scores.shape == (3, 7)


def test_wsi_tile_attention_forecaster_masks_invalid_tiles() -> None:
    model = WSITileAttentionForecaster(
        feature_dim=16,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    tile_features = torch.randn(2, 5, 16)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
        ]
    )

    scores = model(tile_features, mask=mask)

    assert scores.shape == (2, 5)
    assert torch.isfinite(scores[mask]).all()
    assert (scores[~mask] < -1e20).all()


def test_wsi_tile_attention_forecaster_rejects_wrong_feature_dim() -> None:
    model = WSITileAttentionForecaster(
        feature_dim=16,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="expected feature_dim"):
        model(torch.randn(7, 15))


def test_wsi_tile_attention_forecaster_rejects_all_masked_sample() -> None:
    model = WSITileAttentionForecaster(
        feature_dim=16,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    tile_features = torch.randn(2, 5, 16)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [False, False, False, False, False],
        ]
    )

    with pytest.raises(ValueError, match="at least one valid tile"):
        model(tile_features, mask=mask)


def test_wsi_attention_kl_loss_accepts_unnormalized_target_attention() -> None:
    scores = torch.tensor([0.2, 1.0, -0.5], dtype=torch.float32)
    target_attention = torch.tensor([2.0, 6.0, 2.0], dtype=torch.float32)

    loss = wsi_attention_kl_loss(scores, target_attention)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_wsi_attention_kl_loss_supports_masked_batched_inputs() -> None:
    scores = torch.tensor(
        [
            [0.2, 1.0, -0.5, 99.0],
            [0.1, -0.1, 0.4, 99.0],
        ],
        dtype=torch.float32,
    )
    target_attention = torch.tensor(
        [
            [2.0, 6.0, 2.0, 1000.0],
            [0.0, 1.0, 3.0, 1000.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
        ]
    )

    loss = wsi_attention_kl_loss(scores, target_attention, mask=mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_wsi_attention_kl_loss_rejects_negative_target_attention() -> None:
    scores = torch.tensor([0.2, 1.0, -0.5], dtype=torch.float32)
    target_attention = torch.tensor([2.0, -1.0, 2.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="non-negative"):
        wsi_attention_kl_loss(scores, target_attention)


def test_wsi_attention_kl_loss_rejects_zero_mass_target_attention() -> None:
    scores = torch.tensor([0.2, 1.0, -0.5], dtype=torch.float32)
    target_attention = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="positive mass"):
        wsi_attention_kl_loss(scores, target_attention)


def test_wsi_attention_kl_loss_backpropagates_to_scores() -> None:
    scores = torch.randn(2, 5, requires_grad=True)
    target_attention = torch.rand(2, 5)

    loss = wsi_attention_kl_loss(scores, target_attention)
    loss.backward()

    assert scores.grad is not None
    assert scores.grad.shape == scores.shape
    assert torch.isfinite(scores.grad).all()
