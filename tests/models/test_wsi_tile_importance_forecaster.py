import pytest
import torch

from src.models.wsi import (
    WSI_TILE_IMPORTANCE_LOSS_TYPES,
    WSITileAttentionForecaster,
    WSITileImportanceForecaster,
    wsi_attention_kl_loss,
    wsi_tile_importance_loss,
    wsi_tile_importance_mse_loss,
    wsi_tile_importance_rank_loss,
    wsi_tile_importance_topk_bce_loss,
)


def test_wsi_tile_attention_forecaster_is_backward_compatible_alias() -> None:
    assert WSITileAttentionForecaster is WSITileImportanceForecaster


def test_wsi_tile_importance_forecaster_accepts_single_bag() -> None:
    model = WSITileImportanceForecaster(
        feature_dim=16,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    scores = model(torch.randn(7, 16))

    assert scores.shape == (7,)
    assert torch.is_floating_point(scores)


@pytest.mark.parametrize("loss", WSI_TILE_IMPORTANCE_LOSS_TYPES)
def test_wsi_tile_importance_loss_dispatcher_returns_finite_scalar(loss: str) -> None:
    scores = torch.randn(2, 6, requires_grad=True)
    target = torch.rand(2, 6) + 0.05

    value = wsi_tile_importance_loss(scores, target, loss=loss, top_k=2)

    assert value.ndim == 0
    assert torch.isfinite(value)
    assert value.item() >= 0.0


@pytest.mark.parametrize("loss", WSI_TILE_IMPORTANCE_LOSS_TYPES)
def test_wsi_tile_importance_loss_dispatcher_backpropagates(loss: str) -> None:
    scores = torch.randn(2, 6, requires_grad=True)
    target = torch.rand(2, 6) + 0.05

    value = wsi_tile_importance_loss(scores, target, loss=loss, top_k=2)
    value.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_wsi_tile_importance_loss_dispatcher_rejects_unknown_loss() -> None:
    scores = torch.randn(2, 6)
    target = torch.rand(2, 6) + 0.05

    with pytest.raises(ValueError, match="loss must be one of"):
        wsi_tile_importance_loss(scores, target, loss="not_a_loss")


def test_wsi_tile_importance_loss_kl_matches_legacy_kl_loss() -> None:
    scores = torch.tensor([0.2, 1.0, -0.5], dtype=torch.float32)
    target = torch.tensor([2.0, 6.0, 2.0], dtype=torch.float32)

    dispatched = wsi_tile_importance_loss(scores, target, loss="kl")
    legacy = wsi_attention_kl_loss(scores, target)

    assert torch.allclose(dispatched, legacy)


def test_wsi_tile_importance_mse_loss_is_zero_for_matching_distributions() -> None:
    scores = torch.tensor([0.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.0, 1.0, 1.0, 1.0])

    loss = wsi_tile_importance_mse_loss(scores, target)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_wsi_tile_importance_mse_loss_supports_masked_batched_inputs() -> None:
    scores = torch.tensor([[0.2, 1.0, -0.5, 99.0], [0.1, -0.1, 0.4, 99.0]])
    target = torch.tensor([[2.0, 6.0, 2.0, 1000.0], [0.0, 1.0, 3.0, 1000.0]])
    mask = torch.tensor([[True, True, True, False], [True, True, True, False]])

    loss = wsi_tile_importance_mse_loss(scores, target, mask=mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_wsi_tile_importance_mse_loss_rejects_zero_mass_target() -> None:
    scores = torch.tensor([0.2, 1.0, -0.5])
    target = torch.tensor([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="positive mass"):
        wsi_tile_importance_mse_loss(scores, target)


def test_wsi_tile_importance_topk_bce_loss_prefers_correct_top_k() -> None:
    target = torch.tensor([0.1, 0.9, 0.05, 0.05])
    good_scores = torch.tensor([-2.0, 2.0, -2.0, -2.0])
    bad_scores = torch.tensor([2.0, -2.0, -2.0, -2.0])

    good_loss = wsi_tile_importance_topk_bce_loss(good_scores, target, top_k=1)
    bad_loss = wsi_tile_importance_topk_bce_loss(bad_scores, target, top_k=1)

    assert good_loss.item() < bad_loss.item()


def test_wsi_tile_importance_topk_bce_loss_rejects_non_positive_top_k() -> None:
    scores = torch.rand(4)
    target = torch.rand(4) + 0.1

    with pytest.raises(ValueError, match="top_k must be positive"):
        wsi_tile_importance_topk_bce_loss(scores, target, top_k=0)


def test_wsi_tile_importance_topk_bce_loss_requires_two_valid_tiles() -> None:
    scores = torch.rand(4)
    target = torch.rand(4) + 0.1
    mask = torch.tensor([True, False, False, False])

    with pytest.raises(ValueError, match="at least two valid tiles"):
        wsi_tile_importance_topk_bce_loss(scores, target, mask=mask, top_k=1)


def test_wsi_tile_importance_rank_loss_prefers_correct_ranking() -> None:
    target = torch.tensor([0.1, 0.9, 0.2, 0.8])
    good_scores = torch.tensor([-1.0, 3.0, -1.0, 3.0])
    bad_scores = torch.tensor([3.0, -1.0, 3.0, -1.0])

    good_loss = wsi_tile_importance_rank_loss(good_scores, target, top_k=2)
    bad_loss = wsi_tile_importance_rank_loss(bad_scores, target, top_k=2)

    assert good_loss.item() < bad_loss.item()


def test_wsi_tile_importance_rank_loss_rejects_non_positive_margin() -> None:
    scores = torch.rand(4)
    target = torch.rand(4) + 0.1

    with pytest.raises(ValueError, match="margin must be positive"):
        wsi_tile_importance_rank_loss(scores, target, top_k=1, margin=0.0)


def test_wsi_tile_importance_loss_rejects_negative_target() -> None:
    scores = torch.rand(2, 4)
    target = torch.rand(2, 4)
    target[0, 0] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        wsi_tile_importance_mse_loss(scores, target)
