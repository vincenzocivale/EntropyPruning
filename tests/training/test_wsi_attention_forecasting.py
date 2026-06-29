import pytest
import torch
from torch.utils.data import DataLoader

from src.data.wsi import (
    InMemoryWSIBagDataset,
    WSIBag,
    collate_padded_wsi_bags,
)
from src.models.wsi import WSITileAttentionForecaster
from src.training.wsi import (
    WSIAttentionForecastingBatchOutput,
    run_wsi_attention_forecasting_batch,
)


def _make_bag(slide_id: str, n_tiles: int, feature_dim: int = 8) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=1,
        attention=torch.rand(n_tiles) + 0.1,
    )


def _make_batch():
    dataset = InMemoryWSIBagDataset(
        [
            _make_bag("slide_001", n_tiles=4),
            _make_bag("slide_002", n_tiles=7),
        ]
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_padded_wsi_bags,
    )
    return next(iter(loader))


def test_run_wsi_attention_forecasting_batch_returns_loss_scores_and_metrics() -> None:
    batch = _make_batch()
    model = WSITileAttentionForecaster(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    output = run_wsi_attention_forecasting_batch(model, batch, top_k=2)

    assert isinstance(output, WSIAttentionForecastingBatchOutput)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.scores.shape == batch.attention.shape
    assert set(output.metrics) == {"spearmanr", "top2_overlap", "ndcg_at_2"}
    assert all(metric.ndim == 0 for metric in output.metrics.values())
    assert all(torch.isfinite(metric) for metric in output.metrics.values())


def test_run_wsi_attention_forecasting_batch_loss_backpropagates() -> None:
    batch = _make_batch()
    model = WSITileAttentionForecaster(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    output = run_wsi_attention_forecasting_batch(model, batch, top_k=2)
    output.loss.backward()

    grads = [
        param.grad
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_run_wsi_attention_forecasting_batch_rejects_missing_attention() -> None:
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(4, 8),
        coords=torch.zeros(4, 2, dtype=torch.long),
        label=1,
        attention=None,
    )
    dataset = InMemoryWSIBagDataset([bag])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_padded_wsi_bags,
    )
    batch = next(iter(loader))
    model = WSITileAttentionForecaster(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="attention is required"):
        run_wsi_attention_forecasting_batch(model, batch, top_k=2)


def test_run_wsi_attention_forecasting_batch_rejects_non_positive_top_k() -> None:
    batch = _make_batch()
    model = WSITileAttentionForecaster(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        run_wsi_attention_forecasting_batch(model, batch, top_k=0)


def test_run_wsi_attention_forecasting_batch_rejects_wrong_batch_type() -> None:
    model = WSITileAttentionForecaster(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    with pytest.raises(TypeError, match="PaddedWSIBatch"):
        run_wsi_attention_forecasting_batch(model, object(), top_k=2)  # type: ignore[arg-type]
