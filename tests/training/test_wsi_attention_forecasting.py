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
    WSIAttentionForecastingEpochOutput,
    evaluate_wsi_attention_forecasting_epoch,
    run_wsi_attention_forecasting_batch,
    train_wsi_attention_forecasting_epoch,
)


def _make_bag(slide_id: str, n_tiles: int, feature_dim: int = 8) -> WSIBag:
    return WSIBag(
        slide_id=slide_id,
        tile_features=torch.randn(n_tiles, feature_dim),
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=1,
        attention=torch.rand(n_tiles) + 0.1,
    )


def _make_loader(batch_size: int = 2):
    dataset = InMemoryWSIBagDataset(
        [
            _make_bag("slide_001", n_tiles=4),
            _make_bag("slide_002", n_tiles=7),
            _make_bag("slide_003", n_tiles=5),
        ]
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_padded_wsi_bags,
    )


def _make_batch():
    return next(iter(_make_loader(batch_size=2)))


def _make_model() -> WSITileAttentionForecaster:
    return WSITileAttentionForecaster(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )


def test_run_wsi_attention_forecasting_batch_returns_loss_scores_and_metrics() -> None:
    batch = _make_batch()
    model = _make_model()

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
    model = _make_model()

    output = run_wsi_attention_forecasting_batch(model, batch, top_k=2)
    output.loss.backward()

    grads = [
        param.grad
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_train_wsi_attention_forecasting_epoch_updates_parameters() -> None:
    torch.manual_seed(0)
    loader = _make_loader(batch_size=2)
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    output = train_wsi_attention_forecasting_epoch(
        model,
        loader,
        optimizer,
        top_k=2,
    )

    after = {
        name: param.detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    assert isinstance(output, WSIAttentionForecastingEpochOutput)
    assert output.n_batches == 2
    assert output.n_bags == 3
    assert output.loss >= 0.0
    assert set(output.metrics) == {"spearmanr", "top2_overlap", "ndcg_at_2"}
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_evaluate_wsi_attention_forecasting_epoch_does_not_update_parameters() -> None:
    torch.manual_seed(0)
    loader = _make_loader(batch_size=2)
    model = _make_model()

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    output = evaluate_wsi_attention_forecasting_epoch(
        model,
        loader,
        top_k=2,
    )

    after = {
        name: param.detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    assert isinstance(output, WSIAttentionForecastingEpochOutput)
    assert output.n_batches == 2
    assert output.n_bags == 3
    assert output.loss >= 0.0
    assert set(output.metrics) == {"spearmanr", "top2_overlap", "ndcg_at_2"}
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_evaluate_wsi_attention_forecasting_epoch_restores_training_mode() -> None:
    loader = _make_loader(batch_size=2)
    model = _make_model()
    model.train()

    _ = evaluate_wsi_attention_forecasting_epoch(model, loader, top_k=2)

    assert model.training is True


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
    model = _make_model()

    with pytest.raises(ValueError, match="attention is required"):
        run_wsi_attention_forecasting_batch(model, batch, top_k=2)


def test_epoch_utilities_reject_non_positive_top_k() -> None:
    loader = _make_loader(batch_size=2)
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="top_k must be positive"):
        train_wsi_attention_forecasting_epoch(model, loader, optimizer, top_k=0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        evaluate_wsi_attention_forecasting_epoch(model, loader, top_k=0)


def test_train_wsi_attention_forecasting_epoch_rejects_non_positive_grad_clip() -> None:
    loader = _make_loader(batch_size=2)
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="grad_clip_norm"):
        train_wsi_attention_forecasting_epoch(
            model,
            loader,
            optimizer,
            top_k=2,
            grad_clip_norm=0.0,
        )


def test_run_wsi_attention_forecasting_batch_rejects_wrong_batch_type() -> None:
    model = _make_model()

    with pytest.raises(TypeError, match="PaddedWSIBatch"):
        run_wsi_attention_forecasting_batch(model, object(), top_k=2)  # type: ignore[arg-type]
