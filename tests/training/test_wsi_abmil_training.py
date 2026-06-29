import pytest
import torch
from torch.utils.data import DataLoader

from src.data.wsi import InMemoryWSIBagDataset, WSIBag, collate_padded_wsi_bags
from src.models.wsi import ABMILClassifier
from src.training.wsi import (
    ABMILClassificationBatchOutput,
    ABMILClassificationEpochOutput,
    evaluate_abmil_classification_epoch,
    run_abmil_classification_batch,
    train_abmil_classification_epoch,
)


def _make_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int = 8,
    label: int = 0,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim)
    tile_features[:, 0] += 1.0 if label == 1 else -1.0

    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=label,
        attention=None,
    )


def _make_loader(batch_size: int = 2) -> DataLoader:
    dataset = InMemoryWSIBagDataset(
        [
            _make_bag("slide_001", n_tiles=4, label=0),
            _make_bag("slide_002", n_tiles=7, label=1),
            _make_bag("slide_003", n_tiles=5, label=0),
            _make_bag("slide_004", n_tiles=6, label=1),
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


def _make_model() -> ABMILClassifier:
    return ABMILClassifier(
        feature_dim=8,
        hidden_dim=16,
        n_classes=2,
        dropout=0.0,
    )


def test_run_abmil_classification_batch_returns_loss_logits_attention_and_metrics() -> None:
    batch = _make_batch()
    model = _make_model()

    output = run_abmil_classification_batch(model, batch)

    assert isinstance(output, ABMILClassificationBatchOutput)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (2, 2)
    assert output.attention.shape == batch.mask.shape
    assert set(output.metrics) == {"accuracy"}
    assert output.metrics["accuracy"].ndim == 0
    assert 0.0 <= output.metrics["accuracy"].item() <= 1.0


def test_run_abmil_classification_batch_loss_backpropagates() -> None:
    batch = _make_batch()
    model = _make_model()

    output = run_abmil_classification_batch(model, batch)
    output.loss.backward()

    grads = [
        param.grad
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_train_abmil_classification_epoch_updates_parameters() -> None:
    torch.manual_seed(0)
    loader = _make_loader(batch_size=2)
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    output = train_abmil_classification_epoch(model, loader, optimizer)

    after = {
        name: param.detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    assert isinstance(output, ABMILClassificationEpochOutput)
    assert output.n_batches == 2
    assert output.n_bags == 4
    assert output.loss >= 0.0
    assert set(output.metrics) == {"accuracy"}
    assert 0.0 <= output.metrics["accuracy"] <= 1.0
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_evaluate_abmil_classification_epoch_does_not_update_parameters() -> None:
    torch.manual_seed(0)
    loader = _make_loader(batch_size=2)
    model = _make_model()

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    output = evaluate_abmil_classification_epoch(model, loader)

    after = {
        name: param.detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    assert isinstance(output, ABMILClassificationEpochOutput)
    assert output.n_batches == 2
    assert output.n_bags == 4
    assert output.loss >= 0.0
    assert set(output.metrics) == {"accuracy"}
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_evaluate_abmil_classification_epoch_restores_training_mode() -> None:
    loader = _make_loader(batch_size=2)
    model = _make_model()
    model.train()

    _ = evaluate_abmil_classification_epoch(model, loader)

    assert model.training is True


def test_run_abmil_classification_batch_rejects_missing_label() -> None:
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(4, 8),
        coords=torch.zeros(4, 2, dtype=torch.long),
        label=None,
    )
    loader = DataLoader(
        InMemoryWSIBagDataset([bag]),
        batch_size=1,
        collate_fn=collate_padded_wsi_bags,
    )
    model = _make_model()

    with pytest.raises(ValueError, match="missing"):
        run_abmil_classification_batch(model, next(iter(loader)))


def test_run_abmil_classification_batch_rejects_non_integer_float_label() -> None:
    bag = WSIBag(
        slide_id="slide_001",
        tile_features=torch.randn(4, 8),
        coords=torch.zeros(4, 2, dtype=torch.long),
        label=0.5,
    )
    loader = DataLoader(
        InMemoryWSIBagDataset([bag]),
        batch_size=1,
        collate_fn=collate_padded_wsi_bags,
    )
    model = _make_model()

    with pytest.raises(TypeError, match="integer class ids"):
        run_abmil_classification_batch(model, next(iter(loader)))


def test_train_abmil_classification_epoch_rejects_non_positive_grad_clip() -> None:
    loader = _make_loader(batch_size=2)
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="grad_clip_norm"):
        train_abmil_classification_epoch(
            model,
            loader,
            optimizer,
            grad_clip_norm=0.0,
        )


def test_run_abmil_classification_batch_rejects_wrong_batch_type() -> None:
    model = _make_model()

    with pytest.raises(TypeError, match="PaddedWSIBatch"):
        run_abmil_classification_batch(model, object())  # type: ignore[arg-type]
