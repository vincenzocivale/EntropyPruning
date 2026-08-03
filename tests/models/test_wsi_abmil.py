import pytest
import torch
import torch.nn.functional as F

from src.models.wsi import ABMILClassifier, ABMILOutput


def test_abmil_classifier_accepts_single_bag() -> None:
    model = ABMILClassifier(
        feature_dim=16,
        hidden_dim=32,
        n_classes=3,
        dropout=0.0,
    )

    output = model(torch.randn(7, 16))

    assert isinstance(output, ABMILOutput)
    assert output.logits.shape == (3,)
    assert output.attention.shape == (7,)
    assert output.bag_embedding.shape == (32,)
    assert torch.isclose(output.attention.sum(), torch.tensor(1.0), atol=1e-6)
    assert (output.attention >= 0).all()


def test_abmil_classifier_accepts_batched_bags() -> None:
    model = ABMILClassifier(
        feature_dim=16,
        hidden_dim=32,
        n_classes=3,
        dropout=0.0,
    )

    output = model(torch.randn(2, 7, 16))

    assert output.logits.shape == (2, 3)
    assert output.attention.shape == (2, 7)
    assert output.bag_embedding.shape == (2, 32)
    assert torch.allclose(output.attention.sum(dim=1), torch.ones(2), atol=1e-6)


def test_abmil_classifier_masks_invalid_tiles() -> None:
    model = ABMILClassifier(
        feature_dim=16,
        hidden_dim=32,
        n_classes=3,
        dropout=0.0,
    )
    tile_features = torch.randn(2, 5, 16)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
        ]
    )

    output = model(tile_features, mask=mask)

    assert output.attention.shape == (2, 5)
    assert torch.equal(output.attention[~mask], torch.zeros_like(output.attention[~mask]))
    assert torch.allclose(output.attention.sum(dim=1), torch.ones(2), atol=1e-6)


def test_abmil_classifier_supports_non_gated_attention() -> None:
    model = ABMILClassifier(
        feature_dim=16,
        hidden_dim=32,
        n_classes=3,
        dropout=0.0,
        gated=False,
    )

    output = model(torch.randn(7, 16))

    assert output.logits.shape == (3,)
    assert output.attention.shape == (7,)
    assert model.attention_u is None


def test_abmil_classifier_rejects_wrong_feature_dim() -> None:
    model = ABMILClassifier(
        feature_dim=16,
        hidden_dim=32,
        n_classes=3,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="expected feature_dim"):
        model(torch.randn(7, 15))


def test_abmil_classifier_rejects_all_masked_sample() -> None:
    model = ABMILClassifier(
        feature_dim=16,
        hidden_dim=32,
        n_classes=3,
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


def test_abmil_classifier_cross_entropy_backpropagates() -> None:
    model = ABMILClassifier(
        feature_dim=16,
        hidden_dim=32,
        n_classes=3,
        dropout=0.0,
    )
    tile_features = torch.randn(4, 7, 16)
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    output = model(tile_features)
    loss = F.cross_entropy(output.logits, labels)
    loss.backward()

    grads = [
        param.grad
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_abmil_classifier_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        ABMILClassifier(feature_dim=0)

    with pytest.raises(ValueError, match="hidden_dim"):
        ABMILClassifier(feature_dim=8, hidden_dim=0)

    with pytest.raises(ValueError, match="n_classes"):
        ABMILClassifier(feature_dim=8, n_classes=0)

    with pytest.raises(ValueError, match="dropout"):
        ABMILClassifier(feature_dim=8, dropout=1.0)
