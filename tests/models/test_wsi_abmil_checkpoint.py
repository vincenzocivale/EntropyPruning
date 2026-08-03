import pytest
import torch

from src.models.wsi import (
    ABMILClassifier,
    ABMILClassifierConfig,
    load_abmil_classifier_checkpoint,
    save_abmil_classifier_checkpoint,
)


def test_abmil_classifier_config_builds_model() -> None:
    config = ABMILClassifierConfig(
        feature_dim=8,
        hidden_dim=16,
        n_classes=3,
        dropout=0.0,
        gated=True,
    )

    model = config.build()

    assert isinstance(model, ABMILClassifier)
    assert model.feature_dim == 8
    assert model.hidden_dim == 16
    assert model.n_classes == 3
    assert model.dropout == 0.0
    assert model.gated is True


def test_abmil_classifier_checkpoint_roundtrip(tmp_path) -> None:
    path = tmp_path / "abmil.pt"
    config = ABMILClassifierConfig(
        feature_dim=8,
        hidden_dim=16,
        n_classes=3,
        dropout=0.0,
        gated=True,
    )
    model = config.build()

    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            param.fill_(0.01 * (index + 1))

    save_abmil_classifier_checkpoint(
        path,
        model=model,
        config=config,
        epoch=5,
        metrics={"loss": 0.321, "accuracy": 0.75},
        metadata={"dataset": "synthetic"},
    )

    loaded = load_abmil_classifier_checkpoint(path)

    assert loaded.config == config
    assert loaded.epoch == 5
    assert loaded.metrics == {"loss": 0.321, "accuracy": 0.75}
    assert loaded.metadata == {"dataset": "synthetic"}

    for original, restored in zip(model.parameters(), loaded.model.parameters(), strict=True):
        assert torch.equal(original, restored)


def test_abmil_classifier_checkpoint_loaded_model_runs() -> None:
    config = ABMILClassifierConfig(
        feature_dim=8,
        hidden_dim=16,
        n_classes=3,
        dropout=0.0,
        gated=True,
    )
    model = config.build()

    output = model(torch.randn(5, 8))

    assert output.logits.shape == (3,)
    assert output.attention.shape == (5,)


def test_abmil_classifier_checkpoint_loaded_from_disk_runs(tmp_path) -> None:
    path = tmp_path / "abmil.pt"
    config = ABMILClassifierConfig(
        feature_dim=8,
        hidden_dim=16,
        n_classes=3,
        dropout=0.0,
        gated=False,
    )
    model = config.build()

    save_abmil_classifier_checkpoint(
        path,
        model=model,
        config=config,
    )

    loaded = load_abmil_classifier_checkpoint(path)
    output = loaded.model(torch.randn(5, 8))

    assert loaded.config.gated is False
    assert output.logits.shape == (3,)
    assert output.attention.shape == (5,)


def test_save_abmil_classifier_checkpoint_rejects_negative_epoch(tmp_path) -> None:
    config = ABMILClassifierConfig(
        feature_dim=8,
        hidden_dim=16,
        n_classes=3,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="epoch"):
        save_abmil_classifier_checkpoint(
            tmp_path / "abmil.pt",
            model=config.build(),
            config=config,
            epoch=-1,
        )


def test_save_abmil_classifier_checkpoint_rejects_non_numeric_metric(tmp_path) -> None:
    config = ABMILClassifierConfig(
        feature_dim=8,
        hidden_dim=16,
        n_classes=3,
        dropout=0.0,
    )

    with pytest.raises(TypeError, match="numeric"):
        save_abmil_classifier_checkpoint(
            tmp_path / "abmil.pt",
            model=config.build(),
            config=config,
            metrics={"loss": "bad"},  # type: ignore[dict-item]
        )


def test_load_abmil_classifier_checkpoint_rejects_bad_schema(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": 999,
            "model_type": "ABMILClassifier",
            "model_config": {"feature_dim": 8},
            "state_dict": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_abmil_classifier_checkpoint(path)


def test_load_abmil_classifier_checkpoint_rejects_bad_model_type(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_type": "OtherModel",
            "model_config": {"feature_dim": 8},
            "state_dict": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="model_type"):
        load_abmil_classifier_checkpoint(path)
