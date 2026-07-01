import pytest
import torch

from src.models.wsi import (
    WSITileImportanceForecasterConfig,
    load_wsi_tile_importance_forecaster_checkpoint,
    save_wsi_tile_attention_forecaster_checkpoint,
    save_wsi_tile_importance_forecaster_checkpoint,
)


def _config() -> WSITileImportanceForecasterConfig:
    return WSITileImportanceForecasterConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )


def test_save_wsi_tile_importance_forecaster_checkpoint_populates_required_metadata(
    tmp_path,
) -> None:
    path = tmp_path / "forecaster.pt"
    config = _config()
    model = config.build()

    save_wsi_tile_importance_forecaster_checkpoint(
        path,
        model=model,
        config=config,
        loss="kl+rank",
        target_type="tile_importance",
        target_source="abmil",
        input_feature_store="data/features_layer2.h5",
        target_feature_store="data/features_wsi_importance.h5",
        alignment_mode="coords",
        epoch=3,
        metrics={"val_loss": 0.1},
        metadata={"seed": 0},
    )

    loaded = load_wsi_tile_importance_forecaster_checkpoint(path)

    assert loaded.config == config
    assert loaded.epoch == 3
    assert loaded.metrics == {"val_loss": 0.1}
    assert loaded.metadata["model_type"] == "WSITileImportanceForecaster"
    assert loaded.metadata["input_feature_dim"] == 8
    assert loaded.metadata["hidden_dim"] == 16
    assert loaded.metadata["n_heads"] == 4
    assert loaded.metadata["n_layers"] == 1
    assert loaded.metadata["loss"] == "kl+rank"
    assert loaded.metadata["target_type"] == "tile_importance"
    assert loaded.metadata["target_source"] == "abmil"
    assert loaded.metadata["input_feature_store"] == "data/features_layer2.h5"
    assert loaded.metadata["target_feature_store"] == "data/features_wsi_importance.h5"
    assert loaded.metadata["alignment_mode"] == "coords"
    assert loaded.metadata["seed"] == 0


def test_save_wsi_tile_importance_forecaster_checkpoint_loaded_model_runs(tmp_path) -> None:
    path = tmp_path / "forecaster.pt"
    config = _config()

    save_wsi_tile_importance_forecaster_checkpoint(
        path,
        model=config.build(),
        config=config,
        loss="kl",
    )

    loaded = load_wsi_tile_importance_forecaster_checkpoint(path)
    scores = loaded.model(torch.randn(5, 8))

    assert scores.shape == (5,)


def test_save_wsi_tile_importance_forecaster_checkpoint_rejects_unknown_loss(tmp_path) -> None:
    config = _config()

    with pytest.raises(ValueError, match="loss must be one of"):
        save_wsi_tile_importance_forecaster_checkpoint(
            tmp_path / "forecaster.pt",
            model=config.build(),
            config=config,
            loss="not_a_loss",
        )


def test_load_wsi_tile_importance_forecaster_checkpoint_accepts_legacy_attention_checkpoints(
    tmp_path,
) -> None:
    path = tmp_path / "legacy.pt"
    config = _config()
    model = config.build()

    save_wsi_tile_attention_forecaster_checkpoint(
        path,
        model=model,
        config=config,
        epoch=1,
        metrics={"val_loss": 0.5},
        metadata={"dataset": "synthetic"},
    )

    loaded = load_wsi_tile_importance_forecaster_checkpoint(path)

    assert loaded.config == config
    assert loaded.metadata == {"dataset": "synthetic"}
    scores = loaded.model(torch.randn(3, 8))
    assert scores.shape == (3,)


def test_load_wsi_tile_importance_forecaster_checkpoint_rejects_bad_model_type(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_type": "SomeOtherModel",
            "model_config": {"feature_dim": 8},
            "state_dict": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="model_type"):
        load_wsi_tile_importance_forecaster_checkpoint(path)
