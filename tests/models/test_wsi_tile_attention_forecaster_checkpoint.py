import pytest
import torch

from src.models.wsi import (
    WSITileAttentionForecaster,
    WSITileAttentionForecasterConfig,
    load_wsi_tile_attention_forecaster_checkpoint,
    save_wsi_tile_attention_forecaster_checkpoint,
)


def test_wsi_tile_attention_forecaster_config_builds_model() -> None:
    config = WSITileAttentionForecasterConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    model = config.build()

    assert isinstance(model, WSITileAttentionForecaster)
    assert model.feature_dim == 8
    assert model.hidden_dim == 16
    assert model.n_heads == 4
    assert model.n_layers == 1
    assert model.dropout == 0.0


def test_wsi_tile_attention_forecaster_checkpoint_roundtrip(tmp_path) -> None:
    path = tmp_path / "forecaster.pt"
    config = WSITileAttentionForecasterConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    model = config.build()

    # Make parameters deterministic and non-default enough for comparison.
    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            param.fill_(0.01 * (index + 1))

    save_wsi_tile_attention_forecaster_checkpoint(
        path,
        model=model,
        config=config,
        epoch=3,
        metrics={"loss": 0.123, "spearmanr": 0.45},
        metadata={"dataset": "synthetic"},
    )

    loaded = load_wsi_tile_attention_forecaster_checkpoint(path)

    assert loaded.config == config
    assert loaded.epoch == 3
    assert loaded.metrics == {"loss": 0.123, "spearmanr": 0.45}
    assert loaded.metadata == {"dataset": "synthetic"}

    for original, restored in zip(model.parameters(), loaded.model.parameters(), strict=True):
        assert torch.equal(original, restored)


def test_wsi_tile_attention_forecaster_checkpoint_loaded_model_runs(tmp_path) -> None:
    path = tmp_path / "forecaster.pt"
    config = WSITileAttentionForecasterConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    model = config.build()

    save_wsi_tile_attention_forecaster_checkpoint(
        path,
        model=model,
        config=config,
    )

    loaded = load_wsi_tile_attention_forecaster_checkpoint(path)
    scores = loaded.model(torch.randn(5, 8))

    assert scores.shape == (5,)


def test_save_wsi_tile_attention_forecaster_checkpoint_rejects_negative_epoch(tmp_path) -> None:
    config = WSITileAttentionForecasterConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="epoch"):
        save_wsi_tile_attention_forecaster_checkpoint(
            tmp_path / "forecaster.pt",
            model=config.build(),
            config=config,
            epoch=-1,
        )


def test_save_wsi_tile_attention_forecaster_checkpoint_rejects_non_numeric_metric(tmp_path) -> None:
    config = WSITileAttentionForecasterConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    with pytest.raises(TypeError, match="numeric"):
        save_wsi_tile_attention_forecaster_checkpoint(
            tmp_path / "forecaster.pt",
            model=config.build(),
            config=config,
            metrics={"loss": "bad"},  # type: ignore[dict-item]
        )


def test_load_wsi_tile_attention_forecaster_checkpoint_rejects_bad_schema(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": 999,
            "model_type": "WSITileAttentionForecaster",
            "model_config": {"feature_dim": 8},
            "state_dict": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_wsi_tile_attention_forecaster_checkpoint(path)


def test_load_wsi_tile_attention_forecaster_checkpoint_rejects_bad_model_type(tmp_path) -> None:
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
        load_wsi_tile_attention_forecaster_checkpoint(path)
