import pytest
import torch

from src.data.wsi import InMemoryWSIFeatureStore, WSIBag
from src.models.wsi import (
    ABMILClassifier,
    ABMILImportanceProvider,
    PrecomputedImportanceProvider,
    TridentSlideEncoderImportanceProvider,
    WSIImportanceProvider,
)


def _bag(slide_id: str = "slide_001", n_tiles: int = 5, feature_dim: int = 8) -> WSIBag:
    return WSIBag(slide_id=slide_id, tile_features=torch.randn(n_tiles, feature_dim))


def test_precomputed_importance_provider_returns_target_and_metadata() -> None:
    target_store = InMemoryWSIFeatureStore(
        [
            WSIBag(
                slide_id="slide_001",
                tile_features=torch.zeros(5, 1),
                attention=torch.rand(5) + 0.1,
                metadata={"target_source": "gigapath_wsi_fm"},
            )
        ]
    )
    provider = PrecomputedImportanceProvider(target_store)

    importance, metadata = provider.compute_tile_importance(_bag())

    assert importance.shape == (5,)
    assert metadata["target_source"] == "gigapath_wsi_fm"
    assert metadata["target_type"] == "tile_importance"


def test_precomputed_importance_provider_rejects_missing_slide() -> None:
    target_store = InMemoryWSIFeatureStore()
    provider = PrecomputedImportanceProvider(target_store)

    with pytest.raises(KeyError):
        provider.compute_tile_importance(_bag())


def test_precomputed_importance_provider_rejects_missing_attention() -> None:
    target_store = InMemoryWSIFeatureStore(
        [WSIBag(slide_id="slide_001", tile_features=torch.zeros(5, 1))]
    )
    provider = PrecomputedImportanceProvider(target_store)

    with pytest.raises(ValueError, match="no precomputed importance value"):
        provider.compute_tile_importance(_bag())


def test_precomputed_importance_provider_rejects_tile_count_mismatch() -> None:
    target_store = InMemoryWSIFeatureStore(
        [
            WSIBag(
                slide_id="slide_001",
                tile_features=torch.zeros(3, 1),
                attention=torch.rand(3) + 0.1,
            )
        ]
    )
    provider = PrecomputedImportanceProvider(target_store)

    with pytest.raises(ValueError, match="tiles but bag has"):
        provider.compute_tile_importance(_bag(n_tiles=5))


def test_precomputed_importance_provider_rejects_non_feature_store() -> None:
    with pytest.raises(TypeError, match="WSIFeatureStore"):
        PrecomputedImportanceProvider(object())  # type: ignore[arg-type]


def test_abmil_importance_provider_returns_valid_distribution() -> None:
    model = ABMILClassifier(feature_dim=8, hidden_dim=16, n_classes=2)
    provider = ABMILImportanceProvider(model)

    importance, metadata = provider.compute_tile_importance(_bag())

    assert importance.shape == (5,)
    assert torch.isfinite(importance).all()
    assert (importance >= 0).all()
    assert torch.isclose(importance.sum(), torch.tensor(1.0), atol=1e-4)
    assert metadata["target_source"] == "abmil"


def test_abmil_importance_provider_restores_training_mode() -> None:
    model = ABMILClassifier(feature_dim=8, hidden_dim=16, n_classes=2)
    model.train()
    provider = ABMILImportanceProvider(model)

    provider.compute_tile_importance(_bag())

    assert model.training is True


def test_abmil_importance_provider_rejects_non_abmil_model() -> None:
    with pytest.raises(TypeError, match="ABMILClassifier"):
        ABMILImportanceProvider(object())  # type: ignore[arg-type]


def test_trident_slide_encoder_importance_provider_is_a_documented_stub() -> None:
    provider = TridentSlideEncoderImportanceProvider("gigapath_slide_encoder")

    assert isinstance(provider, WSIImportanceProvider)
    assert provider.name == "trident_slide_encoder"

    with pytest.raises((ImportError, NotImplementedError)):
        provider.compute_tile_importance(_bag())
