"""Tile-importance target providers for WSI bags.

A ``WSIImportanceProvider`` computes (or looks up) a tile-level importance
score for a WSI bag. This is the target-generation side of the
importance-forecasting pipeline: ``WSITileImportanceForecaster`` learns to
predict this value from early features, while a provider is how the target
is obtained in the first place — a precomputed store, an ABMIL teacher, or
(eventually, and only where the encoder actually exposes it) a WSI
foundation model slide encoder.

TRIDENT is never a hard dependency of this module: it is imported lazily,
only inside ``TridentSlideEncoderImportanceProvider.compute_tile_importance``,
so importing ``src.models.wsi`` never requires TRIDENT to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from src.data.wsi.bag import WSIBag
from src.data.wsi.feature_store import WSIFeatureStore
from src.models.wsi.abmil import ABMILClassifier


class WSIImportanceProvider(ABC):
    """Compute a tile-importance target for one WSI bag.

    Attributes:
        name: Short, stable identifier for the provider, suitable for use as
            a ``target_source`` metadata value.
    """

    name: str

    @abstractmethod
    def compute_tile_importance(
        self, bag: WSIBag
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``(importance, metadata)`` for ``bag``.

        Args:
            bag: WSI bag to score. Implementations decide which fields they
                need (``tile_features``, ``coords``, ``slide_id``, ...).

        Returns:
            ``importance``: non-negative tensor of shape ``[n_tiles]``,
                aligned to ``bag.tile_features`` order.
            ``metadata``: free-form dict describing the source, at minimum
                ``target_source`` and ``target_type``.
        """


class PrecomputedImportanceProvider(WSIImportanceProvider):
    """Look up a precomputed importance target from a target feature store.

    This is the default provider for targets produced outside of a live
    forward pass — a WSI foundation model tile score, an externally computed
    ABMIL run, or any other precomputed per-tile value — typically imported
    via ``scripts/import_wsi_importance_targets.py``.
    """

    name = "precomputed"

    def __init__(self, target_store: WSIFeatureStore) -> None:
        if not isinstance(target_store, WSIFeatureStore):
            raise TypeError(
                "target_store must implement WSIFeatureStore; "
                f"got {type(target_store).__name__}."
            )
        self.target_store = target_store

    def compute_tile_importance(
        self, bag: WSIBag
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.target_store.exists(bag.slide_id):
            raise KeyError(
                f"slide_id not found in precomputed target store: {bag.slide_id}"
            )

        target_bag = self.target_store.read(bag.slide_id)
        if target_bag.attention is None:
            raise ValueError(
                f"slide {bag.slide_id}: target store has no precomputed "
                "importance value."
            )
        if target_bag.n_tiles != bag.n_tiles:
            raise ValueError(
                f"slide {bag.slide_id}: precomputed target has "
                f"{target_bag.n_tiles} tiles but bag has {bag.n_tiles}; use "
                "load_paired_wsi_bag with alignment_mode='coords' instead if "
                "tile order/count is not guaranteed to match."
            )

        metadata = dict(target_bag.metadata) if target_bag.metadata is not None else {}
        metadata.setdefault("target_source", "precomputed")
        metadata.setdefault("target_type", "tile_importance")

        return target_bag.attention, metadata


class ABMILImportanceProvider(WSIImportanceProvider):
    """Compute tile importance as attention from a trained ABMIL model."""

    name = "abmil"

    def __init__(
        self,
        model: ABMILClassifier,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(model, ABMILClassifier):
            raise TypeError(
                f"model must be an ABMILClassifier; got {type(model).__name__}."
            )
        self.model = model
        self.device = device

    @torch.no_grad()
    def compute_tile_importance(
        self, bag: WSIBag
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        was_training = self.model.training
        self.model.eval()

        tile_features = bag.tile_features
        if self.device is not None:
            tile_features = tile_features.to(self.device)

        try:
            output = self.model(tile_features)
        finally:
            if was_training:
                self.model.train()

        metadata = {
            "target_source": "abmil",
            "target_type": "tile_importance",
        }
        return output.attention.detach().cpu(), metadata


class TridentSlideEncoderImportanceProvider(WSIImportanceProvider):
    """Documented stub for a TRIDENT slide-encoder-based importance provider.

    Most TRIDENT slide encoders pool tile features into a slide embedding
    without exposing a retrievable per-tile attention weight, so this class
    intentionally does not claim to support arbitrary TRIDENT slide
    encoders. It raises ``NotImplementedError`` until a specific encoder with
    a documented, extractable tile-attention mechanism is wired in.

    Use ``PrecomputedImportanceProvider`` (with a target store imported via
    ``scripts/import_wsi_importance_targets.py``) or ``ABMILImportanceProvider``
    for a working importance target today.
    """

    name = "trident_slide_encoder"

    def __init__(self, slide_encoder_name: str) -> None:
        self.slide_encoder_name = slide_encoder_name

    def compute_tile_importance(
        self, bag: WSIBag
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        try:
            import trident  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TridentSlideEncoderImportanceProvider requires the optional "
                "'trident' package to be installed."
            ) from exc

        raise NotImplementedError(
            "TridentSlideEncoderImportanceProvider is a documented stub: "
            f"tile-level attention extraction for slide encoder "
            f"{self.slide_encoder_name!r} is not implemented, because most "
            "TRIDENT slide encoders do not expose tile-level attention. Use "
            "PrecomputedImportanceProvider with a target store imported via "
            "scripts/import_wsi_importance_targets.py, or "
            "ABMILImportanceProvider trained on TRIDENT tile features."
        )
