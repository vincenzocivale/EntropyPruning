"""Paired input/target feature-store loading for WSI bags.

Supports the case where tile-level input features (e.g. early-layer encoder
features) and the tile importance target (physically still the ``attention``
field on ``WSIBag``/``H5WSIFeatureStore``, used here as a semantic
importance target) live in two independently produced feature stores rather
than a single fused store. The legacy single-store case is fully supported by
passing the same store as both ``input_store`` and ``target_store``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data.wsi.coord_alignment import align_by_coords
from src.data.wsi.feature_store import WSIFeatureStore

_ALIGNMENT_MODES = ("index", "coords")


@dataclass(frozen=True)
class PairedWSIBag:
    """One WSI bag assembled from an input feature store and a target store.

    Attributes:
        slide_id: Stable slide identifier.
        input_features: Tensor of shape ``[n_tiles, feature_dim]`` read from
            the input feature store.
        target_importance: Tensor of shape ``[n_tiles]`` read from the
            target feature store's ``attention`` field, used as a tile
            importance target. This class validates shape only; it does not
            normalize the target.
        coords: Optional tensor of shape ``[n_tiles, 2]`` or ``[n_tiles, 4]``,
            aligned to ``input_features``/``target_importance`` order.
        label: Optional slide-level label, taken from the input store.
        metadata: Optional metadata from the input store, augmented with
            ``input_feature_source``.
        target_metadata: Optional metadata from the target store, augmented
            with ``target_source`` and ``target_type``.
    """

    slide_id: str
    input_features: torch.Tensor
    target_importance: torch.Tensor
    coords: torch.Tensor | None = None
    label: int | float | torch.Tensor | None = None
    metadata: dict[str, Any] | None = None
    target_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slide_id, str) or not self.slide_id:
            raise ValueError("slide_id must be a non-empty string.")

        if not isinstance(self.input_features, torch.Tensor):
            raise TypeError("input_features must be a torch.Tensor.")
        if self.input_features.ndim != 2:
            raise ValueError(
                "input_features must have shape [n_tiles, feature_dim]; "
                f"got {tuple(self.input_features.shape)}."
            )

        n_tiles, feature_dim = self.input_features.shape
        if n_tiles <= 0:
            raise ValueError("input_features must contain at least one tile.")
        if feature_dim <= 0:
            raise ValueError("input_features feature_dim must be positive.")

        if not torch.is_floating_point(self.input_features):
            raise TypeError("input_features must be a floating-point tensor.")
        if not torch.isfinite(self.input_features).all():
            raise ValueError("input_features must contain only finite values.")

        if not isinstance(self.target_importance, torch.Tensor):
            raise TypeError("target_importance must be a torch.Tensor.")
        if self.target_importance.ndim != 1:
            raise ValueError(
                "target_importance must have shape [n_tiles]; "
                f"got {tuple(self.target_importance.shape)}."
            )
        if self.target_importance.shape[0] != n_tiles:
            raise ValueError(
                "target_importance must have the same length as input_features; "
                f"got {self.target_importance.shape[0]} and {n_tiles}."
            )
        if not torch.is_floating_point(self.target_importance):
            raise TypeError("target_importance must be a floating-point tensor.")
        if not torch.isfinite(self.target_importance).all():
            raise ValueError("target_importance must contain only finite values.")

        if self.coords is not None:
            if not isinstance(self.coords, torch.Tensor):
                raise TypeError("coords must be a torch.Tensor when provided.")
            if self.coords.ndim != 2:
                raise ValueError(
                    "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
                    f"got {tuple(self.coords.shape)}."
                )
            if self.coords.shape[0] != n_tiles:
                raise ValueError(
                    "coords must have the same number of rows as input_features; "
                    f"got {self.coords.shape[0]} and {n_tiles}."
                )
            if self.coords.shape[1] not in (2, 4):
                raise ValueError(
                    "coords second dimension must be 2 or 4; "
                    f"got {self.coords.shape[1]}."
                )

        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict when provided.")
        if self.target_metadata is not None and not isinstance(self.target_metadata, dict):
            raise TypeError("target_metadata must be a dict when provided.")

    @property
    def n_tiles(self) -> int:
        """Number of tiles in the paired WSI bag."""

        return int(self.input_features.shape[0])

    @property
    def feature_dim(self) -> int:
        """Dimensionality of each input tile feature vector."""

        return int(self.input_features.shape[1])

    def to_wsi_bag(self) -> "Any":
        """Return an equivalent ``WSIBag`` for reuse with existing WSI code.

        ``target_importance`` is mapped onto ``WSIBag.attention`` so that
        ``pad_wsi_bags``, ``collate_padded_wsi_bags``, and the forecaster
        training loop work unchanged on paired bags.
        """

        from src.data.wsi.bag import WSIBag

        return WSIBag(
            slide_id=self.slide_id,
            tile_features=self.input_features,
            coords=self.coords,
            label=self.label,
            attention=self.target_importance,
            metadata=self.metadata,
        )


def _store_source_name(store: WSIFeatureStore) -> str:
    path = getattr(store, "path", None)
    if path is not None:
        return str(path)
    return type(store).__name__


def load_paired_wsi_bag(
    input_store: WSIFeatureStore,
    target_store: WSIFeatureStore,
    slide_id: str,
    *,
    alignment_mode: str = "index",
    require_coords: bool = False,
) -> PairedWSIBag:
    """Load one WSI bag by joining an input store and a target store.

    The legacy single-store case is supported by passing the same store
    object (or two stores backed by the same file) as ``input_store`` and
    ``target_store``.

    Args:
        input_store: Feature store providing ``tile_features`` (and,
            optionally, ``coords``/``label``) for the slide.
        target_store: Feature store providing the ``attention`` field, used
            here as the tile importance target.
        slide_id: Slide identifier, must exist in both stores.
        alignment_mode: ``"index"`` requires identical slide id, tile count,
            and (if coords are present on both sides) identical tile order.
            ``"coords"`` aligns tiles by exact coordinate match and requires
            coords on both sides.
        require_coords: If ``True``, raise unless both stores provide coords
            for this slide.

    Returns:
        A validated ``PairedWSIBag``.

    Raises:
        ValueError: On any alignment mode, coverage, duplicate, or shape
            mismatch. Mismatches are never silently resolved.
        KeyError: If ``slide_id`` is missing from either store.
    """

    if alignment_mode not in _ALIGNMENT_MODES:
        raise ValueError(
            f"alignment_mode must be one of {_ALIGNMENT_MODES}; got {alignment_mode!r}."
        )

    if not input_store.exists(slide_id):
        raise KeyError(f"slide_id not found in input feature store: {slide_id}")
    if not target_store.exists(slide_id):
        raise KeyError(f"slide_id not found in target feature store: {slide_id}")

    input_bag = input_store.read(slide_id)
    target_bag = target_store.read(slide_id)

    if target_bag.attention is None:
        raise ValueError(
            f"slide {slide_id}: target feature store has no attention/importance "
            "target."
        )

    if require_coords:
        if input_bag.coords is None:
            raise ValueError(
                f"slide {slide_id}: input feature store is missing coords but "
                "require_coords was set."
            )
        if target_bag.coords is None:
            raise ValueError(
                f"slide {slide_id}: target feature store is missing coords but "
                "require_coords was set."
            )

    if alignment_mode == "index":
        if input_bag.n_tiles != target_bag.n_tiles:
            raise ValueError(
                f"slide {slide_id}: index alignment requires equal tile counts; "
                f"got input={input_bag.n_tiles}, target={target_bag.n_tiles}."
            )

        if (
            input_bag.coords is not None
            and target_bag.coords is not None
            and not torch.equal(input_bag.coords, target_bag.coords)
        ):
            raise ValueError(
                f"slide {slide_id}: index alignment requires identical tile "
                "order, but input and target coords differ. Use "
                "alignment_mode='coords' instead."
            )

        input_features = input_bag.tile_features
        target_importance = target_bag.attention
        coords = input_bag.coords if input_bag.coords is not None else target_bag.coords

    else:  # alignment_mode == "coords"
        if input_bag.coords is None or target_bag.coords is None:
            raise ValueError(
                f"slide {slide_id}: alignment_mode='coords' requires coords in "
                "both the input and target feature stores."
            )

        input_indices, target_indices = align_by_coords(input_bag.coords, target_bag.coords)

        input_features = input_bag.tile_features[input_indices]
        target_importance = target_bag.attention[target_indices]
        coords = input_bag.coords[input_indices]

    metadata = dict(input_bag.metadata) if input_bag.metadata is not None else {}
    metadata.setdefault("input_feature_source", _store_source_name(input_store))

    target_metadata = dict(target_bag.metadata) if target_bag.metadata is not None else {}
    target_metadata.setdefault("target_source", _store_source_name(target_store))
    target_metadata.setdefault("target_type", "tile_importance")

    return PairedWSIBag(
        slide_id=slide_id,
        input_features=input_features,
        target_importance=target_importance,
        coords=coords,
        label=input_bag.label,
        metadata=metadata,
        target_metadata=target_metadata,
    )
