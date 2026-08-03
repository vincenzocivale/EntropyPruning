"""Padded batch utilities for WSI-level bags."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.data.wsi.bag import WSIBag


@dataclass(frozen=True)
class PaddedWSIBatch:
    """Padded tensor representation of a variable-length WSI bag batch.

    Attributes:
        slide_ids: Slide identifiers, length ``B``.
        tile_features: Padded tile features, shape ``[B, max_tiles, D]``.
        mask: Boolean valid-tile mask, shape ``[B, max_tiles]``.
            ``True`` means valid tile and ``False`` means padding.
        coords: Optional padded coordinates, shape ``[B, max_tiles, C]``.
        labels: Slide-level labels kept as a tuple because task encodings may
            differ across classification, regression, and survival.
        attention: Optional padded MIL attention targets, shape ``[B, max_tiles]``.
        metadata: Optional per-slide metadata.
    """

    slide_ids: tuple[str, ...]
    tile_features: torch.Tensor
    mask: torch.Tensor
    coords: torch.Tensor | None = None
    labels: tuple[int | float | torch.Tensor | None, ...] = ()
    attention: torch.Tensor | None = None
    metadata: tuple[dict[str, Any] | None, ...] = ()

    def __post_init__(self) -> None:
        if self.tile_features.ndim != 3:
            raise ValueError(
                "tile_features must have shape [batch, max_tiles, feature_dim]; "
                f"got {tuple(self.tile_features.shape)}."
            )
        if self.mask.ndim != 2:
            raise ValueError(
                "mask must have shape [batch, max_tiles]; "
                f"got {tuple(self.mask.shape)}."
            )
        if self.mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor.")

        batch_size, max_tiles, _ = self.tile_features.shape

        if len(self.slide_ids) != batch_size:
            raise ValueError(
                "slide_ids length must match batch size; "
                f"got {len(self.slide_ids)} and {batch_size}."
            )
        if self.mask.shape != (batch_size, max_tiles):
            raise ValueError(
                "mask shape must match tile_features first two dimensions; "
                f"got {tuple(self.mask.shape)} and expected {(batch_size, max_tiles)}."
            )
        if not self.mask.any(dim=1).all():
            raise ValueError("each batch item must contain at least one valid tile.")

        if self.coords is not None:
            if self.coords.ndim != 3:
                raise ValueError(
                    "coords must have shape [batch, max_tiles, coord_dim]; "
                    f"got {tuple(self.coords.shape)}."
                )
            if self.coords.shape[:2] != (batch_size, max_tiles):
                raise ValueError(
                    "coords first two dimensions must match tile_features; "
                    f"got {tuple(self.coords.shape[:2])} and {(batch_size, max_tiles)}."
                )
            if self.coords.shape[2] not in (2, 4):
                raise ValueError(
                    "coords last dimension must be 2 or 4; "
                    f"got {self.coords.shape[2]}."
                )

        if self.attention is not None:
            if self.attention.ndim != 2:
                raise ValueError(
                    "attention must have shape [batch, max_tiles]; "
                    f"got {tuple(self.attention.shape)}."
                )
            if self.attention.shape != (batch_size, max_tiles):
                raise ValueError(
                    "attention shape must match tile_features first two dimensions; "
                    f"got {tuple(self.attention.shape)} and {(batch_size, max_tiles)}."
                )
            if not torch.is_floating_point(self.attention):
                raise TypeError("attention must be a floating-point tensor.")
            if not torch.isfinite(self.attention).all():
                raise ValueError("attention must contain only finite values.")

        if self.labels and len(self.labels) != batch_size:
            raise ValueError(
                "labels length must match batch size; "
                f"got {len(self.labels)} and {batch_size}."
            )
        if self.metadata and len(self.metadata) != batch_size:
            raise ValueError(
                "metadata length must match batch size; "
                f"got {len(self.metadata)} and {batch_size}."
            )

    @property
    def n_bags(self) -> int:
        return int(self.tile_features.shape[0])

    @property
    def max_tiles(self) -> int:
        return int(self.tile_features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.tile_features.shape[2])

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PaddedWSIBatch":
        """Return a copy with tensor fields moved to ``device``/``dtype``.

        ``dtype`` is applied only to floating-point tensors.
        """

        def move_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            target_dtype = dtype if torch.is_floating_point(tensor) else None
            return tensor.to(device=device, dtype=target_dtype)

        labels = tuple(
            label.to(device=device) if isinstance(label, torch.Tensor) else label
            for label in self.labels
        )

        return PaddedWSIBatch(
            slide_ids=self.slide_ids,
            tile_features=move_tensor(self.tile_features),  # type: ignore[arg-type]
            mask=self.mask.to(device=device),
            coords=move_tensor(self.coords),
            labels=labels,
            attention=move_tensor(self.attention),
            metadata=self.metadata,
        )


def pad_wsi_bags(
    bags: Sequence[WSIBag],
    *,
    pad_value: float = 0.0,
    attention_pad_value: float = 0.0,
    coord_pad_value: int = -1,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PaddedWSIBatch:
    """Pad a sequence of variable-length WSI bags into a tensor batch.

    If any bag has ``attention``, all bags must have it. Same rule for
    ``coords``. This avoids silent target/coordinate dropping.
    """

    if len(bags) == 0:
        raise ValueError("bags must contain at least one WSIBag.")

    for index, bag in enumerate(bags):
        if not isinstance(bag, WSIBag):
            raise TypeError(
                "pad_wsi_bags expects WSIBag instances; "
                f"item {index} has type {type(bag).__name__}."
            )

    feature_dim = bags[0].feature_dim
    feature_dtype = dtype or bags[0].tile_features.dtype
    target_device = device or bags[0].tile_features.device

    for bag in bags:
        if bag.feature_dim != feature_dim:
            raise ValueError(
                "all bags must have the same feature_dim; "
                f"got {feature_dim} and {bag.feature_dim}."
            )

    has_coords = [bag.coords is not None for bag in bags]
    has_attention = [bag.attention is not None for bag in bags]

    if any(has_coords) and not all(has_coords):
        raise ValueError("either all bags must have coords or none of them.")
    if any(has_attention) and not all(has_attention):
        raise ValueError("either all bags must have attention or none of them.")

    coord_dim = None
    if all(has_coords):
        coord_dim = int(bags[0].coords.shape[1])  # type: ignore[union-attr]
        for bag in bags:
            assert bag.coords is not None
            if bag.coords.shape[1] != coord_dim:
                raise ValueError(
                    "all coords tensors must have the same coordinate dimension; "
                    f"got {coord_dim} and {bag.coords.shape[1]}."
                )

    batch_size = len(bags)
    max_tiles = max(bag.n_tiles for bag in bags)

    tile_features = torch.full(
        (batch_size, max_tiles, feature_dim),
        fill_value=pad_value,
        dtype=feature_dtype,
        device=target_device,
    )
    mask = torch.zeros(
        (batch_size, max_tiles),
        dtype=torch.bool,
        device=target_device,
    )

    coords = None
    if all(has_coords):
        assert coord_dim is not None
        coord_dtype = bags[0].coords.dtype  # type: ignore[union-attr]
        coords = torch.full(
            (batch_size, max_tiles, coord_dim),
            fill_value=coord_pad_value,
            dtype=coord_dtype,
            device=target_device,
        )

    attention = None
    if all(has_attention):
        attention_dtype = dtype or bags[0].attention.dtype  # type: ignore[union-attr]
        attention = torch.full(
            (batch_size, max_tiles),
            fill_value=attention_pad_value,
            dtype=attention_dtype,
            device=target_device,
        )

    for row, bag in enumerate(bags):
        n_tiles = bag.n_tiles
        tile_features[row, :n_tiles] = bag.tile_features.to(
            device=target_device,
            dtype=feature_dtype,
        )
        mask[row, :n_tiles] = True

        if coords is not None:
            assert bag.coords is not None
            coords[row, :n_tiles] = bag.coords.to(device=target_device)

        if attention is not None:
            assert bag.attention is not None
            attention[row, :n_tiles] = bag.attention.to(
                device=target_device,
                dtype=attention.dtype,
            )

    return PaddedWSIBatch(
        slide_ids=tuple(bag.slide_id for bag in bags),
        tile_features=tile_features,
        mask=mask,
        coords=coords,
        labels=tuple(bag.label for bag in bags),
        attention=attention,
        metadata=tuple(bag.metadata for bag in bags),
    )
