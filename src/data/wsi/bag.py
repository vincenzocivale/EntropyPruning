"""Data contracts for WSI-level bags.

A WSI bag represents one slide as a variable-length set of tile-level
features. It is intentionally independent from Thunder, Trident,
Patho-Bench, MIL models, and any specific storage backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class WSIBag:
    """Container for one WSI represented as tile features.

    Attributes:
        slide_id: Stable slide identifier.
        tile_features: Tensor of shape ``[n_tiles, feature_dim]``.
        coords: Optional tensor of shape ``[n_tiles, 2]`` or ``[n_tiles, 4]``.
            Common conventions are ``(x, y)`` or ``(x, y, width, height)``.
        label: Optional slide-level label. Kept deliberately generic because
            classification, regression, and survival tasks encode labels
            differently.
        attention: Optional MIL attention target of shape ``[n_tiles]``.
            This class validates shape only; it does not normalize attention.
        metadata: Optional free-form metadata. Values should remain lightweight.
    """

    slide_id: str
    tile_features: torch.Tensor
    coords: torch.Tensor | None = None
    label: int | float | torch.Tensor | None = None
    attention: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slide_id, str) or not self.slide_id:
            raise ValueError("slide_id must be a non-empty string.")

        if not isinstance(self.tile_features, torch.Tensor):
            raise TypeError("tile_features must be a torch.Tensor.")

        if self.tile_features.ndim != 2:
            raise ValueError(
                "tile_features must have shape [n_tiles, feature_dim]; "
                f"got shape {tuple(self.tile_features.shape)}."
            )

        n_tiles, feature_dim = self.tile_features.shape
        if n_tiles <= 0:
            raise ValueError("tile_features must contain at least one tile.")
        if feature_dim <= 0:
            raise ValueError("tile_features feature_dim must be positive.")

        if not torch.is_floating_point(self.tile_features):
            raise TypeError("tile_features must be a floating-point tensor.")

        if self.coords is not None:
            if not isinstance(self.coords, torch.Tensor):
                raise TypeError("coords must be a torch.Tensor when provided.")
            if self.coords.ndim != 2:
                raise ValueError(
                    "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
                    f"got shape {tuple(self.coords.shape)}."
                )
            if self.coords.shape[0] != n_tiles:
                raise ValueError(
                    "coords must have the same number of rows as tile_features; "
                    f"got {self.coords.shape[0]} and {n_tiles}."
                )
            if self.coords.shape[1] not in (2, 4):
                raise ValueError(
                    "coords second dimension must be 2 or 4; "
                    f"got {self.coords.shape[1]}."
                )

        if self.attention is not None:
            if not isinstance(self.attention, torch.Tensor):
                raise TypeError("attention must be a torch.Tensor when provided.")
            if self.attention.ndim != 1:
                raise ValueError(
                    "attention must have shape [n_tiles]; "
                    f"got shape {tuple(self.attention.shape)}."
                )
            if self.attention.shape[0] != n_tiles:
                raise ValueError(
                    "attention must have the same length as tile_features; "
                    f"got {self.attention.shape[0]} and {n_tiles}."
                )
            if not torch.is_floating_point(self.attention):
                raise TypeError("attention must be a floating-point tensor.")
            if not torch.isfinite(self.attention).all():
                raise ValueError("attention must contain only finite values.")

        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict when provided.")

    @property
    def n_tiles(self) -> int:
        """Number of tiles in the WSI bag."""

        return int(self.tile_features.shape[0])

    @property
    def feature_dim(self) -> int:
        """Dimensionality of each tile feature vector."""

        return int(self.tile_features.shape[1])

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "WSIBag":
        """Return a copy with tensor fields moved to ``device``/``dtype``.

        ``dtype`` is applied only to floating-point tensors. Coordinates keep
        their original dtype unless they are floating-point.
        """

        def move_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            target_dtype = dtype if torch.is_floating_point(tensor) else None
            return tensor.to(device=device, dtype=target_dtype)

        label = self.label
        if isinstance(label, torch.Tensor):
            label = label.to(device=device)

        return WSIBag(
            slide_id=self.slide_id,
            tile_features=move_tensor(self.tile_features),  # type: ignore[arg-type]
            coords=move_tensor(self.coords),
            label=label,
            attention=move_tensor(self.attention),
            metadata=self.metadata,
        )
