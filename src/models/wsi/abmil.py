"""Attention-based MIL model for WSI-level bags.

This module provides a small ABMIL teacher that maps tile features to
slide-level logits and tile-level attention weights. Its attention output can
be used later as the target for ``WSITileAttentionForecaster``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _as_batched_features(tile_features: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(tile_features, torch.Tensor):
        raise TypeError("tile_features must be a torch.Tensor.")

    if tile_features.ndim == 2:
        return tile_features.unsqueeze(0), True

    if tile_features.ndim == 3:
        return tile_features, False

    raise ValueError(
        "tile_features must have shape [n_tiles, feature_dim] or "
        f"[batch, n_tiles, feature_dim]; got {tuple(tile_features.shape)}."
    )


def _validate_mask(mask: torch.Tensor | None, batch_size: int, n_tiles: int) -> torch.Tensor | None:
    if mask is None:
        return None

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor when provided.")

    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    if mask.ndim != 2:
        raise ValueError(
            "mask must have shape [n_tiles] or [batch, n_tiles]; "
            f"got {tuple(mask.shape)}."
        )

    if mask.shape != (batch_size, n_tiles):
        raise ValueError(
            "mask shape must match the first two dimensions of tile_features; "
            f"got mask {tuple(mask.shape)} and expected {(batch_size, n_tiles)}."
        )

    if mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor where True means valid tile.")

    if not mask.any(dim=1).all():
        raise ValueError("each sample must contain at least one valid tile.")

    return mask


@dataclass(frozen=True)
class ABMILOutput:
    """Output of an ABMIL forward pass."""

    logits: torch.Tensor
    attention: torch.Tensor
    bag_embedding: torch.Tensor


class ABMILClassifier(nn.Module):
    """Minimal gated attention MIL classifier.

    Args:
        feature_dim: Dimensionality of tile feature vectors.
        hidden_dim: Hidden width for tile projection and attention.
        n_classes: Number of slide-level classes.
        dropout: Dropout probability.
        gated: If true, use gated ABMIL attention. If false, use tanh attention.

    Shape:
        - Input: ``[n_tiles, feature_dim]`` or ``[batch, n_tiles, feature_dim]``.
        - Mask: optional bool tensor where ``True`` means valid tile.
        - Output logits: ``[n_classes]`` for unbatched input, otherwise
          ``[batch, n_classes]``.
        - Output attention: ``[n_tiles]`` for unbatched input, otherwise
          ``[batch, n_tiles]``.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.1,
        gated: bool = True,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if n_classes <= 0:
            raise ValueError("n_classes must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.dropout = dropout
        self.gated = gated

        self.feature_projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention_v = nn.Linear(hidden_dim, hidden_dim)
        self.attention_u = nn.Linear(hidden_dim, hidden_dim) if gated else None
        self.attention_w = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        tile_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> ABMILOutput:
        features, was_unbatched = _as_batched_features(tile_features)
        batch_size, n_tiles, feature_dim = features.shape

        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}; got {feature_dim}."
            )

        valid_mask = _validate_mask(mask, batch_size, n_tiles)

        hidden = self.feature_projector(features)

        attention_hidden = torch.tanh(self.attention_v(hidden))
        if self.gated:
            assert self.attention_u is not None
            attention_hidden = attention_hidden * torch.sigmoid(self.attention_u(hidden))

        attention_logits = self.attention_w(attention_hidden).squeeze(-1)

        if valid_mask is not None:
            attention_logits = attention_logits.masked_fill(
                ~valid_mask,
                torch.finfo(attention_logits.dtype).min,
            )

        attention = torch.softmax(attention_logits, dim=1)

        if valid_mask is not None:
            attention = attention.masked_fill(~valid_mask, 0.0)

        bag_embedding = torch.sum(hidden * attention.unsqueeze(-1), dim=1)
        logits = self.classifier(bag_embedding)

        if was_unbatched:
            return ABMILOutput(
                logits=logits.squeeze(0),
                attention=attention.squeeze(0),
                bag_embedding=bag_embedding.squeeze(0),
            )

        return ABMILOutput(
            logits=logits,
            attention=attention,
            bag_embedding=bag_embedding,
        )
