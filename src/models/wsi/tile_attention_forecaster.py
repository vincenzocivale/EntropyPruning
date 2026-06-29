"""WSI-level tile attention forecaster.

This module predicts the MIL/WSI-level attention assigned to each tile from
tile-level features, typically extracted from an early layer of a tile encoder.

It is intentionally separate from ``src.models.forecaster.AttentionForecaster``:
that module predicts patch-token attention inside one tile, while this module
predicts tile attention inside one WSI bag.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _as_batched_scores(scores: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor.")

    if scores.ndim == 1:
        return scores.unsqueeze(0), True

    if scores.ndim == 2:
        return scores, False

    raise ValueError(
        "scores must have shape [n_tiles] or [batch, n_tiles]; "
        f"got {tuple(scores.shape)}."
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
            "mask shape must match the first two dimensions of the batch; "
            f"got mask {tuple(mask.shape)} and expected {(batch_size, n_tiles)}."
        )

    if mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor where True means valid tile.")

    if not mask.any(dim=1).all():
        raise ValueError("each sample must contain at least one valid tile.")

    return mask


class WSITileAttentionForecaster(nn.Module):
    """Predict MIL attention scores for tiles in a WSI bag.

    Args:
        feature_dim: Dimensionality of each tile feature vector.
        hidden_dim: Hidden transformer width.
        n_heads: Number of self-attention heads.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout probability.

    Shape:
        - Input: ``[n_tiles, feature_dim]`` or ``[batch, n_tiles, feature_dim]``.
        - Mask: optional bool tensor ``[n_tiles]`` or ``[batch, n_tiles]`` where
          ``True`` means valid tile and ``False`` means padding.
        - Output: ``[n_tiles]`` for unbatched input, ``[batch, n_tiles]`` for
          batched input.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        tile_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one unnormalised attention score per tile.

        ``mask`` uses the EAF-facing convention ``True = valid tile``. Internally,
        PyTorch's transformer receives the inverse padding mask.
        """

        features, was_unbatched = _as_batched_features(tile_features)
        batch_size, n_tiles, feature_dim = features.shape

        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}; got {feature_dim}."
            )

        valid_mask = _validate_mask(mask, batch_size, n_tiles)

        hidden = self.input_proj(features)
        padding_mask = None if valid_mask is None else ~valid_mask
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)

        scores = self.scorer(hidden).squeeze(-1)

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)

        if was_unbatched:
            return scores.squeeze(0)

        return scores


def wsi_attention_kl_loss(
    scores: torch.Tensor,
    target_attention: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL loss between predicted scores and target MIL attention.

    The target attention is normalized inside the loss over valid tiles only.
    This keeps ``WSIBag.attention`` as a raw target container and avoids hidden
    normalization in the data layer.

    Args:
        scores: Unnormalised predicted scores, shape ``[n_tiles]`` or
            ``[batch, n_tiles]``.
        target_attention: Non-negative target attention with same shape as
            ``scores``.
        mask: Optional bool tensor where ``True`` means valid tile.

    Returns:
        Scalar KL divergence averaged over batch samples.
    """

    scores_batched, was_unbatched = _as_batched_scores(scores)

    if not isinstance(target_attention, torch.Tensor):
        raise TypeError("target_attention must be a torch.Tensor.")

    if target_attention.ndim == 1:
        target_batched = target_attention.unsqueeze(0)
    elif target_attention.ndim == 2:
        target_batched = target_attention
    else:
        raise ValueError(
            "target_attention must have shape [n_tiles] or [batch, n_tiles]; "
            f"got {tuple(target_attention.shape)}."
        )

    if target_batched.shape != scores_batched.shape:
        raise ValueError(
            "target_attention shape must match scores shape; "
            f"got {tuple(target_batched.shape)} and {tuple(scores_batched.shape)}."
        )

    batch_size, n_tiles = scores_batched.shape
    valid_mask = _validate_mask(mask, batch_size, n_tiles)

    if was_unbatched and target_attention.ndim != 1:
        raise ValueError("unbatched scores require unbatched target_attention.")

    if not torch.is_floating_point(scores_batched):
        raise TypeError("scores must be a floating-point tensor.")
    if not torch.is_floating_point(target_batched):
        raise TypeError("target_attention must be a floating-point tensor.")
    if not torch.isfinite(scores_batched).all():
        raise ValueError("scores must contain only finite values.")
    if not torch.isfinite(target_batched).all():
        raise ValueError("target_attention must contain only finite values.")
    if (target_batched < 0).any():
        raise ValueError("target_attention must be non-negative.")

    if valid_mask is None:
        valid_mask = torch.ones(
            (batch_size, n_tiles),
            dtype=torch.bool,
            device=scores_batched.device,
        )
    else:
        valid_mask = valid_mask.to(device=scores_batched.device)

    target_batched = target_batched.to(device=scores_batched.device)
    valid_mask_float = valid_mask.to(dtype=target_batched.dtype)

    masked_target = target_batched * valid_mask_float
    target_mass = masked_target.sum(dim=1, keepdim=True)
    if (target_mass <= 0).any():
        raise ValueError("target_attention must have positive mass on valid tiles.")

    target_probs = masked_target / target_mass

    masked_scores = scores_batched.masked_fill(
        ~valid_mask,
        torch.finfo(scores_batched.dtype).min,
    )
    log_probs = F.log_softmax(masked_scores, dim=1)

    return F.kl_div(log_probs, target_probs, reduction="batchmean")
