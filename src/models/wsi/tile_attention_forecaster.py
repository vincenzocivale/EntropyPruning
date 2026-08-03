"""WSI-level tile importance forecaster.

This module predicts a scalar importance score for each tile in a WSI bag
from tile-level features, typically extracted from an early layer of a tile
encoder. The supervision target (``target_importance``) is tile-level
attention/importance extracted from an MIL teacher (e.g. ABMIL), a WSI
foundation model, or any other precomputed tile-importance source — this
module does not care where the target came from.

It is intentionally separate from ``src.models.forecaster.AttentionForecaster``:
that module predicts patch-token attention inside one tile, while this module
predicts tile importance inside one WSI bag.

``WSITileAttentionForecaster`` is kept as a backward-compatible alias of
``WSITileImportanceForecaster``: the model predicts an importance score, and
"attention" was the name used before this module supported non-attention
importance targets (e.g. WSI-FM tile scores).
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


class WSITileImportanceForecaster(nn.Module):
    """Predict tile-importance scores for tiles in a WSI bag.

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


# Backward-compatible alias: this class used to be the only importance
# source (ABMIL attention), hence the "Attention" name. It now predicts a
# generic tile-importance score regardless of target source.
WSITileAttentionForecaster = WSITileImportanceForecaster


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


WSI_TILE_IMPORTANCE_LOSS_TYPES = ("kl", "mse", "topk_bce", "kl+rank")


def _validate_scores_and_target(
    scores: torch.Tensor,
    target_importance: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared validation for the non-KL importance losses below.

    Returns ``(scores_batched, target_batched, valid_mask)`` with ``mask``
    resolved to an all-True tensor when not provided. ``wsi_attention_kl_loss``
    keeps its own inlined validation so its error messages/behaviour stay
    stable for existing callers; this helper only backs the newer losses.
    """

    scores_batched, _ = _as_batched_scores(scores)

    if not isinstance(target_importance, torch.Tensor):
        raise TypeError("target_importance must be a torch.Tensor.")

    if target_importance.ndim == 1:
        target_batched = target_importance.unsqueeze(0)
    elif target_importance.ndim == 2:
        target_batched = target_importance
    else:
        raise ValueError(
            "target_importance must have shape [n_tiles] or [batch, n_tiles]; "
            f"got {tuple(target_importance.shape)}."
        )

    if target_batched.shape != scores_batched.shape:
        raise ValueError(
            "target_importance shape must match scores shape; "
            f"got {tuple(target_batched.shape)} and {tuple(scores_batched.shape)}."
        )

    batch_size, n_tiles = scores_batched.shape
    valid_mask = _validate_mask(mask, batch_size, n_tiles)

    if not torch.is_floating_point(scores_batched):
        raise TypeError("scores must be a floating-point tensor.")
    if not torch.is_floating_point(target_batched):
        raise TypeError("target_importance must be a floating-point tensor.")
    if not torch.isfinite(scores_batched).all():
        raise ValueError("scores must contain only finite values.")
    if not torch.isfinite(target_batched).all():
        raise ValueError("target_importance must contain only finite values.")
    if (target_batched < 0).any():
        raise ValueError("target_importance must be non-negative.")

    if valid_mask is None:
        valid_mask = torch.ones(
            (batch_size, n_tiles),
            dtype=torch.bool,
            device=scores_batched.device,
        )

    target_batched = target_batched.to(device=scores_batched.device)
    valid_mask = valid_mask.to(device=scores_batched.device)

    return scores_batched, target_batched, valid_mask


def wsi_tile_importance_mse_loss(
    scores: torch.Tensor,
    target_importance: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss between predicted and target tile-importance distributions.

    Both ``scores`` and ``target_importance`` are normalized into probability
    distributions over valid tiles per bag (scores via softmax, target via
    sum-normalization), so this loss lives in the same probability space as
    ``wsi_attention_kl_loss`` and is a drop-in alternative divergence.
    """

    scores_batched, target_batched, valid_mask = _validate_scores_and_target(
        scores, target_importance, mask
    )
    valid_mask_float = valid_mask.to(dtype=target_batched.dtype)

    masked_target = target_batched * valid_mask_float
    target_mass = masked_target.sum(dim=1, keepdim=True)
    if (target_mass <= 0).any():
        raise ValueError("target_importance must have positive mass on valid tiles.")
    target_probs = masked_target / target_mass

    masked_scores = scores_batched.masked_fill(
        ~valid_mask,
        torch.finfo(scores_batched.dtype).min,
    )
    pred_probs = F.softmax(masked_scores, dim=1) * valid_mask_float

    n_valid = valid_mask_float.sum(dim=1)
    per_sample = ((pred_probs - target_probs) ** 2 * valid_mask_float).sum(dim=1) / n_valid

    return per_sample.mean()


def wsi_tile_importance_topk_bce_loss(
    scores: torch.Tensor,
    target_importance: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    top_k: int = 10,
) -> torch.Tensor:
    """Binary cross-entropy loss treating the target top-k tiles as positives.

    For each bag, the ``top_k`` valid tiles by target importance are labelled
    ``1`` and the rest ``0``; ``scores`` are treated as per-tile logits for
    "this tile belongs to the retained top-k set".
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    scores_batched, target_batched, valid_mask = _validate_scores_and_target(
        scores, target_importance, mask
    )

    losses = []
    for sample_scores, sample_target, sample_mask in zip(
        scores_batched, target_batched, valid_mask, strict=True
    ):
        valid_scores = sample_scores[sample_mask]
        valid_target = sample_target[sample_mask]
        n_valid = int(valid_scores.numel())

        if n_valid < 2:
            raise ValueError("topk_bce loss requires at least two valid tiles per sample.")

        k = min(top_k, n_valid)
        pos_idx = torch.topk(valid_target, k=k).indices

        binary_target = torch.zeros_like(valid_scores)
        binary_target[pos_idx] = 1.0

        losses.append(F.binary_cross_entropy_with_logits(valid_scores, binary_target))

    return torch.stack(losses).mean()


def wsi_tile_importance_rank_loss(
    scores: torch.Tensor,
    target_importance: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    top_k: int = 10,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise margin ranking loss between top-k and bottom-k target tiles.

    For each bag, the ``top_k`` valid tiles by target importance should be
    scored higher than the bottom-``k`` valid tiles by at least ``margin``.
    This is a bounded-cost (``O(k^2)`` per bag) ranking surrogate, used as the
    "rank" component of the ``kl+rank`` combined loss.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if margin <= 0:
        raise ValueError("margin must be positive.")

    scores_batched, target_batched, valid_mask = _validate_scores_and_target(
        scores, target_importance, mask
    )

    losses = []
    for sample_scores, sample_target, sample_mask in zip(
        scores_batched, target_batched, valid_mask, strict=True
    ):
        valid_scores = sample_scores[sample_mask]
        valid_target = sample_target[sample_mask]
        n_valid = int(valid_scores.numel())

        if n_valid < 2:
            raise ValueError("rank loss requires at least two valid tiles per sample.")

        k = max(1, min(top_k, n_valid // 2))
        pos_idx = torch.topk(valid_target, k=k, largest=True).indices
        neg_idx = torch.topk(valid_target, k=k, largest=False).indices

        pos_scores = valid_scores[pos_idx]
        neg_scores = valid_scores[neg_idx]

        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        losses.append(F.relu(margin - diff).mean())

    return torch.stack(losses).mean()


def wsi_tile_importance_loss(
    scores: torch.Tensor,
    target_importance: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    loss: str = "kl",
    top_k: int = 10,
    rank_weight: float = 0.1,
    rank_margin: float = 1.0,
) -> torch.Tensor:
    """Dispatch to one of the supported tile-importance loss functions.

    Args:
        loss: One of ``"kl"``, ``"mse"``, ``"topk_bce"``, or ``"kl+rank"``.
            ``"kl+rank"`` adds ``rank_weight * wsi_tile_importance_rank_loss``
            to the KL loss.
    """

    if loss not in WSI_TILE_IMPORTANCE_LOSS_TYPES:
        raise ValueError(
            f"loss must be one of {WSI_TILE_IMPORTANCE_LOSS_TYPES}; got {loss!r}."
        )

    if loss == "kl":
        return wsi_attention_kl_loss(scores, target_importance, mask=mask)

    if loss == "mse":
        return wsi_tile_importance_mse_loss(scores, target_importance, mask=mask)

    if loss == "topk_bce":
        return wsi_tile_importance_topk_bce_loss(
            scores, target_importance, mask=mask, top_k=top_k
        )

    kl = wsi_attention_kl_loss(scores, target_importance, mask=mask)
    rank = wsi_tile_importance_rank_loss(
        scores, target_importance, mask=mask, top_k=top_k, margin=rank_margin
    )
    return kl + rank_weight * rank
