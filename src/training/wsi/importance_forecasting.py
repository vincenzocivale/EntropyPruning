"""Training utilities for WSI-level tile importance forecasting.

Generalizes ``src.training.wsi.attention_forecasting`` to support
configurable loss functions (``kl``, ``mse``, ``topk_bce``, ``kl+rank``) and
paired input/target feature stores. The legacy KL-only module is left
untouched; this module is additive.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.data.wsi import PaddedWSIBatch
from src.evaluation.wsi_attention_metrics import (
    wsi_attention_ndcg_at_k,
    wsi_attention_spearmanr,
    wsi_attention_topk_overlap,
)
from src.models.wsi import wsi_tile_importance_loss


@dataclass(frozen=True)
class WSIImportanceForecastingBatchOutput:
    """Output of one WSI tile importance forecasting batch pass."""

    loss: torch.Tensor
    scores: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class WSIImportanceForecastingEpochOutput:
    """Aggregated output of one WSI tile importance forecasting epoch."""

    loss: float
    metrics: dict[str, float]
    n_batches: int
    n_bags: int


def _mean_target_entropy(
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean per-bag entropy (nats) of the target normalized over valid tiles."""

    if mask is None:
        mask_float = torch.ones_like(target)
    else:
        mask_float = mask.to(dtype=target.dtype)

    masked_target = (target * mask_float).clamp_min(0.0)
    mass = masked_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probs = masked_target / mass

    log_probs = torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))
    entropy = -(probs * log_probs).sum(dim=1)

    return entropy.mean()


def run_wsi_tile_importance_forecasting_batch(
    model: nn.Module,
    batch: PaddedWSIBatch,
    *,
    loss: str = "kl",
    top_k: int = 10,
    rank_weight: float = 0.1,
    rank_margin: float = 1.0,
    target_smoothing: float = 0.0,
) -> WSIImportanceForecastingBatchOutput:
    """Run model, configurable loss, and ranking metrics on one padded batch.

    Args:
        model: Module that accepts ``model(tile_features, mask=mask)`` and
            returns unnormalised tile scores with shape ``[B, max_tiles]``.
        batch: Padded WSI batch with importance targets in ``batch.attention``.
        loss: One of ``"kl"``, ``"mse"``, ``"topk_bce"``, ``"kl+rank"``.
        top_k: k used for top-k overlap, NDCG@k, and top-k-based losses.
        rank_weight: Weight of the rank component in the ``"kl+rank"`` loss.
        rank_margin: Margin used by the rank component in ``"kl+rank"``.
        target_smoothing: Non-negative value added to the target importance
            before computing the loss/metrics. Use a positive value to turn
            an all-zero target into a uniform distribution instead of raising
            an explicit error.

    Returns:
        Loss, raw scores, and detached ranking/diagnostic metrics.

    Raises:
        TypeError: If ``batch`` is not a ``PaddedWSIBatch``.
        ValueError: If importance targets are missing, or if ``top_k``/
            ``target_smoothing`` are invalid.
    """

    if not isinstance(batch, PaddedWSIBatch):
        raise TypeError(
            "batch must be a PaddedWSIBatch; "
            f"got {type(batch).__name__}."
        )

    if batch.attention is None:
        raise ValueError(
            "batch.attention is required for tile importance forecasting."
        )

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if target_smoothing < 0:
        raise ValueError("target_smoothing must be non-negative.")

    target = batch.attention
    if target_smoothing > 0:
        target = target + target_smoothing

    scores = model(batch.tile_features, mask=batch.mask)
    loss_value = wsi_tile_importance_loss(
        scores,
        target,
        mask=batch.mask,
        loss=loss,
        top_k=top_k,
        rank_weight=rank_weight,
        rank_margin=rank_margin,
    )

    with torch.no_grad():
        detached_scores = scores.detach()
        detached_target = target.detach()
        metrics = {
            "spearmanr": wsi_attention_spearmanr(
                detached_scores,
                detached_target,
                mask=batch.mask,
            ),
            f"top{top_k}_overlap": wsi_attention_topk_overlap(
                detached_scores,
                detached_target,
                k=top_k,
                mask=batch.mask,
            ),
            f"ndcg_at_{top_k}": wsi_attention_ndcg_at_k(
                detached_scores,
                detached_target,
                k=top_k,
                mask=batch.mask,
            ),
            "target_entropy": _mean_target_entropy(detached_target, batch.mask),
        }

    return WSIImportanceForecastingBatchOutput(
        loss=loss_value,
        scores=scores,
        metrics=metrics,
    )


def train_wsi_tile_importance_forecasting_epoch(
    model: nn.Module,
    loader: Iterable[PaddedWSIBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str | None = None,
    loss: str = "kl",
    top_k: int = 10,
    rank_weight: float = 0.1,
    rank_margin: float = 1.0,
    target_smoothing: float = 0.0,
    grad_clip_norm: float | None = None,
) -> WSIImportanceForecastingEpochOutput:
    """Train the WSI tile importance forecaster for one epoch.

    The loader is expected to yield ``PaddedWSIBatch`` objects, typically via
    ``collate_padded_wsi_bags``.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when provided.")

    model.train()

    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    n_batches = 0
    n_bags = 0

    for batch in loader:
        if not isinstance(batch, PaddedWSIBatch):
            raise TypeError(
                "loader must yield PaddedWSIBatch objects; "
                f"got {type(batch).__name__}."
            )

        batch = batch.to(device=device) if device is not None else batch

        optimizer.zero_grad(set_to_none=True)
        output = run_wsi_tile_importance_forecasting_batch(
            model,
            batch,
            loss=loss,
            top_k=top_k,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
            target_smoothing=target_smoothing,
        )
        output.loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_weight = batch.n_bags
        total_loss += float(output.loss.detach().cpu()) * batch_weight
        for name, value in output.metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value.cpu()) * batch_weight

        n_batches += 1
        n_bags += batch_weight

    if n_batches == 0 or n_bags == 0:
        raise ValueError("loader yielded no batches.")

    return WSIImportanceForecastingEpochOutput(
        loss=total_loss / n_bags,
        metrics={name: value / n_bags for name, value in metric_sums.items()},
        n_batches=n_batches,
        n_bags=n_bags,
    )


@torch.no_grad()
def evaluate_wsi_tile_importance_forecasting_epoch(
    model: nn.Module,
    loader: Iterable[PaddedWSIBatch],
    *,
    device: torch.device | str | None = None,
    loss: str = "kl",
    top_k: int = 10,
    rank_weight: float = 0.1,
    rank_margin: float = 1.0,
    target_smoothing: float = 0.0,
) -> WSIImportanceForecastingEpochOutput:
    """Evaluate the WSI tile importance forecaster for one epoch."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    n_batches = 0
    n_bags = 0

    for batch in loader:
        if not isinstance(batch, PaddedWSIBatch):
            raise TypeError(
                "loader must yield PaddedWSIBatch objects; "
                f"got {type(batch).__name__}."
            )

        batch = batch.to(device=device) if device is not None else batch

        output = run_wsi_tile_importance_forecasting_batch(
            model,
            batch,
            loss=loss,
            top_k=top_k,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
            target_smoothing=target_smoothing,
        )

        batch_weight = batch.n_bags
        total_loss += float(output.loss.cpu()) * batch_weight
        for name, value in output.metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value.cpu()) * batch_weight

        n_batches += 1
        n_bags += batch_weight

    if was_training:
        model.train()

    if n_batches == 0 or n_bags == 0:
        raise ValueError("loader yielded no batches.")

    return WSIImportanceForecastingEpochOutput(
        loss=total_loss / n_bags,
        metrics={name: value / n_bags for name, value in metric_sums.items()},
        n_batches=n_batches,
        n_bags=n_bags,
    )
