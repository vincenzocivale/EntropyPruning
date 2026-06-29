"""Training utilities for WSI-level tile attention forecasting."""

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
from src.models.wsi import wsi_attention_kl_loss


@dataclass(frozen=True)
class WSIAttentionForecastingBatchOutput:
    """Output of one WSI attention forecasting batch pass."""

    loss: torch.Tensor
    scores: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class WSIAttentionForecastingEpochOutput:
    """Aggregated output of one WSI attention forecasting epoch."""

    loss: float
    metrics: dict[str, float]
    n_batches: int
    n_bags: int


def run_wsi_attention_forecasting_batch(
    model: nn.Module,
    batch: PaddedWSIBatch,
    *,
    top_k: int = 10,
) -> WSIAttentionForecastingBatchOutput:
    """Run model, KL loss, and ranking metrics on one padded WSI batch.

    Args:
        model: Module that accepts ``model(tile_features, mask=mask)`` and
            returns unnormalised tile scores with shape ``[B, max_tiles]``.
        batch: Padded WSI batch with attention targets.
        top_k: k used for top-k overlap and NDCG@k.

    Returns:
        Loss, raw scores, and detached ranking metrics.

    Raises:
        TypeError: If ``batch`` is not a ``PaddedWSIBatch``.
        ValueError: If attention targets are missing or if ``top_k`` is invalid.
    """

    if not isinstance(batch, PaddedWSIBatch):
        raise TypeError(
            "batch must be a PaddedWSIBatch; "
            f"got {type(batch).__name__}."
        )

    if batch.attention is None:
        raise ValueError("batch.attention is required for attention forecasting.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    scores = model(batch.tile_features, mask=batch.mask)
    loss = wsi_attention_kl_loss(scores, batch.attention, mask=batch.mask)

    with torch.no_grad():
        detached_scores = scores.detach()
        detached_attention = batch.attention.detach()
        metrics = {
            "spearmanr": wsi_attention_spearmanr(
                detached_scores,
                detached_attention,
                mask=batch.mask,
            ),
            f"top{top_k}_overlap": wsi_attention_topk_overlap(
                detached_scores,
                detached_attention,
                k=top_k,
                mask=batch.mask,
            ),
            f"ndcg_at_{top_k}": wsi_attention_ndcg_at_k(
                detached_scores,
                detached_attention,
                k=top_k,
                mask=batch.mask,
            ),
        }

    return WSIAttentionForecastingBatchOutput(
        loss=loss,
        scores=scores,
        metrics=metrics,
    )


def train_wsi_attention_forecasting_epoch(
    model: nn.Module,
    loader: Iterable[PaddedWSIBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str | None = None,
    top_k: int = 10,
    grad_clip_norm: float | None = None,
) -> WSIAttentionForecastingEpochOutput:
    """Train the WSI attention forecaster for one epoch.

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
        output = run_wsi_attention_forecasting_batch(model, batch, top_k=top_k)
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

    return WSIAttentionForecastingEpochOutput(
        loss=total_loss / n_bags,
        metrics={name: value / n_bags for name, value in metric_sums.items()},
        n_batches=n_batches,
        n_bags=n_bags,
    )


@torch.no_grad()
def evaluate_wsi_attention_forecasting_epoch(
    model: nn.Module,
    loader: Iterable[PaddedWSIBatch],
    *,
    device: torch.device | str | None = None,
    top_k: int = 10,
) -> WSIAttentionForecastingEpochOutput:
    """Evaluate the WSI attention forecaster for one epoch."""

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

        output = run_wsi_attention_forecasting_batch(model, batch, top_k=top_k)

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

    return WSIAttentionForecastingEpochOutput(
        loss=total_loss / n_bags,
        metrics={name: value / n_bags for name, value in metric_sums.items()},
        n_batches=n_batches,
        n_bags=n_bags,
    )
