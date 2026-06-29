"""Batch utilities for WSI-level tile attention forecasting."""

from __future__ import annotations

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
