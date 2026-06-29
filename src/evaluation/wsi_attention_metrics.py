"""Ranking metrics for WSI-level tile attention forecasting."""

from __future__ import annotations

import torch


def _as_batched_2d(name: str, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if tensor.ndim == 1:
        return tensor.unsqueeze(0), True

    if tensor.ndim == 2:
        return tensor, False

    raise ValueError(
        f"{name} must have shape [n_tiles] or [batch, n_tiles]; "
        f"got {tuple(tensor.shape)}."
    )


def _validate_scores_target_mask(
    scores: torch.Tensor,
    target_attention: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    scores_batched, scores_was_unbatched = _as_batched_2d("scores", scores)
    target_batched, target_was_unbatched = _as_batched_2d(
        "target_attention", target_attention
    )

    if scores_was_unbatched != target_was_unbatched:
        raise ValueError("scores and target_attention must have the same rank.")

    if scores_batched.shape != target_batched.shape:
        raise ValueError(
            "scores and target_attention must have the same shape; "
            f"got {tuple(scores_batched.shape)} and {tuple(target_batched.shape)}."
        )

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

    batch_size, n_tiles = scores_batched.shape

    if mask is None:
        mask_batched = torch.ones(
            (batch_size, n_tiles),
            dtype=torch.bool,
            device=scores_batched.device,
        )
    else:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor when provided.")

        if mask.ndim == 1:
            mask_batched = mask.unsqueeze(0)
        elif mask.ndim == 2:
            mask_batched = mask
        else:
            raise ValueError(
                "mask must have shape [n_tiles] or [batch, n_tiles]; "
                f"got {tuple(mask.shape)}."
            )

        if mask_batched.shape != scores_batched.shape:
            raise ValueError(
                "mask must have the same shape as scores; "
                f"got {tuple(mask_batched.shape)} and {tuple(scores_batched.shape)}."
            )
        if mask_batched.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor where True means valid tile.")
        mask_batched = mask_batched.to(device=scores_batched.device)

    if not mask_batched.any(dim=1).all():
        raise ValueError("each sample must contain at least one valid tile.")

    return scores_batched, target_batched.to(scores_batched.device), mask_batched, scores_was_unbatched


def _rank_1d(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic zero-based ranks for a 1D tensor.

    Ties are broken by PyTorch's sort order. For WSI attention benchmarking this
    is acceptable because scores/attention targets are typically continuous.
    """

    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _spearman_1d(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if scores.numel() < 2:
        raise ValueError("Spearman requires at least two valid tiles per sample.")

    score_ranks = _rank_1d(scores)
    target_ranks = _rank_1d(target)

    score_centered = score_ranks - score_ranks.mean()
    target_centered = target_ranks - target_ranks.mean()

    denom = score_centered.norm() * target_centered.norm()
    if denom <= 0:
        return torch.zeros((), device=scores.device, dtype=torch.float32)

    return (score_centered * target_centered).sum() / denom


def wsi_attention_spearmanr(
    scores: torch.Tensor,
    target_attention: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean Spearman rank correlation over WSI bags.

    Returns a scalar tensor. Higher is better; perfect ranking gives ``1`` and
    perfectly reversed ranking gives ``-1``.
    """

    scores_batched, target_batched, mask_batched, _ = _validate_scores_target_mask(
        scores, target_attention, mask
    )

    values = []
    for sample_scores, sample_target, sample_mask in zip(
        scores_batched, target_batched, mask_batched, strict=True
    ):
        values.append(_spearman_1d(sample_scores[sample_mask], sample_target[sample_mask]))

    return torch.stack(values).mean()


def wsi_attention_topk_overlap(
    scores: torch.Tensor,
    target_attention: torch.Tensor,
    k: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean top-k overlap between predicted and target attention rankings.

    The metric is ``|topk(scores) ∩ topk(target)| / k`` averaged over samples.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    scores_batched, target_batched, mask_batched, _ = _validate_scores_target_mask(
        scores, target_attention, mask
    )

    values = []
    for sample_scores, sample_target, sample_mask in zip(
        scores_batched, target_batched, mask_batched, strict=True
    ):
        valid_scores = sample_scores[sample_mask]
        valid_target = sample_target[sample_mask]
        sample_k = min(k, int(valid_scores.numel()))

        pred_idx = torch.topk(valid_scores, k=sample_k).indices
        target_idx = torch.topk(valid_target, k=sample_k).indices

        pred_set = set(pred_idx.detach().cpu().tolist())
        target_set = set(target_idx.detach().cpu().tolist())
        overlap = len(pred_set.intersection(target_set)) / sample_k

        values.append(torch.tensor(overlap, device=scores_batched.device, dtype=torch.float32))

    return torch.stack(values).mean()


def wsi_attention_ndcg_at_k(
    scores: torch.Tensor,
    target_attention: torch.Tensor,
    k: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean NDCG@k using target attention as graded relevance.

    Returns a scalar in ``[0, 1]`` when target attention is non-negative and has
    positive mass on valid tiles.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    scores_batched, target_batched, mask_batched, _ = _validate_scores_target_mask(
        scores, target_attention, mask
    )

    values = []
    for sample_scores, sample_target, sample_mask in zip(
        scores_batched, target_batched, mask_batched, strict=True
    ):
        valid_scores = sample_scores[sample_mask]
        valid_target = sample_target[sample_mask]
        sample_k = min(k, int(valid_scores.numel()))

        if valid_target.sum() <= 0:
            raise ValueError("target_attention must have positive mass on valid tiles.")

        pred_order = torch.topk(valid_scores, k=sample_k).indices
        ideal_order = torch.topk(valid_target, k=sample_k).indices

        discounts = 1.0 / torch.log2(
            torch.arange(sample_k, device=valid_scores.device, dtype=torch.float32) + 2.0
        )

        dcg = (valid_target[pred_order].to(torch.float32) * discounts).sum()
        ideal_dcg = (valid_target[ideal_order].to(torch.float32) * discounts).sum()

        if ideal_dcg <= 0:
            raise ValueError("ideal DCG must be positive.")

        values.append(dcg / ideal_dcg)

    return torch.stack(values).mean()
