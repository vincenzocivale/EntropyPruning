"""Phase-A ranking metrics: how well forecaster scores reproduce the teacher attention ranking.

All functions take ``pred`` (B, N) forecaster scores and ``target`` (B, N) raw (unnormalized,
non-negative) teacher attention, and return a (B,) tensor (mean over the batch for a scalar summary).
"""

import torch


def spearman_rho(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r_pred = pred.argsort(dim=-1).argsort(dim=-1).float()
    r_true = target.argsort(dim=-1).argsort(dim=-1).float()
    r_pred = r_pred - r_pred.mean(dim=-1, keepdim=True)
    r_true = r_true - r_true.mean(dim=-1, keepdim=True)
    num = (r_pred * r_true).sum(dim=-1)
    den = torch.sqrt((r_pred ** 2).sum(dim=-1) * (r_true ** 2).sum(dim=-1))
    return num / (den + 1e-8)


def topk_overlap(pred: torch.Tensor, target: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Fraction of the teacher's top-k patches also selected in the student's top-k.

    This is the quantity Phase B actually consumes (the deployed top-k overlap), and the
    pre-registered tie-break metric (overlap@0.2) in docs/experiments.md.
    """
    B, N = pred.shape
    k = max(1, round(keep_ratio * N))
    pred_idx = pred.topk(k, dim=-1).indices
    tgt_idx = target.topk(k, dim=-1).indices
    pred_mask = torch.zeros_like(pred, dtype=torch.bool).scatter_(-1, pred_idx, True)
    tgt_mask = torch.zeros_like(target, dtype=torch.bool).scatter_(-1, tgt_idx, True)
    return (pred_mask & tgt_mask).sum(dim=-1).float() / k
