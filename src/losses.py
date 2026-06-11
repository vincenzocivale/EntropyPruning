"""Phase-A distillation loss for the AttentionForecaster.

The forecaster outputs per-patch logits ``s`` (B, N); the teacher is the final-block CLS->patch attention
``target`` (B, N), non-negative and not necessarily normalized. The teacher is L1-normalized to a
distribution ``a`` and the loss is KL(a || softmax(s)).
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def distillation_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """KL divergence between the L1-normalized teacher attention and the forecaster's softmax.

    Args:
        logits: (B, N) forecaster scores (raw logits).
        target: (B, N) teacher importance (non-negative, unnormalized).

    Returns:
        Scalar loss with gradient flowing to ``logits``.
    """
    a = target.clamp_min(0).float()
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_p = F.log_softmax(logits, dim=-1)
    log_a = a.clamp_min(eps).log()
    term = torch.where(a > 0, a * (log_a - log_p), torch.zeros_like(a))
    return term.sum(dim=-1).mean()
