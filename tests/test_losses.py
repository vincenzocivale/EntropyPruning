"""Behavioral tests for the Phase-A KL distillation loss (CPU, synthetic tensors)."""

import torch
import torch.nn.functional as F

from src.losses import distillation_loss


def _inputs(B=4, N=16, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, N, requires_grad=True)
    target = torch.rand(B, N)  # non-negative, unnormalized
    return logits, target


def test_finite_and_differentiable():
    logits, target = _inputs()
    loss = distillation_loss(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    (grad,) = torch.autograd.grad(loss, logits)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_kl_matches_reference():
    logits, target = _inputs()
    a = target / target.sum(-1, keepdim=True)
    ref = F.kl_div(logits.log_softmax(-1), a, reduction="batchmean")
    mine = distillation_loss(logits, target)
    assert torch.allclose(mine, ref, atol=1e-6)


def test_nan_safe_with_zeros():
    logits, target = _inputs()
    target = target.clone()
    target[:, :4] = 0.0  # exact zeros in the teacher
    loss = distillation_loss(logits, target)
    assert torch.isfinite(loss)
    (grad,) = torch.autograd.grad(loss, logits)
    assert torch.isfinite(grad).all()
