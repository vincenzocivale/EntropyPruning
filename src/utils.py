import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(
    model,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer with differential learning rates.

    Backbone params (from ``model.trainable_backbone_params``) receive
    ``lr_backbone``; head params receive ``lr_head``.  For strategies that
    freeze the backbone entirely (e.g. ``LinearProbingClassifier``),
    ``trainable_backbone_params`` is empty and only the head group is created.
    """
    param_groups = [{"params": model.head.parameters(), "lr": lr_head}]
    backbone_params = model.trainable_backbone_params
    if backbone_params:
        param_groups.insert(0, {"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of all gradient tensors in the model.

    Call after ``scaler.unscale_(optimizer)`` (AMP) or after
    ``loss.backward()`` (full precision) and before ``clip_grad_norm_``.
    Returns 0.0 if no parameter has a gradient.
    """
    total_sq = sum(
        p.grad.detach().norm() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return float(total_sq ** 0.5)


def save_results(path: Path, results: dict[str, Any]) -> Path:
    """Write a results dictionary to the given JSON file path.

    Adds an ISO-format timestamp under ``"saved_at"``.  Parent directories are
    created automatically.  Existing file is overwritten on re-runs.

    Args:
        path:    Full path to the output JSON file (e.g. ``output_dir / "results.json"``).
        results: Serialisable dictionary of metrics and configuration.

    Returns:
        The resolved path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {**results, "saved_at": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(results, indent=2, default=str))
    return path


class EarlyStopping:
    """Stop training if validation metric does not improve for `patience` epochs.

    Tracks the best metric value and increments a counter each epoch the metric
    does not improve by at least `min_delta`. When counter reaches `patience`,
    returns True, indicating training should stop.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.wait_count = 0

    def step(self, current: float) -> bool:
        """
        Update early stopping state.

        Args:
            current: Current value of the metric (e.g., validation loss).

        Returns:
            True if training should stop, False otherwise.
        """
        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.wait_count = 0
            return False
        else:
            self.wait_count += 1
            return self.wait_count >= self.patience

    @property
    def best(self) -> float:
        """Best metric value seen so far."""
        return self.best_value


class PlateauStopper:
    """Detect plateau on a noisy, higher-is-better metric (e.g. Spearman ρ, cosine sim).

    Designed for intra-epoch use: called every few hundred training steps with a
    quick validation metric. Smooths the signal with an EMA so isolated noisy
    drops don't trigger false stops.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-3,
        ema_alpha: float = 0.3,
        higher_is_better: bool = True,
        warmup: int = 2,
    ):
        """
        Args:
            patience: Number of consecutive checks with no EMA improvement before stopping.
            min_delta: Minimum EMA change considered an improvement.
            ema_alpha: EMA smoothing factor in (0, 1]. Higher = less smoothing.
            higher_is_better: True for ρ / cosine sim; False for losses.
            warmup: Number of initial calls used only to seed the EMA (no stop possible).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.ema_alpha = ema_alpha
        self.higher_is_better = higher_is_better
        self.warmup = warmup

        self.ema: float | None = None
        self.best_ema: float = -float("inf") if higher_is_better else float("inf")
        self.wait_count = 0
        self.calls = 0

    def step(self, current: float) -> bool:
        """Update with a new metric reading. Returns True if training should stop."""
        self.calls += 1
        self.ema = current if self.ema is None else self.ema_alpha * current + (1 - self.ema_alpha) * self.ema

        if self.calls <= self.warmup:
            self.best_ema = self.ema
            return False

        improved = (
            self.ema > self.best_ema + self.min_delta
            if self.higher_is_better
            else self.ema < self.best_ema - self.min_delta
        )
        if improved:
            self.best_ema = self.ema
            self.wait_count = 0
            return False
        self.wait_count += 1
        return self.wait_count >= self.patience

    @property
    def smoothed(self) -> float | None:
        """Current EMA value (None until first step)."""
        return self.ema
