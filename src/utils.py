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
