import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model, lr_backbone: float, lr_head: float,
                    weight_decay: float) -> torch.optim.AdamW:
    """AdamW with separate lr for backbone and head."""
    param_groups = [{"params": model.head.parameters(), "lr": lr_head}]
    backbone_params = model.trainable_backbone_params
    if backbone_params:
        param_groups.insert(0, {"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def grad_norm(model: nn.Module) -> float:
    """L2 norm of all gradients. Call after unscale_(), before clip_grad_norm_."""
    total_sq = sum(
        p.grad.detach().norm() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return float(total_sq ** 0.5)


def save_results(path: Path, results: dict[str, Any]) -> Path:
    """Write results dict to JSON, adding a timestamp. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {**results, "saved_at": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(results, indent=2, default=str))
    return path
