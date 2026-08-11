from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def find_transformer_blocks(model: nn.Module) -> Sequence[nn.Module]:
    candidates = (
        "visual.trunk.blocks",
        "visual.blocks",
        "visual.transformer.resblocks",
        "trunk.blocks",
        "blocks",
        "encoder.layer",
    )
    for dotted in candidates:
        current: Any = model
        ok = True
        for part in dotted.split("."):
            if not hasattr(current, part):
                ok = False
                break
            current = getattr(current, part)
        if ok and isinstance(current, (nn.ModuleList, nn.Sequential, list, tuple)) and len(current) >= 2:
            return current
    discovered: list[tuple[str, Sequence[nn.Module]]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) >= 2:
            discovered.append((name, module))
    if not discovered:
        raise RuntimeError("Could not discover transformer blocks; pass an explicit block path in a custom adapter")
    discovered.sort(key=lambda item: len(item[1]), reverse=True)
    return discovered[0][1]


def resolve_num_prefix_tokens(model: nn.Module, blocks: Sequence[nn.Module]) -> int:
    """Find how many prefix tokens (CLS + optional register tokens) precede patches.

    Tries, in order: (1) ``model.num_prefix_tokens`` directly; (2) the same module
    whose ``.blocks`` attribute *is* ``blocks`` (the actual trunk owning them, since
    ``model`` itself may be a thin wrapper, e.g. CONCH's ``EncoderWithAttentionalPooler``
    around ``model.trunk``); (3) a conservative fallback of ``1`` (CLS-only), which is
    correct for every model this pipeline currently targets (CONCH v1.5, UNI-family).
    """
    direct = getattr(model, "num_prefix_tokens", None)
    if isinstance(direct, int):
        return direct
    for _, module in model.named_modules():
        if getattr(module, "blocks", None) is blocks:
            owned = getattr(module, "num_prefix_tokens", None)
            if isinstance(owned, int):
                return owned
    return 1


def extract_cls_token(value: Any, batch_size: int) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Hook output is not a tensor: {type(value)!r}")
    if value.ndim == 2:
        if value.shape[0] != batch_size:
            raise ValueError(f"Expected batch dimension {batch_size}, got {tuple(value.shape)}")
        return value
    if value.ndim != 3:
        raise ValueError(f"Expected [B,T,D] or [T,B,D], got {tuple(value.shape)}")
    if value.shape[0] == batch_size:
        return value[:, 0, :]
    if value.shape[1] == batch_size:
        return value[0, :, :]
    raise ValueError(f"Could not identify batch dimension in {tuple(value.shape)} for B={batch_size}")
