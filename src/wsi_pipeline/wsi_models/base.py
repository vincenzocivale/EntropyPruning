from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class WSIModelOutput:
    slide_embedding: torch.Tensor
    attention: Mapping[str, torch.Tensor] = field(default_factory=dict)
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WSIModelAdapter(ABC):
    name: str
    required_feature_key: str
    required_feature_dim: int | None
    native_attention: bool

    @abstractmethod
    def to(self, device: torch.device) -> "WSIModelAdapter": ...

    @abstractmethod
    def eval(self) -> "WSIModelAdapter": ...

    @abstractmethod
    def encode(self, features: torch.Tensor, coords: torch.Tensor, patch_size_level0: int) -> WSIModelOutput: ...
