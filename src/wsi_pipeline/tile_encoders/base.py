from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class TileEncoderOutput:
    embeddings: Mapping[str, torch.Tensor]


class TileEncoderAdapter(ABC):
    name: str
    input_size: int

    @abstractmethod
    def to(self, device: torch.device) -> "TileEncoderAdapter": ...

    @abstractmethod
    def eval(self) -> "TileEncoderAdapter": ...

    @abstractmethod
    def encode(self, images: torch.Tensor) -> TileEncoderOutput: ...

    @property
    @abstractmethod
    def transform(self): ...
