"""Versioned contracts for frozen EAF teacher caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

CACHE_SCHEMA_VERSION = 1


def _stable_id(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class TileCacheSpec:
    tile_encoder: str
    model_revision: str = "unknown"
    early_layer: int = 2
    input_mag: int = 20
    patch_size: int = 512
    dtype: str = "float16"
    attention_reduction: str = "cls_mean_heads"
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def cache_id(self) -> str:
        return _stable_id(asdict(self))

    def metadata(self) -> dict:
        return asdict(self) | {"cache_id": self.cache_id, "kind": "tile_eaf"}


@dataclass(frozen=True)
class WSICacheSpec:
    tile_encoder: str
    wsi_encoder: str
    tile_model_revision: str = "unknown"
    wsi_model_revision: str = "unknown"
    input_mag: int = 20
    patch_size: int = 512
    dtype: str = "float16"
    tile_score_policy: str = "teacher_default"
    store_raw_attention: bool = False
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def cache_id(self) -> str:
        return _stable_id(asdict(self))

    def metadata(self) -> dict:
        return asdict(self) | {"cache_id": self.cache_id, "kind": "wsi_eaf"}
