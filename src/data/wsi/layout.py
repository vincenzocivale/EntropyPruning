"""Canonical filesystem layout for EAF WSI data and artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DatasetRole(str, Enum):
    PRETRAINING = "pretraining"
    DOWNSTREAM = "downstream"


@dataclass(frozen=True)
class StoreLayout:
    """Resolve paths below the single ``$EAF_WSI_ROOT`` runtime store."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "StoreLayout":
        if root is None:
            value = os.environ.get("EAF_WSI_ROOT")
            if not value:
                raise ValueError("Pass data_root or set EAF_WSI_ROOT")
            root = value
        return cls(Path(root).expanduser().resolve())

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def caches(self) -> Path:
        return self.root / "caches"

    @property
    def archives(self) -> Path:
        return self.root / "archives"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def dataset_dir(self, role: DatasetRole | str, name: str) -> Path:
        role = DatasetRole(role)
        return self.datasets / role.value / name

    def manifest_path(self, role: DatasetRole | str, name: str) -> Path:
        return self.dataset_dir(role, name) / "manifests" / "slides.csv"

    def tile_cache_dir(self, dataset: str, encoder: str, cache_id: str) -> Path:
        return self.caches / "tile_eaf" / dataset / encoder / cache_id

    def wsi_cache_dir(
        self,
        dataset: str,
        tile_encoder: str,
        wsi_encoder: str,
        cache_id: str,
    ) -> Path:
        pair = f"{tile_encoder}__{wsi_encoder}"
        return self.caches / "wsi_eaf" / dataset / pair / cache_id

    def ensure_base_dirs(self) -> None:
        for path in (
            self.sources,
            self.datasets / DatasetRole.PRETRAINING.value,
            self.datasets / DatasetRole.DOWNSTREAM.value,
            self.caches,
            self.archives,
            self.checkpoints,
            self.results,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
