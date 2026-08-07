"""Read (source_embedding, target_attention) pairs from an on-disk EAF cache.

Handles both on-disk layouts produced in this repo:

  - one HDF5 file per split, datasets at the file root
    (scripts/build_wsi_tile_eaf_cache.py)
  - one HDF5 file per dataset, one group per split
    (scripts/build_thunder_online_forecaster_cache.py,
     and the legacy src/data/h5_dataset.py::H5ForecastDataset layout)

so a single training script can read caches built by either extractor.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


class CachedForecastDataset(Dataset):
    """Lazy-loading dataset over a cached ``(source, target)`` HDF5 file."""

    def __init__(
        self,
        h5_path: str | Path,
        layer_source: int,
        layer_target: int,
        split: str | None = None,
    ) -> None:
        self.h5_path = str(h5_path)
        self.layer_source = layer_source
        self.layer_target = layer_target
        self._file: h5py.File | None = None

        with h5py.File(self.h5_path, "r") as f:
            self._group_path = split if (split is not None and split in f) else None
            root = f[self._group_path] if self._group_path else f
            key = f"attn_layer{layer_target}"
            if key not in root:
                raise KeyError(
                    f"{self.h5_path}: missing '{key}' "
                    f"(available: {list(root.keys())})"
                )
            self.length = len(root[key])

    def _root(self) -> h5py.Group:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")  # re-opened per worker process
        return self._file[self._group_path] if self._group_path else self._file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        root = self._root()
        emb = torch.from_numpy(root[f"emb_layer{self.layer_source}"][idx]).float()
        target = torch.from_numpy(root[f"attn_layer{self.layer_target}"][idx]).float()
        return emb, target
