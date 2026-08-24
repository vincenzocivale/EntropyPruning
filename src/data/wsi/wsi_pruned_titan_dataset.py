"""Dataset for LoRA+pruning-aware TITAN distillation training.

Reads, per slide: the tile encoder's final embeddings + coords (`tile_embeddings`,
Tile-EAF cache, same as `wsi_forecaster_dataset.py`) as the student's input bag,
and the frozen, unpruned TITAN slide embedding (`slide_embedding`, wsi_eaf output
file -- already computed once, offline, by `scripts/wsi_eaf_infer_wsi_fm.py`) as
the distillation target. No TITAN forward pass runs to produce the teacher signal
at training time -- it was already cached for the whole corpus.

Reuses `build_manifest`/`assign_splits` from `wsi_forecaster_dataset.py` so the
train/val/test split is identical (same seed, same method) to the forecaster
that was trained against this same corpus.

Every slide is read in full -- no tile subsampling. The whole point of EAF is
selecting which real tiles matter via the forecaster's own ranking over the
complete bag; pre-filtering tiles before that ranking runs would defeat it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_tile_bag(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as handle:
        tile_features = torch.from_numpy(handle["tile_embeddings"][...]).to(torch.float32)
        coords = torch.from_numpy(handle["coords"][...]).to(torch.long)
    return tile_features, coords


def _load_teacher_embedding(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        embedding = torch.from_numpy(handle["slide_embedding"][...]).to(torch.float32)
    while embedding.dim() > 1 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    return embedding


class WSIPrunedTitanDataset(Dataset):
    """One item = one slide's full tile bag + the frozen TITAN teacher's slide embedding."""

    def __init__(self, manifest: pd.DataFrame, *, split: str) -> None:
        if "split" not in manifest.columns:
            raise ValueError("manifest is missing a 'split' column; call assign_splits() first.")
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No slides assigned to split={split!r}.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.rows.iloc[index]
        tile_features, coords = _load_tile_bag(Path(row.tile_path))
        teacher_embedding = _load_teacher_embedding(Path(row.attention_path))
        if coords.shape[0] != tile_features.shape[0]:
            raise ValueError(
                f"slide {row.slide_id}: tile_embeddings has {tile_features.shape[0]} rows, "
                f"coords has {coords.shape[0]}"
            )
        return tile_features, coords, teacher_embedding, row.slide_id
