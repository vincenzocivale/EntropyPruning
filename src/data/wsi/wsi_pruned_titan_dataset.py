"""WSI student inputs from the pruned tile encoder, targets from full TITAN."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.wsi_pipeline.numpy_store import read_array


def _load_tile_bag(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(read_array(path, "tile_embeddings")).float(),
        torch.from_numpy(read_array(path, "coords")).long(),
    )


def _load_teacher_embedding(path: Path) -> torch.Tensor:
    embedding = torch.from_numpy(read_array(path, "slide_embedding")).float()
    while embedding.dim() > 1 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    return embedding


class WSIPrunedTitanDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, *, split: str) -> None:
        if "split" not in manifest.columns:
            raise ValueError("manifest is missing split; call assign_splits first")
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No slides assigned to split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        tile_features, coords = _load_tile_bag(Path(row.tile_path))
        teacher_embedding = _load_teacher_embedding(Path(row.teacher_wsi_path))
        if len(tile_features) != len(coords):
            raise ValueError(
                f"slide {row.slide_id}: tile/coord mismatch {len(tile_features)} != {len(coords)}"
            )
        return tile_features, coords, teacher_embedding, row.slide_id
