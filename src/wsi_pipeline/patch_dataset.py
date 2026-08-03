from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OpenSlideCoordinateDataset(Dataset):
    """Read level-0 patches lazily from an OpenSlide-compatible WSI."""

    def __init__(
        self,
        wsi_path: Path,
        coords_path: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        output_size: int,
        patch_size_level0: int | None = None,
    ) -> None:
        self.wsi_path = Path(wsi_path)
        self.coords_path = Path(coords_path)
        self.transform = transform
        self.output_size = int(output_size)
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        with h5py.File(self.coords_path, "r") as handle:
            self.coords = np.asarray(handle["coords"][:], dtype=np.int64)
            attr_value = handle["coords"].attrs.get("patch_size_level0")
            if attr_value is None:
                attr_value = handle.attrs.get("patch_size_level0")
        self.patch_size_level0 = int(patch_size_level0 or attr_value or output_size)
        self._slide = None

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def _get_slide(self):
        if self._slide is None:
            import openslide

            self._slide = openslide.OpenSlide(str(self.wsi_path))
        return self._slide

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = (int(v) for v in self.coords[index, :2])
        slide = self._get_slide()
        image = slide.read_region((x, y), 0, (self.patch_size_level0, self.patch_size_level0)).convert("RGB")
        if image.size != (self.output_size, self.output_size):
            image = image.resize((self.output_size, self.output_size), Image.Resampling.BICUBIC)
        return self.transform(image), torch.tensor([x, y], dtype=torch.int64)

    def __del__(self) -> None:
        slide = getattr(self, "_slide", None)
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass
