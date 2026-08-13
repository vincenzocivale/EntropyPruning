from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Callable, Iterator, Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


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
        openslide_cache_bytes: int = 0,
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
        self.openslide_cache_bytes = int(openslide_cache_bytes)
        if self.openslide_cache_bytes < 0:
            raise ValueError("openslide_cache_bytes must be non-negative")
        self._slide = None

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def _get_slide(self):
        if self._slide is None:
            import openslide

            self._slide = openslide.OpenSlide(str(self.wsi_path))
            if self.openslide_cache_bytes:
                self._slide.set_cache(openslide.OpenSlideCache(self.openslide_cache_bytes))
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


class MultiSlideCoordinateDataset(Dataset):
    """Concatenate WSI coordinate datasets while retaining slide/local indexes.

    Each DataLoader worker receives one copy of this object. Every child dataset opens
    its OpenSlide handle lazily, so workers can prefetch the next WSI without creating
    a fresh worker pool at every slide boundary.
    """

    def __init__(
        self,
        wsi_paths: Sequence[Path],
        coords_paths: Sequence[Path],
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        output_size: int,
        openslide_cache_bytes: int = 0,
    ) -> None:
        if len(wsi_paths) != len(coords_paths):
            raise ValueError("wsi_paths and coords_paths must have equal length")
        if not wsi_paths:
            raise ValueError("At least one slide is required")
        self.datasets = [
            OpenSlideCoordinateDataset(
                Path(wsi_path), Path(coords_path), transform, output_size=output_size,
                openslide_cache_bytes=openslide_cache_bytes,
            )
            for wsi_path, coords_path in zip(wsi_paths, coords_paths)
        ]
        self.lengths = tuple(len(dataset) for dataset in self.datasets)
        cumulative = 0
        self.offsets = []
        for length in self.lengths:
            self.offsets.append(cumulative)
            cumulative += length
        self.total = cumulative
        self.ends = tuple(
            offset + length for offset, length in zip(self.offsets, self.lengths)
        )

    def __len__(self) -> int:
        return self.total

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(index)
        slide_index = bisect.bisect_right(self.ends, index)
        local_index = index - self.offsets[slide_index]
        image, coords = self.datasets[slide_index][local_index]
        return image, coords, slide_index, local_index


class SlideSequentialBatchSampler(Sampler[list[int]]):
    """Yield full batches in slide order without crossing WSI boundaries."""

    def __init__(self, lengths: Sequence[int], batch_size: int) -> None:
        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(length <= 0 for length in self.lengths):
            raise ValueError("Every slide must contain at least one tile")

    def __iter__(self) -> Iterator[list[int]]:
        offset = 0
        for length in self.lengths:
            stop = offset + length
            for start in range(offset, stop, self.batch_size):
                yield list(range(start, min(start + self.batch_size, stop)))
            offset = stop

    def __len__(self) -> int:
        return sum(math.ceil(length / self.batch_size) for length in self.lengths)
