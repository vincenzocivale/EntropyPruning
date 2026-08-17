"""Training-time pairing of the compact Tile-EAF cache with raw tile pixels.

The permanent cache (``TileCacheWriter``/``eaf.py cache tile``) stores only ``coords``,
``final_attention`` and ``tile_embeddings`` -- ``early_tokens`` is deliberately not
persisted (see ``CACHE_SCHEMA_VERSION`` v2 in ``cache_contracts.py`` and
``docs/offline_eaf_pipeline.md`` for the storage-cost rationale). EAF Tile training needs
``early_tokens`` as its input, so it must be recomputed ONLINE, once per training step,
from raw tile pixels re-read from the source WSI:

    compact cache (coords, final_attention)  ---+
                                                  |--> (image, target) pairs
    raw WSI + TRIDENT coords (pixels)        ---+
              |
              v
    HookedViTTileTeacherAdapter.extract_early(images, early_layer)   <- run every step,
              |                                                          NOT cached
              v
    AttentionForecaster(early_tokens) -> loss vs. cached target -> backward

This keeps the expensive, full-forward quantities (``final_attention``,
``tile_embeddings``) computed exactly once and reused forever, while the cheap,
early-exit partial forward for ``early_tokens`` is the only thing that ever re-touches
the encoder during training -- never the full 24-block CONCH forward.

Caveat this creates for the cold-archive/raw-release policy (``docs/offline_eaf_pipeline.md``
"Cold archive" section): EAF Tile training is no longer pixel-free once raw WSI access is
released. It needs *some* source of tile pixels at every step -- today that is the
original WSI via ``OpenSlideCoordinateDataset``; the cold-archive JPEGs are a documented
future alternative pixel source (not yet wired here) for slides whose raw has been
released. Do not release raw for a corpus this dataset is actively training against
without first pointing training at the archive instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache_io import open_h5_with_retry
from .patch_dataset import OpenSlideCoordinateDataset
from src.data.wsi_tile_stream import SlideRecord, WSIBalancedBatchSampler, WSITileDataset


class CompactTileTargetDataset(Dataset):
    """Yields ``(image, target)`` for one slide from a compact Tile-EAF cache.

    ``image`` is the raw tile pixel crop, transformed exactly as the tile encoder
    expects (same ``transform``/``output_size`` used to build the cache). It does NOT
    include ``early_tokens`` -- callers run
    ``HookedViTTileTeacherAdapter.extract_early(images_batch, early_layer=...)`` on the
    collated batch to get that online. ``target`` is read straight from the cache's
    ``target_key`` array -- ``"final_attention"`` (the EAF forecaster's KL target,
    default) or ``"tile_embeddings"`` (the frozen unpruned model's final per-tile
    embedding, used as the pruned-encoder distillation target so that path never has
    to re-run a full unpruned forward pass online).

    Row order is verified, not assumed: the compact cache's ``coords`` must exactly
    match the TRIDENT coords file's ``coords`` (both are written/read in the same
    on-disk row order by construction, but a stale/rebuilt-out-of-band cache could
    disagree -- this fails loudly at construction time rather than silently pairing
    the wrong target with the wrong tile).
    """

    def __init__(
        self,
        wsi_path: str | Path,
        coords_path: str | Path,
        compact_cache_path: str | Path,
        transform: Any,
        *,
        output_size: int,
        target_key: str = "final_attention",
    ) -> None:
        self._pixels = OpenSlideCoordinateDataset(
            Path(wsi_path), Path(coords_path), transform, output_size=output_size
        )
        with open_h5_with_retry(Path(compact_cache_path), "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Cache is not complete: {compact_cache_path}")
            self.target_key = target_key
            self.targets = np.asarray(handle[target_key][:], dtype=np.float32)
            cached_coords = np.asarray(handle["coords"][:])

        if len(self._pixels) != len(self.targets):
            raise RuntimeError(
                f"{coords_path}: {len(self._pixels)} coords vs "
                f"{compact_cache_path}: {len(self.targets)} cached {target_key!r} targets"
            )
        if not np.array_equal(self._pixels.coords[:, :2], cached_coords[:, :2]):
            raise RuntimeError(
                f"Coord order mismatch between {coords_path} and {compact_cache_path}; "
                "refusing to silently pair mismatched tiles/targets"
            )

    def __len__(self) -> int:
        return len(self._pixels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, _coord = self._pixels[index]
        target = torch.from_numpy(self.targets[index])
        return image, target


class CompactCachedWSITileDataset(WSITileDataset):
    """Multi-WSI sampling dataset backed by worker-local compact-cache handles.

    It accepts the same ``(slide_index, coord_index)`` keys as
    ``WSIBalancedBatchSampler`` and therefore preserves the existing case/cohort
    sampling policy while replacing the expensive final teacher pass with one HDF5
    row lookup.
    """

    def __init__(
        self,
        records: list[SlideRecord],
        cache_paths: dict[str, Path],
        transform: Any,
        *,
        augment: bool,
        slide_cache_size: int = 4,
        target_cache_size: int = 8,
        openslide_cache_bytes: int = 0,
        resize_to: int | None = None,
        target_key: str = "final_attention",
    ) -> None:
        super().__init__(
            records,
            transform,
            augment=augment,
            slide_cache_size=slide_cache_size,
            openslide_cache_bytes=openslide_cache_bytes,
            resize_to=resize_to,
        )
        missing = [record.slide_id for record in records if record.slide_id not in cache_paths]
        if missing:
            raise RuntimeError(f"Missing compact caches for {len(missing)} slides: {missing[:5]}")
        self.cache_paths = {key: Path(value) for key, value in cache_paths.items()}
        self.target_cache_size = max(1, int(target_cache_size))
        self.target_key = target_key
        self._target_handles: OrderedDict[str, h5py.File] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_target_handles"] = None
        return state

    def _target(self, record: SlideRecord, coord_index: int) -> torch.Tensor:
        if self._target_handles is None:
            self._target_handles = OrderedDict()
        key = str(self.cache_paths[record.slide_id])
        if key in self._target_handles:
            handle = self._target_handles.pop(key)
        else:
            handle = open_h5_with_retry(Path(key), "r")
        self._target_handles[key] = handle
        while len(self._target_handles) > self.target_cache_size:
            _, old = self._target_handles.popitem(last=False)
            old.close()
        return torch.from_numpy(np.asarray(handle[self.target_key][coord_index], dtype=np.float32))

    def __getitem__(self, index: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        slide_index, coord_index = (int(index[0]), int(index[1]))
        image, observed_slide_index = super().__getitem__((slide_index, coord_index))
        if observed_slide_index != slide_index:  # pragma: no cover - defensive
            raise RuntimeError("WSI dataset returned an unexpected slide index")
        target = self._target(self.records[slide_index], coord_index)
        return image, target

    def __getitems__(
        self, indices: list[tuple[int, int]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # Reuse the parent's spatially ordered pixel reads, then attach targets
        # in the original sampler order.  This preserves exact image/target pairs.
        images = super().__getitems__(indices)
        return [
            (item[0], self._target(self.records[int(index[0])], int(index[1])))
            for item, index in zip(images, indices)
        ]

    def __del__(self) -> None:
        handles = getattr(self, "_target_handles", None)
        if handles:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception:
                    pass


def build_compact_cache_tile_loaders(
    split_records: dict[str, list[SlideRecord]],
    cache_paths: dict[str, Path],
    transform: Any,
    *,
    batch_size: int,
    slides_per_batch: int,
    train_slides_per_epoch: int,
    train_tiles_per_slide: int,
    val_slides_per_epoch: int,
    val_tiles_per_slide: int,
    num_workers: int,
    prefetch_factor: int,
    slide_cache_size: int,
    openslide_cache_bytes: int,
    cohort_balance_power: float,
    seed: int,
    resize_to: int | None = None,
    target_key: str = "final_attention",
):
    """Build balanced loaders whose second item is a cached per-tile target.

    ``target_key`` selects which cached array is yielded as the target:
    ``"final_attention"`` (default, EAF forecaster's KL target) or
    ``"tile_embeddings"`` (frozen unpruned model's final per-tile embedding, used by
    the pruned-encoder distillation trainer).
    """
    train_dataset = CompactCachedWSITileDataset(
        split_records["train"], cache_paths, transform,
        # Cached targets have a fixed per-tile/per-patch-token layout keyed to the
        # exact cache-time crop. Random flips, rotations, or crop jitter would
        # require applying the identical transform to the target; use the exact
        # deterministic cache-time crop instead.
        augment=False, slide_cache_size=slide_cache_size,
        openslide_cache_bytes=openslide_cache_bytes,
        resize_to=resize_to,
        target_key=target_key,
    )
    val_dataset = CompactCachedWSITileDataset(
        split_records["val"], cache_paths, transform,
        augment=False, slide_cache_size=slide_cache_size,
        openslide_cache_bytes=openslide_cache_bytes,
        resize_to=resize_to,
        target_key=target_key,
    )
    train_sampler = WSIBalancedBatchSampler(
        split_records["train"], batch_size=batch_size,
        slides_per_batch=slides_per_batch, slides_per_epoch=train_slides_per_epoch,
        tiles_per_slide=train_tiles_per_slide, seed=seed,
        cohort_balance_power=cohort_balance_power,
    )
    val_sampler = WSIBalancedBatchSampler(
        split_records["val"], batch_size=batch_size,
        slides_per_batch=slides_per_batch,
        slides_per_epoch=min(val_slides_per_epoch, len(split_records["val"])),
        tiles_per_slide=val_tiles_per_slide, seed=seed + 1,
        cohort_balance_power=cohort_balance_power,
    )
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_sampler=val_sampler, **kwargs
    )
    return train_loader, val_loader, train_sampler, val_sampler
