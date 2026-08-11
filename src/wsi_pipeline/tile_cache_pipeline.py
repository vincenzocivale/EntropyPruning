"""End-to-end offline Tile-EAF/WSI-EAF cache production, one WSI at a time.

    dataset adapter
          v
    segmentation / coords            (external: TRIDENT, not this module)
          v
    TileEncoderAdapter                (HookedViTTileTeacherAdapter, frozen, eval())
          v
    offline PERMANENT tile cache      (TileCacheWriter -> coords, final_attention,
          |                            tile_embeddings only -- versioned per-slide HDF5)
          v
    EAF Tile training                 (final_attention read from cache; early_tokens
                                        recomputed ONLINE, see compact_cache_dataset.py)

The ``tile_embeddings`` array this module writes is the same shared artifact WSI-EAF
consumes as input to its own teacher cache -- it is not exclusive to Tile-EAF. See
``docs/offline_eaf_pipeline.md``.

This module makes exactly one full forward pass per tile batch when *building* the
cache: ``final_attention`` (the Tile-EAF teacher target) and ``tile_embeddings`` (final
pooled representation) are both read off that single pass via
``HookedViTTileTeacherAdapter.extract_final`` -- there is no separate forward per
quantity. ``early_tokens`` (the layer-2 representation) is deliberately NOT computed or
stored here at all: it is >99.8% of what a combined cache would cost on disk (~1.53
MiB/tile vs. ~3.1 KiB/tile for everything else), and is instead recomputed online,
per training step, via the much cheaper ``HookedViTTileTeacherAdapter.extract_early``
(an early-exit partial forward through blocks 0..early_layer only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .cache_contracts import TileCacheSpec
from .cache_io import TileCacheWriter, tile_cache_status
from .model_adapters import HookedViTTileTeacherAdapter
from .patch_dataset import OpenSlideCoordinateDataset


@dataclass(frozen=True)
class TileCacheItem:
    slide_id: str
    case_id: str
    wsi_path: Path
    coords_path: Path


@dataclass(frozen=True)
class TileCacheRunConfig:
    output_dir: Path
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 2
    device: str = "cuda"
    compression: str | None = "lzf"
    overwrite: bool = False
    profile: bool = False


def cache_path_for(output_dir: Path, slide_id: str) -> Path:
    return Path(output_dir) / f"{slide_id}.h5"


# Revision tag recorded in every conch_v15 TileCacheSpec/cache_id. Exposed as a
# constant (not just a literal inside build_encoder) so callers that need to
# pre-compute a matching cache_id/output path -- e.g. the HISTAI orchestrator's
# per-subset directory naming -- never have to instantiate the model to do it, and
# can never drift from the string build_encoder() actually uses.
CONCH_V15_REVISION = "titan-return_conch"


def build_encoder(
    encoder_name: str, *, token: str | None, device: torch.device
) -> HookedViTTileTeacherAdapter:
    if encoder_name == "conch_v15":
        return HookedViTTileTeacherAdapter.from_conch(
            token=token, revision=CONCH_V15_REVISION, device=device
        )
    raise ValueError(
        f"Unsupported --encoder {encoder_name!r}; only 'conch_v15' is wired end-to-end today. "
        "Extend HookedViTTileTeacherAdapter.from_timm(...) for UNI/UNI2/Virchow-style teachers."
    )


@torch.no_grad()
def probe_batch_size(
    adapter: HookedViTTileTeacherAdapter,
    *,
    device: torch.device,
    candidates: tuple[int, ...] = (32, 64, 96),
) -> int:
    """Find the largest batch size in ``candidates`` that fits ``extract_final``
    (the cache-building hot path -- a full forward) without OOM.

    Runs throwaway forward passes on random pixels -- a failed candidate never touches
    cache I/O, so an OOM here only narrows the search, never corrupts anything.
    Candidates are tried in the given order and assumed monotonically increasing;
    the first OOM stops the probe (a smaller size succeeding after a larger one failed
    would indicate flaky external GPU pressure, not a batch-size effect).
    """
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("batch-size candidates must be positive")
    if tuple(sorted(candidates)) != candidates:
        raise ValueError("batch-size candidates must be monotonically increasing")
    size = adapter.input_size or 224
    best: int | None = None
    for batch_size in candidates:
        try:
            dummy = torch.randn(batch_size, 3, size, size, device=device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                adapter.extract_final(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            best = batch_size
            del dummy
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
    if best is None:
        raise RuntimeError(
            f"Even the smallest candidate batch_size={candidates[0]} ran out of memory"
        )
    return best


@torch.no_grad()
def autotune_loader(
    item: TileCacheItem,
    adapter: HookedViTTileTeacherAdapter,
    *,
    device: torch.device,
    batch_size_candidates: tuple[int, ...],
    worker_candidates: tuple[int, ...] = (4, 8, 16),
    prefetch_candidates: tuple[int, ...] = (2, 4),
    benchmark_batches: int = 4,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    """Tune on real WSI reads after selecting the largest batch that fits.

    The benchmark is deliberately short and publishes no cache.  It measures the
    complete loader + H2D + teacher path, which catches configurations where a
    synthetic OOM probe chooses a batch that starves on OpenSlide I/O.
    """
    batch_size = probe_batch_size(
        adapter, device=device, candidates=batch_size_candidates
    )
    dataset = OpenSlideCoordinateDataset(
        item.wsi_path, item.coords_path, adapter.transform, output_size=adapter.input_size
    )
    results: list[dict[str, Any]] = []
    for workers in worker_candidates:
        for prefetch in prefetch_candidates:
            kwargs: dict[str, Any] = {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": workers,
                "pin_memory": device.type == "cuda",
                "persistent_workers": workers > 0,
            }
            if workers > 0:
                kwargs["prefetch_factor"] = prefetch
            loader = DataLoader(**kwargs)
            count = 0
            started = time.perf_counter()
            for batch_index, (images, _coords) in enumerate(loader):
                if batch_index >= benchmark_batches:
                    break
                images = images.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    adapter.extract_final(images)
                count += int(images.shape[0])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "batch_size": batch_size,
                    "num_workers": workers,
                    "prefetch_factor": prefetch,
                    "tiles": count,
                    "elapsed_seconds": elapsed,
                    "tiles_per_second": count / max(elapsed, 1e-9),
                }
            )
            del loader
    best = max(results, key=lambda row: row["tiles_per_second"])
    return batch_size, int(best["num_workers"]), int(best["prefetch_factor"]), results


def cache_one_slide(
    item: TileCacheItem,
    *,
    adapter: HookedViTTileTeacherAdapter,
    spec: TileCacheSpec,
    config: TileCacheRunConfig,
) -> dict[str, Any]:
    """Build (or validate/skip) the tile cache for exactly one WSI.

    Resume behavior: if a valid, ``complete``, matching-``cache_id`` cache already
    exists with the same tile count as ``item.coords_path``, this is a no-op unless
    ``config.overwrite``. A corrupt/partial/stale cache is silently rebuilt from
    scratch (it never gets treated as "close enough").
    """
    output_path = cache_path_for(config.output_dir, item.slide_id)
    if not config.overwrite:
        status = tile_cache_status(output_path, coords_path=item.coords_path, spec=spec)
        if status["ok"]:
            return {
                "slide_id": item.slide_id,
                "status": "skipped_valid",
                "path": str(output_path),
                **status,
            }

    device = torch.device(config.device)
    dataset = OpenSlideCoordinateDataset(
        item.wsi_path, item.coords_path, adapter.transform, output_size=adapter.input_size
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{item.slide_id}: coords registry has zero tiles: {item.coords_path}")

    loader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        # A fresh DataLoader is built per slide and iterated exactly once, so
        # persistent_workers would buy nothing but delay worker teardown into the
        # window where this slide's cache file gets reopened for its post-write
        # integrity check -- avoid that race entirely rather than only retrying it.
        persistent_workers=False,
    )
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
    loader = DataLoader(**loader_kwargs)

    n_written = 0
    if config.profile and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wait_seconds = 0.0
    transfer_compute_seconds = 0.0
    write_seconds = 0.0
    started = time.perf_counter()
    # TileCacheWriter is atomic (tmp -> rename on clean __exit__ only): an exception
    # anywhere in this loop (including CUDA OOM) leaves no file at `output_path`.
    with TileCacheWriter(
        output_path,
        spec,
        slide_id=item.slide_id,
        case_id=item.case_id,
        compression=config.compression,
        expected_n=len(dataset),
    ) as writer:
        iterator = iter(loader)
        progress = tqdm(total=len(dataset), desc=item.slide_id, unit="tile", leave=False)
        try:
            while True:
                waited_at = time.perf_counter()
                try:
                    images, coords = next(iterator)
                except StopIteration:
                    break
                wait_seconds += time.perf_counter() - waited_at
                compute_at = time.perf_counter()
                images = images.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    # extract_final, not the combined extract(): early_tokens is not part
                    # of the permanent cache (CACHE_SCHEMA_VERSION v2) -- EAF Tile training
                    # recomputes it online instead (compact_cache_dataset.py).
                    output = adapter.extract_final(images)
                transfer_compute_seconds += time.perf_counter() - compute_at
                write_at = time.perf_counter()
                writer.append(
                    coords=coords.numpy(),
                    final_attention=output.final_attention.numpy(),
                    tile_embeddings=output.tile_embeddings.numpy(),
                )
                write_seconds += time.perf_counter() - write_at
                take = int(images.shape[0])
                n_written += take
                progress.update(take)
        finally:
            progress.close()

    final_status = tile_cache_status(output_path, coords_path=item.coords_path, spec=spec)
    if not final_status["ok"]:
        raise RuntimeError(
            f"{item.slide_id}: cache failed post-write integrity check: {final_status['reason']}"
        )
    elapsed = time.perf_counter() - started
    row = {
        "slide_id": item.slide_id,
        "status": "built",
        "path": str(output_path),
        "n_tiles": n_written,
        **final_status,
    }
    if config.profile:
        row.update(
            elapsed_seconds=elapsed,
            tiles_per_second=n_written / max(elapsed, 1e-9),
            data_wait_seconds=wait_seconds,
            transfer_compute_seconds=transfer_compute_seconds,
            write_seconds=write_seconds,
        )
        if device.type == "cuda":
            row["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    return row


def cache_many_slides(
    items: Iterable[TileCacheItem],
    *,
    adapter: HookedViTTileTeacherAdapter,
    spec: TileCacheSpec,
    config: TileCacheRunConfig,
) -> list[dict[str, Any]]:
    """Cache every slide in ``items``; one failure never aborts the rest of the run."""
    item_list = list(items)
    rows: list[dict[str, Any]] = []
    for item in tqdm(item_list, desc="tile-cache slides", unit="slide"):
        try:
            rows.append(cache_one_slide(item, adapter=adapter, spec=spec, config=config))
        except Exception as exc:  # noqa: BLE001 - record and continue, never silently drop a slide
            rows.append({"slide_id": item.slide_id, "status": "error", "error": repr(exc)})
    return rows
