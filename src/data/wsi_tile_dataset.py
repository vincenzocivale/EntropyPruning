import csv
import multiprocessing as mp
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from trident import load_wsi
from trident.wsi_objects.WSIPatcher import WSIPatcher
from trident.segmentation_models import segmentation_model_factory


def load_mpp_map(csv_path):
    """Load a {wsi_basename: mpp} map from a CSV with `wsi` and `mpp` columns.

    The `wsi` column may be an absolute or relative path; only the basename is keyed.
    Returns an empty dict if the file doesn't exist or is empty.
    """
    if csv_path is None or not os.path.exists(csv_path):
        return {}
    out = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("wsi") or row.get("slide") or row.get("path")
            mpp = row.get("mpp") or row.get("pixel_size")
            if path and mpp:
                out[os.path.basename(path)] = float(mpp)
    return out


def _spawn_init(project_root):
    """Initializer for spawn workers — restores sys.path so imports resolve."""
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)


def _index_one_wsi(args):
    """Worker function: load a WSI, segment tissue, pick `tiles_per_wsi` valid coords.

    Runs in a subprocess. Returns (wsi_path, src_mpp, level, patch_size_level, coords)
    where `coords` is an (n, 2) int array of (x, y) at level 0. Returns None on failure.
    """
    wsi_path, src_mpp, mag, patch_size, tiles_per_wsi, seed = args
    try:
        # Pass mpp at construction so TRIDENT doesn't try (and fail) to auto-detect.
        load_kwargs = {"mpp": src_mpp} if src_mpp is not None else {}
        try:
            wsi = load_wsi(wsi_path, **load_kwargs)
        except Exception:
            # Some readers don't accept the kwarg — retry without and patch after.
            wsi = load_wsi(wsi_path)
            if src_mpp is not None:
                wsi.mpp = src_mpp

        otsu = segmentation_model_factory("otsu")
        mask_gdf = wsi.segment_tissue(otsu, target_mag=1.25, job_dir=None)

        patcher_kwargs = dict(
            patch_size=patch_size,
            dst_mag=mag,
            mask=mask_gdf,
            pil=True,
            overlap=0,
            threshold=0.0,
            coords_only=True,
        )
        if src_mpp is not None:
            patcher_kwargs["src_pixel_size"] = src_mpp

        patcher = WSIPatcher(wsi, **patcher_kwargs)
        n_tiles = len(patcher)
        if n_tiles == 0:
            return None

        if tiles_per_wsi is None or tiles_per_wsi <= 0:
            # Use every valid coord — no subsampling.
            coords = patcher.valid_coords.astype(np.int64)
        else:
            rng = np.random.RandomState(seed + abs(hash(wsi_path)) % (2**31))
            sel = rng.choice(n_tiles, size=tiles_per_wsi, replace=(tiles_per_wsi > n_tiles))
            coords = patcher.valid_coords[sel].astype(np.int64)
        return (wsi_path, src_mpp, int(patcher.level), int(patcher.patch_size_level), coords)
    except Exception as e:  # noqa: BLE001
        return ("__error__", wsi_path, str(e))


class WSITileDataset(Dataset):
    """On-the-fly WSI tile dataset.

    Indexing (segmentation + tile selection) runs in parallel across worker processes.
    The main process only keeps a flat list of per-tile records — the heavy WSIPatcher
    objects are dropped once their coordinates are extracted.

    During training, DataLoader workers reopen WSIs lazily and cache a small LRU of
    OpenSlide handles per worker. RAM stays bounded.
    """

    def __init__(
        self,
        wsi_paths,
        transform,
        mag=20,
        patch_size=256,
        tiles_per_wsi=64,
        seed=42,
        verbose=False,
        mpp_map=None,
        num_prep_workers=8,
        worker_handle_cache=16,
    ):
        """
        Args:
            wsi_paths: list of paths to WSI files.
            transform: torchvision.transforms.Compose for preprocessing.
            mag: target magnification for tile extraction.
            patch_size: tile size in pixels at the target magnification.
            tiles_per_wsi: tiles sampled per WSI per epoch.
            seed: base random seed for reproducible per-WSI sampling.
            verbose: print progress bar.
            mpp_map: optional {basename: mpp} dict, used when TRIDENT can't infer MPP.
            num_prep_workers: subprocesses used for parallel segmentation/indexing.
            worker_handle_cache: per-DataLoader-worker LRU size for opened WSI handles.
        """
        self.wsi_paths = [str(p) for p in wsi_paths]
        self.transform = transform
        self.mag = mag
        self.patch_size = patch_size
        self.tiles_per_wsi = tiles_per_wsi
        self.seed = seed
        self.verbose = verbose
        self.mpp_map = mpp_map or {}
        self.worker_handle_cache = worker_handle_cache

        # Per-tile record: (wsi_path, src_mpp_or_None, x, y, level, patch_size_level)
        self.tile_records = []
        self._wsi_count = 0

        tasks = [
            (
                p,
                self.mpp_map.get(os.path.basename(p)),
                self.mag,
                self.patch_size,
                self.tiles_per_wsi,
                self.seed,
            )
            for p in self.wsi_paths
        ]

        n_workers = max(1, num_prep_workers)
        if n_workers == 1:
            results_iter = (_index_one_wsi(t) for t in tasks)
        else:
            # `spawn` avoids inheriting CUDA state from the parent process — fork would
            # crash with "Cannot re-initialize CUDA in forked subprocess" because the
            # main process has already loaded the encoder onto GPU.
            ctx = mp.get_context("spawn")
            project_root = str(Path(__file__).resolve().parents[2])
            executor = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_spawn_init,
                initargs=(project_root,),
            )
            futures = [executor.submit(_index_one_wsi, t) for t in tasks]
            results_iter = (f.result() for f in as_completed(futures))

        n_failed = 0
        n_empty = 0
        try:
            iterator = tqdm(
                results_iter,
                total=len(tasks),
                desc=f"Indexing WSIs (parallel x{n_workers})",
                disable=not verbose,
            )
            for result in iterator:
                if result is None:
                    n_empty += 1
                    continue
                if isinstance(result, tuple) and result and result[0] == "__error__":
                    n_failed += 1
                    if verbose:
                        print(f"  Error indexing {result[1]}: {result[2]}")
                    continue
                wsi_path, src_mpp, level, patch_size_level, coords = result
                self._wsi_count += 1
                for x, y in coords:
                    self.tile_records.append(
                        (wsi_path, src_mpp, int(x), int(y), level, patch_size_level)
                    )
        finally:
            if n_workers > 1:
                executor.shutdown(wait=True)

        if verbose:
            print(
                f"Indexed {self._wsi_count} WSIs ({len(self.tile_records)} tiles) "
                f"| empty: {n_empty}, failed: {n_failed}"
            )

    def __len__(self):
        return len(self.tile_records)

    def _get_handle(self, wsi_path, src_mpp):
        """Per-process LRU cache of opened WSI handles."""
        if not hasattr(self, "_handle_cache"):
            self._handle_cache = OrderedDict()
        if wsi_path in self._handle_cache:
            self._handle_cache.move_to_end(wsi_path)
            return self._handle_cache[wsi_path]
        load_kwargs = {"mpp": src_mpp} if src_mpp is not None else {}
        try:
            wsi = load_wsi(wsi_path, **load_kwargs)
        except Exception:
            wsi = load_wsi(wsi_path)
            if src_mpp is not None:
                wsi.mpp = src_mpp
        self._handle_cache[wsi_path] = wsi
        if len(self._handle_cache) > self.worker_handle_cache:
            self._handle_cache.popitem(last=False)
        return wsi

    def __getitem__(self, idx):
        wsi_path, src_mpp, x, y, level, patch_size_level = self.tile_records[idx]
        wsi = self._get_handle(wsi_path, src_mpp)
        tile_pil = wsi.read_region(
            location=(int(x), int(y)),
            level=int(level),
            size=(int(patch_size_level), int(patch_size_level)),
            read_as="pil",
        )
        # Match the prior pipeline: rescale to `patch_size` if level read isn't already.
        if tile_pil.size != (self.patch_size, self.patch_size):
            tile_pil = tile_pil.resize((self.patch_size, self.patch_size))
        return self.transform(tile_pil)

    @property
    def n_wsis(self):
        return self._wsi_count
