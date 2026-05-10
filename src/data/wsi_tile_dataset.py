import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm.auto import tqdm
from collections import OrderedDict

from trident import load_wsi
from trident.wsi_objects.WSIPatcher import WSIPatcher
from trident.segmentation_models import segmentation_model_factory


class WSITileDataset(Dataset):
    """
    On-the-fly WSI tile dataset. Loads WSI files, segments tissue, and samples random tiles.
    Supports in-memory caching after first pass for faster subsequent epochs.
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
        use_cache=True,
        cache_memory_limit_gb=4.0,
    ):
        """
        Args:
            wsi_paths: list of paths to WSI files (str or Path)
            transform: torchvision.transforms.Compose for preprocessing
            mag: target magnification for tile extraction (default: 20)
            patch_size: tile size in pixels (default: 256)
            tiles_per_wsi: number of tiles to sample per WSI per epoch (default: 64)
            seed: random seed for reproducible sampling (default: 42)
            verbose: print progress (default: False)
            use_cache: cache tile tensors in RAM after first full pass (default: True)
            cache_memory_limit_gb: max memory for tile cache in GB (default: 4.0, 0 = unlimited)
        """
        self.wsi_paths = [str(p) for p in wsi_paths]
        self.transform = transform
        self.mag = mag
        self.patch_size = patch_size
        self.tiles_per_wsi = tiles_per_wsi
        self.seed = seed
        self.verbose = verbose
        self.use_cache = use_cache
        self.cache_memory_limit_gb = cache_memory_limit_gb
        self.cache_memory_limit_bytes = int(cache_memory_limit_gb * 1e9)

        self.rng = np.random.RandomState(seed)
        self.patchers = []
        self.tile_indices = []  # per-WSI list of randomly selected tile indices
        self.tile_cache = OrderedDict()  # LRU cache: (wsi_idx, tile_idx) -> tensor
        self.cache_bytes_used = 0  # track memory usage
        self.first_pass_complete = False

        # Load WSI files and build patchers
        otsu = segmentation_model_factory("otsu")

        iterator = tqdm(
            enumerate(self.wsi_paths),
            total=len(self.wsi_paths),
            desc="Loading and segmenting WSIs",
            disable=not verbose,
        )

        for wsi_idx, wsi_path in iterator:
            try:
                wsi = load_wsi(wsi_path)
                mask_gdf = wsi.segment_tissue(otsu, target_mag=1.25, job_dir=None)

                patcher = WSIPatcher(
                    wsi,
                    patch_size=self.patch_size,
                    dst_mag=self.mag,
                    mask=mask_gdf,
                    pil=True,
                    overlap=0,
                    threshold=0.0,
                )

                n_tiles = len(patcher)
                if n_tiles == 0:
                    if self.verbose:
                        print(f"  Warning: WSI {wsi_path} has no tissue tiles, skipping")
                    continue

                self.patchers.append(patcher)

                # Select random tile indices (with replacement if tiles_per_wsi > n_tiles)
                selected_indices = self.rng.choice(
                    n_tiles, size=self.tiles_per_wsi, replace=(self.tiles_per_wsi > n_tiles)
                )
                self.tile_indices.append(selected_indices)

            except Exception as e:
                if self.verbose:
                    print(f"  Error loading {wsi_path}: {e}")
                continue

        if self.verbose:
            print(f"Successfully loaded {len(self.patchers)} WSIs")

    def __len__(self):
        """Total number of tiles available (across all WSIs)."""
        return len(self.patchers) * self.tiles_per_wsi

    def __getitem__(self, idx):
        """
        Fetch a tile by global index.

        Args:
            idx: global index in range [0, len(self))

        Returns:
            tile_tensor: transformed tile image tensor (3, H, W)
        """
        # Map global index to (wsi_idx, tile_idx)
        wsi_idx = idx // self.tiles_per_wsi
        tile_local_idx = idx % self.tiles_per_wsi

        # Check cache first (move to end for LRU)
        cache_key = (wsi_idx, tile_local_idx)
        if cache_key in self.tile_cache:
            self.tile_cache.move_to_end(cache_key)  # Mark as recently used
            return self.tile_cache[cache_key]

        # Load tile from patcher
        patcher = self.patchers[wsi_idx]
        selected_tile_index = self.tile_indices[wsi_idx][tile_local_idx]
        tile_pil, _, _ = patcher[selected_tile_index]

        # Transform to tensor
        tile_tensor = self.transform(tile_pil)

        # Cache if enabled and first pass is complete
        if self.use_cache and self.first_pass_complete:
            self._add_to_cache(cache_key, tile_tensor)

        return tile_tensor

    def _add_to_cache(self, cache_key, tile_tensor):
        """
        Add tile to cache with LRU eviction if memory limit exceeded.

        Args:
            cache_key: (wsi_idx, tile_idx)
            tile_tensor: tensor to cache
        """
        # Estimate memory size (rough: tensor size in bytes)
        tensor_bytes = tile_tensor.element_size() * tile_tensor.nelement()

        # Check if adding would exceed memory limit
        if (
            self.cache_memory_limit_bytes > 0
            and self.cache_bytes_used + tensor_bytes > self.cache_memory_limit_bytes
        ):
            # Evict oldest (first) item
            if self.tile_cache:
                evicted_key, evicted_tensor = self.tile_cache.popitem(last=False)
                evicted_bytes = evicted_tensor.element_size() * evicted_tensor.nelement()
                self.cache_bytes_used -= evicted_bytes
                if self.verbose:
                    print(
                        f"  Cache evicted {evicted_key}, used: {self.cache_bytes_used / 1e9:.2f} GB"
                    )

        # Add new item
        self.tile_cache[cache_key] = tile_tensor
        self.cache_bytes_used += tensor_bytes

        if self.verbose and len(self.tile_cache) % 100 == 0:
            print(
                f"  Cache size: {len(self.tile_cache)} tiles, {self.cache_bytes_used / 1e9:.2f} GB / {self.cache_memory_limit_gb:.1f} GB"
            )

    def mark_epoch_complete(self):
        """
        Call after one full epoch to enable caching for subsequent accesses.
        This method should be called after the first full pass through the dataset.
        """
        self.first_pass_complete = True
        if self.verbose:
            print(f"First epoch complete. Tile caching enabled ({len(self.tile_cache)} cached so far)")

    def clear_cache(self):
        """Clear the tile tensor cache."""
        self.tile_cache.clear()
        self.cache_bytes_used = 0
        self.first_pass_complete = False

    @property
    def n_wsis(self):
        """Number of WSIs loaded."""
        return len(self.patchers)

    @property
    def total_tiles_available(self):
        """Total number of tiles across all WSIs."""
        return sum(len(self.patchers[i]) for i in range(len(self.patchers)))
