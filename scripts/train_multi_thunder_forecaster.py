#!/usr/bin/env python
"""Compatibility entry point for task-agnostic online tile-level EAF training.

Thunder remains the tile-encoder provider, but training samples raw tiles from
canonical WSI manifests and extracts teacher targets in memory. Per-dataset HDF5
feature caches are no longer produced.
"""

from train_wsi_tile_eaf_online import main


if __name__ == "__main__":
    print(
        "[EAF] train_multi_thunder_forecaster.py now delegates to the "
        "online WSI trainer; see docs/wsi_tile_online_training.md."
    )
    main()
