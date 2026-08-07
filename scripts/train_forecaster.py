#!/usr/bin/env python
"""Compatibility entry point for online WSI tile-level EAF training.

The former implementation materialized source tokens and teacher attention in a
large HDF5 cache. It is intentionally retired: this command now delegates to
``train_wsi_tile_eaf_online.py`` and never creates a tile-feature cache.
"""

from train_wsi_tile_eaf_online import main


if __name__ == "__main__":
    print(
        "[EAF] train_forecaster.py now uses online WSI sampling; "
        "see docs/wsi_tile_online_training.md for the new CLI."
    )
    main()
