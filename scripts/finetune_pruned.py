#!/usr/bin/env python
"""Compatibility entry point for online pruning-aware tile-encoder adaptation.

The retired implementation optimized a dataset-specific classification head.
The replacement is task-agnostic full-vs-pruned embedding distillation over WSI
tiles, with no tile cache and no full-backbone checkpoint.
"""

from finetune_wsi_tile_encoder_pruned_online import main


if __name__ == "__main__":
    print(
        "[EAF] finetune_pruned.py now performs task-agnostic online WSI "
        "distillation; see docs/wsi_tile_online_training.md."
    )
    main()
