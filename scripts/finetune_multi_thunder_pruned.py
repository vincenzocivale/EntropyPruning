#!/usr/bin/env python
"""Compatibility entry point for online pruning-aware tile-encoder adaptation.

The previous multi-head supervised trainer has been replaced by a single
encoder-specific, downstream-task-independent distillation stage over WSI tiles.
"""

from finetune_wsi_tile_encoder_pruned_online import main


if __name__ == "__main__":
    print(
        "[EAF] finetune_multi_thunder_pruned.py now delegates to the "
        "task-agnostic online WSI trainer; see documentation."
    )
    main()
