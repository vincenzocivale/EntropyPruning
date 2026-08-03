# WSI Unsupervised FM Experiment

This runbook is the shortest path for the experiment:

```text
tile-level FM layer 2 features
  -> EAF tile-importance forecaster
  -> predicted tile importance
  -> prune tiles
  -> materialize kept tiles from a late/full store
  -> evaluate the downstream WSI FM separately
```

The setting is "unsupervised" in the slide-level sense:

- no pathology class labels are required
- the training target is a precomputed per-tile importance score
- train/val/test splits are created by slide id only, not by label

This is the right path when you already have per-slide tile artifacts on disk
from two different models or stages:

- an **early/tile-level feature source** used as EAF input
- a **WSI-level tile-importance target** used as supervision
- optionally a **late/full tile feature source** to materialize after pruning

## Required on-disk layout

The wrapper script assumes one file per slide and identical slide ids across
directories:

```text
early_features/
  slide_001.pt
  slide_002.npy
late_features/
  slide_001.pt
  slide_002.npy
targets/
  slide_001.npy
  slide_002.npy
coords/
  slide_001.npy
  slide_002.npy
```

Supported file formats are the same already supported by the repository:

- features: `.pt`, `.pth`, `.npy`, `.npz`
- targets: `.h5`, `.hdf5`, `.pt`, `.pth`, `.npy`, `.npz`
- coords: `.pt`, `.pth`, `.npy`, `.npz`, or `.h5` if used for target coords

Coordinate alignment is important. For independently produced early, late,
and target artifacts, use exact coords matching rather than index matching.

## One-command wrapper

Use the wrapper added for this scenario:

```bash
EARLY_FEATURES_DIR=/path/to/layer2_features \
LATE_FEATURES_DIR=/path/to/late_features \
TARGETS_DIR=/path/to/wsi_fm_tile_scores \
COORDS_DIR=/path/to/coords \
OUTPUT_ROOT=/path/to/exp_unsup_wsi \
INPUT_FEATURE_DIM=384 \
LATE_FEATURE_DIM=1024 \
TARGET_SOURCE=gigapath_wsi_fm \
TARGET_KEY=tile_importance \
TARGET_NORMALIZE=softmax \
DEVICE=cuda \
OVERWRITE=1 \
bash scripts/run_wsi_unsupervised_fm_experiment_template.sh
```

What it does:

1. Builds manifests for early features, late features, and importance targets.
2. Imports all three into EAF `H5WSIFeatureStore` files.
3. Validates stores and paired alignment.
4. Creates train/val/test splits without labels.
5. Trains `WSITileImportanceForecaster`.
6. Evaluates pruning quality on the held-out test split.
7. Materializes a pruned late/full feature store.

Main outputs:

```text
OUTPUT_ROOT/
  manifests/
  stores/features_layer2.h5
  stores/features_late.h5
  stores/features_wsi_importance.h5
  splits/train.txt
  splits/val.txt
  splits/test.txt
  checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt
  results/exp001_importance_pruning.csv
  features_late_pruned_keep_0.10.h5
```

## Important environment variables

Required:

- `EARLY_FEATURES_DIR`
- `LATE_FEATURES_DIR`
- `TARGETS_DIR`
- `COORDS_DIR`
- `OUTPUT_ROOT`
- `INPUT_FEATURE_DIM`
- `LATE_FEATURE_DIM`

Common optional:

- `TARGET_SOURCE`
- `TARGET_KEY`
- `TARGET_NORMALIZE`
- `EARLY_FEATURE_KEY`
- `LATE_FEATURE_KEY`
- `TARGET_COORDS_DIR`
- `TARGET_COORDS_KEY`
- `DEVICE`
- `LOSS`
- `TOP_K`
- `EPOCHS`
- `BATCH_SIZE`
- `PRUNED_KEEP_RATIO`

## When to change normalization

- `TARGET_NORMALIZE=none`: use when targets are already comparable slide-wise.
- `TARGET_NORMALIZE=sum`: use when targets are non-negative masses that should sum to 1.
- `TARGET_NORMALIZE=softmax`: use when raw WSI-FM scores are unbounded logits.
- `TARGET_NORMALIZE=minmax`: use only when the rank matters more than the scale and each slide has non-constant scores.

For KL training, `softmax` or `sum` is usually the safest starting point if
the WSI-level model emits raw per-tile scores.

## Constraints

- This pipeline learns a **scalar per tile**. If the WSI FM exposes a full
  attention matrix, reduce it to one scalar per tile before import.
- The downstream WSI model evaluation after pruning is still external to this
  wrapper. The wrapper prepares the pruned store that your WSI FM evaluation
  should consume.
- Slide ids and coords must agree across early, late, and target artifacts.
