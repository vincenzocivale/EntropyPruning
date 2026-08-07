# Data layout

The repository does **not** own raw WSI data. Code and documentation live in
Git; raw slides, TRIDENT coordinates, model checkpoints, and runtime logs live
in an external, versioned data root.

The current canonical root on the shared server is:

```text
/data2/home/vcivale/projects/imaging/data/WSI
```

Set it once per shell:

```bash
export EAF_WSI_ROOT=/data2/home/vcivale/projects/imaging/data/WSI
```

All new commands accept `--data-root`, so another machine can use the same
layout under a different absolute path.

## Canonical store

```text
$EAF_WSI_ROOT/
├── catalog/
│   ├── slides.csv
│   ├── artifacts.csv
│   ├── datasets.csv
│   └── encoders.csv
├── sources/
│   └── gdc/tcga/<cohort>/<diagnostic|tissue>/*.svs
├── datasets/
│   ├── pretraining/
│   │   └── tcga_eaf_multicohort_v1/
│   │       ├── dataset.yaml
│   │       ├── manifests/
│   │       │   ├── slides.csv
│   │       │   └── trident_pending_*.csv
│   │       ├── views/
│   │       │   └── raw_flat/*.svs -> ../../../../../sources/...
│   │       └── artifacts/
│   │           ├── trident/
│   │           │   ├── contours/
│   │           │   ├── contours_geojson/
│   │           │   ├── thumbnails/
│   │           │   ├── wsi_states/
│   │           │   └── 20x_512px_0px_overlap/
│   │           │       ├── patches/*_patches.h5
│   │           │       └── visualization/
│   │           └── tile_fm/
│   │               └── conch_v15/       # retained legacy features
│   └── benchmarks/                       # labeled downstream datasets
├── logs/
├── cache/
└── quarantine/
```

Raw WSI files have exactly one physical copy under `sources/`. Dataset views
and compatibility paths are symlinks; scripts must not copy slides into the
repository.

## Canonical slide manifest

Online tile training reads:

```text
$EAF_WSI_ROOT/datasets/pretraining/tcga_eaf_multicohort_v1/manifests/slides.csv
```

Required columns are:

| Column | Meaning |
|---|---|
| `slide_id` | stable slide identifier |
| `case_id` | patient/case identifier used for case-disjoint splitting |
| `cohort` | cohort used by the balanced sampler |
| `raw_path` | WSI path, absolute or relative to `--data-root` |
| `coords_path` | TRIDENT HDF5 coordinate path |

The loader also understands `slide_group`, `include_in_pretraining`,
`coords_available`, `preprocessing_status`, and an optional `split` column.
Rows without an explicit split are assigned deterministically by `case_id`; no
runtime split file is written.

## TRIDENT coordinate contract

The preprocessing profile is:

```text
magnification: 20x
patch size:    512 px
stride:        512 px
overlap:       0 px
```

Coordinates are stored as one HDF5 file per slide:

```text
artifacts/trident/20x_512px_0px_overlap/patches/<slide_id>_patches.h5
```

The online loader searches for a two-dimensional coordinate dataset named
`coords`, `coordinates`, or `patches/coords`, then falls back to the first
`[N, >=2]` HDF5 dataset. Coordinates are interpreted as level-0 OpenSlide
locations. `patch_size_level0` takes precedence because a 512 px patch at 20x
corresponds to a larger level-0 crop on a 40x-native slide. Older coordinate
files fall back to `patch_level`, `patch_size`, and magnification attributes,
then to the CLI-configurable default.

When the encoder expects a smaller physical field of view than the canonical
512 px@20x coordinate window, pass `--tile-size-at-target-mag`. Training uses a
random sub-crop inside the coordinate window; validation uses its centered
sub-crop. This preserves the requested magnification/FOV instead of merely
resizing a 512 px field to a 224/256 px model input.

## What is no longer stored

Tile-level EAF training and pruning-aware tile-encoder adaptation no longer
materialize any of the following:

```text
per-tile source embeddings
per-tile teacher attention maps
per-dataset forecaster HDF5 caches
full duplicated fine-tuned backbones
```

Tiles are read from WSI files on demand. Source tokens, teacher attention, and
full-teacher embeddings exist only in GPU memory for the current batch.

The only persistent training outputs are small, reproducible run artifacts:

```text
checkpoints/wsi_tile_eaf_online/
  best_<run>.pt
  summary_<run>.json

checkpoints/wsi_tile_pruned_online/
  best_<run>_adapter.pt       # trainable LoRA tensors only
  summary_<run>.json
```

Metrics, sampling coverage, throughput, early-stopping state, and peak CUDA
memory are logged to Weights & Biases. Set `--wandb-mode offline` or
`--wandb-mode disabled` when appropriate.

## Legacy artifacts

Existing CONCH feature stores are retained under `artifacts/tile_fm/conch_v15`
for audit and comparison. They are not inputs to the new online training path
and are not extended to newly preprocessed slides. THUNDER supplies and
initializes each supported tile encoder at training time.

See [`wsi_tile_online_training.md`](wsi_tile_online_training.md) for the
sampling policy, training commands, and storage guarantees.
