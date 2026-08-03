# Data Layout

All data under `data/` is local-only and ignored by Git. The repository
contains code, manifests templates, and documentation; it must not contain raw
WSI files, extracted features, checkpoints, or runtime logs.

```text
data/
  labeled/
    tile_level/
      thunder_tiles -> external Thunder dataset (symlink; never copied)
    wsi_level/
      <cohort>/
        manifests/              # slide id and downstream label
        splits/                 # train/validation/test slide-id lists
  unlabeled/
    tcga/
      <cohort>/
        raw_wsi/                # one physical copy of source slides
        trident/                # reusable segmentation, coordinates and tile features
        eaf/
          tile_encoder_attention/ # tile-level EAF cache, if generated
        manifests/              # acquisition and feature manifests
        splits/                 # unlabeled EAF splits
```

## Placement rules

- Keep raw slides only in `data/unlabeled/.../raw_wsi/`; consumers reference
  them through manifests or symlinks rather than copying them.
- Keep reusable TRIDENT outputs next to the cohort in `trident/`.
- Keep Thunder data outside this repository and expose it only through the
  `data/labeled/tile_level/thunder_tiles` symlink.
- Treat aggregate WSI feature stores, WSI attention targets, WSI checkpoints,
  ranking outputs, and logs as run artifacts. Place them outside `data/` or
  remove them after a run; they are not a canonical dataset.
- The current checkout intentionally contains no retained WSI forecaster
  feature store or WSI run checkpoint. Recreate those artifacts from the raw
  slides/TRIDENT outputs when resuming WSI work.

The generic WSI pipeline and its HDF5 contracts are documented in
`docs/wsi_attention_forecasting.md`. The Thunder tile-level pipeline remains
independent from the WSI layout.
