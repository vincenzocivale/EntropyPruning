# Data layout

`$EAF_WSI_ROOT` is the single runtime root. The git repository contains code and
small metadata only.

```text
$EAF_WSI_ROOT/
├── sources/       raw WSI
├── datasets/
│   ├── pretraining/histai_eaf_wsi_v1/
│   └── downstream/                 # THUNDER/EAGLE labels/manifests only
├── caches/
│   ├── tile_full/
│   ├── tile_pruned/
│   ├── wsi_full_teacher/
│   └── wsi_pruned_source/
├── checkpoints/
├── results/
└── logs/
```

The primary training corpus is HISTAI. Downstream benchmark data never enters an
EAF/embedding-distillation training manifest.

New generated numerical artifacts use `.npyd` directories and `metadata.json`;
historical HDF5 remains readable during migration.
