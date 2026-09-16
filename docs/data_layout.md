# Data layout

`$EAF_WSI_ROOT` is the single runtime root.

```text
$EAF_WSI_ROOT/
├── sources/
├── datasets/
│   ├── pretraining/
│   │   └── histai_core_v1/
│   └── downstream/
│       ├── thunder/
│       ├── wsi_level/
│       └── hest/
├── caches/
│   ├── tile_eaf/                         # reusable full tile teacher
│   ├── wsi_eaf/                          # reusable full WSI teacher/hidden
│   └── experiments/
│       └── <experiment>/<variant>/seed_<seed>/
│           ├── tile_pruned/
│           └── wsi_source/
├── checkpoints/
│   ├── tile_eaf/{forecaster,distillation}/...
│   └── wsi_eaf/{forecaster,distillation}/...
├── results/
│   ├── tile_eaf/{forecaster,distillation,evaluation}/...
│   └── wsi_eaf/{forecaster,distillation,evaluation}/...
└── logs/
    ├── tile_eaf/{forecaster,distillation,evaluation}/...
    └── wsi_eaf/{forecaster,distillation,evaluation}/...
```

Existing validated large caches are not moved merely to satisfy this layout. Moving or
duplicating hundreds of GB/TB has no scientific value. Only new experiment-derived
outputs must use registry paths.

New numerical artifacts use `.npyd` directories with memory-mappable `.npy` arrays and
`metadata.json`. Historical HDF5 may be deleted only after the matching `.npyd`
conversion has been verified.

See `docs/experiments.md`.
