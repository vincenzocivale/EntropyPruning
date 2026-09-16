# EAF contributor notes

Use the Conda environment in `environment.yml` and run commands from the
repository root.

```bash
conda create -n trident --file environment.yml
conda activate trident
export EAF_WSI_ROOT=/path/to/WSI
pytest
```

`$EAF_WSI_ROOT` is the single location for data, caches, checkpoints, results
and logs. The repository stores code and small metadata only.

The supported workflow is documented in `README.md`, `docs/data_layout.md` and
`docs/pipeline.md`. Prefer `python scripts/eaf.py ...` for data and cache
operations. Preserve raw WSI, existing TCGA/HEST assets and the separation
between pretraining and downstream datasets.

New derived numerical data use `.npyd` directories. HDF5 remains readable for
historical artifacts during migration.
