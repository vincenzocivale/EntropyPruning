# Data layout

`$EAF_WSI_ROOT` is the only runtime root. The repository must contain code and
small metadata only.

```text
$EAF_WSI_ROOT/
├── sources/       raw WSI; exactly one physical copy
├── datasets/      manifests, views, TRIDENT artifacts and labels
├── caches/        frozen Tile-EAF and WSI-EAF teacher outputs
├── archives/      verified cold pixel archives
├── checkpoints/   model weights
├── results/       summaries, metrics and experiment catalog
└── logs/          local runtime logs
```

Pretraining and downstream data are separate namespaces:

```text
datasets/pretraining/
datasets/downstream/
```

A downstream benchmark must never enter a strict pretraining manifest. Existing
TCGA and HEST assets are preserved in place.

## Numerical artifacts

New derived arrays use one `<slide_id>.npyd/` directory per slide:

```text
<slide_id>.npyd/
├── coords.npy
├── tile_embeddings.npy
├── final_attention.npy
└── metadata.json
```

Nested fields retain their path, for example
`attention/global_to_tiles_mass_share.npy`. Arrays can be read lazily with
`numpy.load(path, mmap_mode="r")`. Metadata records schema, model revision,
preprocessing, dtype, attention policy and completion state.

HDF5 is accepted for existing inputs. Convert derived HDF5 safely and
resumably, without modifying the source file:

```bash
python scripts/eaf.py cache convert-numpy --scope caches --workers 8
python scripts/eaf.py cache convert-numpy --scope datasets --workers 4
```

The converter validates every array and attribute before publishing the NumPy
directory. It never converts or removes raw SVS/DICOM files.

## Data safety

- Do not duplicate raw WSI outside `sources/`.
- Do not remove raw data before its cold archive and required caches validate.
- Do not commit datasets, caches, checkpoints, logs or results.
- Use manifests and symlinks instead of copies.
