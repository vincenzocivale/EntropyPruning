# EAF

EAF trains lightweight attention forecasters for pathology foundation models.
The repository contains code only. WSI, caches, checkpoints and results live
under `$EAF_WSI_ROOT`.

## Start here

```bash
conda create -n trident --file environment.yml
conda activate trident
export EAF_WSI_ROOT=/path/to/WSI

python scripts/eaf.py layout --data-root "$EAF_WSI_ROOT"
python scripts/eaf.py data build-strict --data-root "$EAF_WSI_ROOT"
python scripts/eaf.py cache tile --help
```

The operational commands are deliberately few:

| Goal | Command |
| --- | --- |
| Inspect data layout | `python scripts/eaf.py layout` |
| Acquire or register data | `python scripts/eaf.py data ...` |
| Build or validate Tile-EAF caches | `python scripts/eaf.py cache tile` / `cache validate` |
| Convert historical derived data | `python scripts/eaf.py cache convert-numpy --scope caches` |
| Build a strict cache index | `python scripts/eaf.py cache index-tile` |
| Audit prior experiments | `python scripts/eaf.py experiments audit` |
| Check raw-data release conditions | `python scripts/eaf.py archive lifecycle` |

New numerical artifacts use `.npyd` directories: memory-mappable `.npy`
arrays and `metadata.json`. Historical HDF5 remains readable during migration.
Raw WSI stay in their original pyramidal format under `sources/`.

Read [data layout](docs/data_layout.md) before changing data,
[pipeline](docs/pipeline.md) before running an experiment, and
[project status](docs/continuity.md) before continuing experimental work.

For the paper experiment program, start with the
[experimental roadmap](docs/experimental_roadmap.md): verified run status,
planned Tile/WSI foundation models, EAGLE coverage and biomedical experiments.
The [scientific protocols](docs/experimental_protocols.md) and
[execution/implementation runbook](docs/experimental_runbook.md) distinguish
available commands from work that still needs implementation.

## Script layout

`eaf.py` is the supported operational CLI. Specialized entry points are grouped
by purpose: `data/` for acquisition and manifests, `features/` for preprocessing
and feature stores, `training/` for fitting and distillation, `evaluation/` for
benchmarks, and `analysis/` for offline inspection.

## Tests

```bash
pytest
```

Use focused tests while iterating; optional heavy packages are skipped when
unavailable. Runtime artifacts must never be committed.
