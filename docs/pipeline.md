# EAF pipeline

For the full paper program, use the [experimental roadmap](experimental_roadmap.md),
[scientific protocols](experimental_protocols.md), and
[execution/implementation runbook](experimental_runbook.md). Those documents
separate existing entry points from planned external, survival and biomedical
evaluation capabilities; the commands below are not yet the complete paper runner.

## 1. Prepare data

Use `python scripts/eaf.py data ...` to register or acquire sources and build a
strict pretraining manifest. TRIDENT supplies segmentation and coordinates:

```bash
python scripts/features/wsi_preprocess.py \
  --trident-repo /path/to/TRIDENT --wsi-dir /path/to/wsis --job-dir /path/to/job
```

The wrapper converts TRIDENT's derived outputs to `.npyd` after a successful
run. Raw WSI are never copied or converted.

## 2. Build frozen Tile-EAF targets

```bash
python scripts/eaf.py cache tile \
  --data-root "$EAF_WSI_ROOT" \
  --manifest datasets/pretraining/eaf_wsi_pretrain_strict_v1/manifests/slides.csv \
  --output-dir "$EAF_WSI_ROOT/caches/tile_eaf/<dataset>/<encoder>/<cache-id>" \
  --encoder conch_v15 --dataset <dataset>
```

Each slide stores `coords`, final attention and final tile embeddings. The early
layer is recomputed online during training; it is intentionally not persisted.
The cache writer publishes only complete, validated slides.

## 3. Train and evaluate

Use the explicit scripts for the three learning stages:

```text
scripts/training/train_wsi_tile_eaf_online.py              Tile-EAF forecaster
scripts/training/finetune_wsi_tile_encoder_pruned_online.py tile encoder distillation
scripts/training/train_multi_thunder_classifier.py         Tile-FM paired evaluation (THUNDER linear probing)
scripts/features/wsi_eaf_infer_wsi_fm.py                   WSI-FM output cache
scripts/training/train_wsi_landmark_forecaster.py          WSI-EAF forecaster
scripts/training/finetune_wsi_titan_pruned.py              WSI distillation
scripts/evaluation/eval_wsi_linear_probing.py                paired downstream evaluation
```

Each new run writes a compact summary below `$EAF_WSI_ROOT/results/`, alongside
its metrics and timing metadata. Checkpoints remain under `checkpoints/`.

### Tile-FM vs WSI-FM evaluation: not the same protocol or the same tool

The two levels are evaluated on different benchmark families and must not be
substituted for one another:

- **Tile-FM** (e.g. CONCH v1.5 Tile-EAF): linear probing on the **THUNDER**
  tile-classification datasets, via `train_multi_thunder_classifier.py
  --pruned-adapter-ckpt <best_*_adapter.pt> --adaptation linear_probing`
  (baseline: omit `--pruned-adapter-ckpt`). `--base-data-folder` must point at
  the directory that directly contains `data_splits/` and `datasets/` — on
  this host that is `$THUNDER_BASE_DATA_FOLDER/datasets`, not
  `$THUNDER_BASE_DATA_FOLDER` itself. 15 datasets are available locally (bach,
  bracs, break_his, ccrcc, crc, esca, patch_camelyon, spider_breast/
  colorectal/skin/thorax, tcga_crc_msi, tcga_tils, tcga_uniform, wilds); the
  script holds out the N smallest as out-of-distribution generalization tasks.
  A WSI-level mean-pooled downstream task is NOT a substitute for this.
- **WSI-FM** (e.g. TITAN WSI-EAF): linear probing on the labeled datasets used
  by the **EAGLE** study, via `eval_wsi_linear_probing.py` against the TCGA
  cohort label CSVs under `datasets/downstream/wsi_level/<cohort>/labels/`
  (`--pruned-checkpoint` for the WSI-EAF arm, baseline is always included).

Neither script existed as a gap to fill — both were already implemented before
either evaluation had actually been run end-to-end; check
`checkpoints/multi_thunder/` and `results/wsi_eaf/evaluation/` for prior runs
before adding a new ad hoc evaluator.

## 4. Inspect state and performance

```bash
python scripts/eaf.py experiments audit --data-root "$EAF_WSI_ROOT"
```

The audit writes `results/experiment_catalog/catalog.json`. It reports complete,
partial and non-comparable runs from verified artifacts. Before an optimization
or model sweep, capture a profile with the same data and parameters, change one
factor, then profile again.

[Project status and handoff](continuity.md) records the verified experiment inventory and the next work sequence.

## Current scope

The supported path is offline frozen-teacher caching followed by online early
feature extraction during Tile-EAF training. HDF5 remains an input compatibility
format; new EAF output is NumPy. Historical queue wrappers and the older UNI
ablation were removed because their behavior was tied to machine-specific paths
or deleted training code.
