# EAF pipeline

## 1. Prepare data

Use `python scripts/eaf.py data ...` to register or acquire sources and build a
strict pretraining manifest. TRIDENT supplies segmentation and coordinates:

```bash
python scripts/wsi_preprocess.py \
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
scripts/train_wsi_tile_eaf_online.py              Tile-EAF forecaster
scripts/finetune_wsi_tile_encoder_pruned_online.py tile encoder distillation
scripts/wsi_eaf_infer_wsi_fm.py                   WSI-FM output cache
scripts/train_wsi_landmark_forecaster.py          WSI-EAF forecaster
scripts/finetune_wsi_titan_pruned.py              WSI distillation
scripts/eval_wsi_linear_probing.py                paired downstream evaluation
```

Each new run writes a compact summary below `$EAF_WSI_ROOT/results/`, alongside
its metrics and timing metadata. Checkpoints remain under `checkpoints/`.

## 4. Inspect state and performance

```bash
python scripts/eaf.py experiments audit --data-root "$EAF_WSI_ROOT"
```

The audit writes `results/experiment_catalog/catalog.json`. It reports complete,
partial and non-comparable runs from verified artifacts. Before an optimization
or model sweep, capture a profile with the same data and parameters, change one
factor, then profile again.

## Current scope

The supported path is offline frozen-teacher caching followed by online early
feature extraction during Tile-EAF training. HDF5 remains an input compatibility
format; new EAF output is NumPy. Historical queue wrappers and the older UNI
ablation were removed because their behavior was tied to machine-specific paths
or deleted training code.
