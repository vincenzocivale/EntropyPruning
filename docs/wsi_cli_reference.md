# WSI CLI Reference

This document lists the WSI-related command line tools in `scripts/`.

---

## 1. Import and feature-store creation

### `build_trident_manifest.py`

Builds a CSV manifest from TRIDENT-style per-slide HDF5 feature files.

Output manifest columns:

```text
slide_id,features_path,coords_path,label
```

Example:

```bash
python scripts/build_trident_manifest.py \
  --features-dir trident_processed/20x_256px_0px_overlap/features_uni_v1 \
  --coords-dir trident_processed/20x_256px_0px_overlap/patches \
  --labels-csv labels.csv \
  --output-manifest manifests/trident.csv \
  --require-coords \
  --require-labels
```

Common options:

```text
--feature-glob
--require-coords
--require-labels
--absolute-paths
--overwrite
```

---

### `import_trident_feature_store.py`

Imports a TRIDENT-style manifest into EAF `H5WSIFeatureStore`.

```bash
python scripts/import_trident_feature_store.py \
  --manifest manifests/trident.csv \
  --output-feature-store data/features_trident_eaf.h5 \
  --feature-dim 1024
```

Common options:

```text
--feature-dataset
--coords-dataset
--overwrite
```

---

### `build_generic_feature_manifest.py`

Builds a manifest from generic per-slide `.pt`, `.pth`, `.npy`, or `.npz` files.

```bash
python scripts/build_generic_feature_manifest.py \
  --features-dir features \
  --coords-dir coords \
  --labels-csv labels.csv \
  --output-manifest manifests/generic.csv \
  --require-coords \
  --require-labels
```

Common options:

```text
--feature-glob
--require-coords
--require-labels
--absolute-paths
--overwrite
```

---

### `import_generic_feature_store.py`

Imports generic `.pt`, `.pth`, `.npy`, or `.npz` features into EAF HDF5.

```bash
python scripts/import_generic_feature_store.py \
  --manifest manifests/generic.csv \
  --output-feature-store data/features_generic_eaf.h5 \
  --feature-dim 1024
```

Common options:

```text
--feature-key
--coords-key
--overwrite
```

Use `--feature-key` and `--coords-key` when `.pt` or `.npz` files contain multiple arrays.

---

### `create_synthetic_wsi_feature_store.py`

Creates a synthetic WSI feature store for tests and smoke runs.

```bash
python scripts/create_synthetic_wsi_feature_store.py \
  --output-feature-store /tmp/features_raw.h5 \
  --n-slides 8 \
  --feature-dim 16 \
  --min-tiles 4 \
  --max-tiles 8 \
  --n-classes 2 \
  --seed 0
```

---

### `create_pruned_wsi_feature_store.py`

Creates a feature store containing only the top predicted tiles per slide.

```bash
python scripts/create_pruned_wsi_feature_store.py \
  --input-feature-store data/features_real_abmil_attention.h5 \
  --output-feature-store data/features_real_pruned_keep_0.10.h5 \
  --forecaster-checkpoint checkpoints/exp001_forecaster/best_wsi_tile_attention_forecaster.pt \
  --keep-ratio 0.10 \
  --device cuda
```

---

## 2. Inspection, validation, and splitting

### `validate_wsi_feature_store.py`

Validates feature-store integrity.

```bash
python scripts/validate_wsi_feature_store.py \
  --feature-store data/features_real.h5 \
  --feature-dim 1024 \
  --require-coords
```

For attention stores:

```bash
python scripts/validate_wsi_feature_store.py \
  --feature-store data/features_real_abmil_attention.h5 \
  --feature-dim 1024 \
  --require-coords \
  --require-attention
```

---

### `inspect_wsi_feature_store.py`

Prints descriptive statistics for an EAF WSI feature store.

```bash
python scripts/inspect_wsi_feature_store.py \
  --feature-store data/features_real.h5 \
  --output-json reports/features_real_inspection.json
```

Reported groups:

```text
tile_count
feature_dims
coords
attention
labels
metadata
quality_flags
examples
```

---

### `split_wsi_feature_store.py`

Creates reproducible train/val/test split files.

```bash
python scripts/split_wsi_feature_store.py \
  --feature-store data/features_real.h5 \
  --output-dir splits/exp001 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --stratify-label \
  --seed 0
```

Outputs:

```text
train.txt
val.txt
test.txt
split_summary.json
```

Training CLIs consume `train.txt` and `val.txt` through `--split-dir`.

Evaluation CLIs should consume `test.txt` through `--slide-ids-file`.

---

## 3. Teacher model

### `train_wsi_abmil.py`

Trains the ABMIL WSI teacher.

```bash
python scripts/train_wsi_abmil.py \
  --feature-store data/features_real.h5 \
  --output-dir checkpoints/exp001_abmil \
  --feature-dim 1024 \
  --hidden-dim 256 \
  --n-classes 2 \
  --epochs 20 \
  --batch-size 4 \
  --device cuda \
  --split-dir splits/exp001
```

Important outputs:

```text
best_abmil_classifier.pt
training_summary.json
```

The training CLI supports either:

```text
--split-dir splits/exp001
```

or explicit files:

```text
--train-slide-ids-file splits/exp001/train.txt
--val-slide-ids-file splits/exp001/val.txt
```

Do not pass both forms at the same time.

---

### `extract_wsi_abmil_attention.py`

Extracts ABMIL tile attention into a new feature store.

```bash
python scripts/extract_wsi_abmil_attention.py \
  --input-feature-store data/features_real.h5 \
  --output-feature-store data/features_real_abmil_attention.h5 \
  --abmil-checkpoint checkpoints/exp001_abmil/best_abmil_classifier.pt \
  --batch-size 4 \
  --device cuda
```

The output store is used to train the attention forecaster.

---

## 4. Attention forecaster

### `train_wsi_attention_forecaster.py`

Trains the WSI tile attention forecaster.

```bash
python scripts/train_wsi_attention_forecaster.py \
  --feature-store data/features_real_abmil_attention.h5 \
  --output-dir checkpoints/exp001_forecaster \
  --feature-dim 1024 \
  --hidden-dim 256 \
  --n-heads 4 \
  --n-layers 2 \
  --dropout 0.1 \
  --epochs 20 \
  --batch-size 4 \
  --top-k 10 \
  --device cuda \
  --split-dir splits/exp001
```

Important outputs:

```text
best_wsi_tile_attention_forecaster.pt
training_summary.json
```

The training CLI supports either:

```text
--split-dir splits/exp001
```

or explicit files:

```text
--train-slide-ids-file splits/exp001/train.txt
--val-slide-ids-file splits/exp001/val.txt
```

Do not pass both forms at the same time.

---

## 5. Evaluation

### `evaluate_wsi_forecaster_pruning.py`

Evaluates attention-level pruning quality.

```bash
python scripts/evaluate_wsi_forecaster_pruning.py \
  --feature-store data/features_real_abmil_attention.h5 \
  --forecaster-checkpoint checkpoints/exp001_forecaster/best_wsi_tile_attention_forecaster.pt \
  --slide-ids-file splits/exp001/test.txt \
  --keep-ratios 0.05 0.1 0.25 0.5 1.0 \
  --output-csv results/exp001_forecaster_pruning.csv \
  --device cuda
```

Main metrics:

```text
spearman
top-k overlap
ndcg
attention mass retained
oracle attention mass
relative attention mass retained
```

---

### `evaluate_wsi_abmil_pruning_agreement.py`

Evaluates prediction preservation under pruning.

```bash
python scripts/evaluate_wsi_abmil_pruning_agreement.py \
  --feature-store data/features_real_abmil_attention.h5 \
  --abmil-checkpoint checkpoints/exp001_abmil/best_abmil_classifier.pt \
  --forecaster-checkpoint checkpoints/exp001_forecaster/best_wsi_tile_attention_forecaster.pt \
  --slide-ids-file splits/exp001/test.txt \
  --keep-ratios 0.05 0.1 0.25 0.5 1.0 \
  --output-csv results/exp001_abmil_pruning_agreement.csv \
  --device cuda
```

Main metrics:

```text
full_accuracy
pruned_accuracy
prediction_agreement
mean_logit_cosine_similarity
mean_prob_kl_full_to_pruned
attention_mass_retained
```

---

## 6. Reporting

### `plot_wsi_pruning_curves.py`

Plots pruning curves and writes a summary JSON.

```bash
python scripts/plot_wsi_pruning_curves.py \
  --forecaster-pruning-csv results/exp001_forecaster_pruning.csv \
  --abmil-agreement-csv results/exp001_abmil_pruning_agreement.csv \
  --output-dir reports/exp001_pruning \
  --overwrite
```

Outputs:

```text
*.png
summary.json
```

---

## 7. Smoke scripts

### `run_trident_import_smoke.sh`

Synthetic TRIDENT-style import smoke:

```bash
bash scripts/run_trident_import_smoke.sh
```

Checks:

```text
fake TRIDENT-like HDF5 files
build_trident_manifest.py
import_trident_feature_store.py
validate_wsi_feature_store.py
inspect_wsi_feature_store.py
```

---

### `run_generic_import_smoke.sh`

Synthetic generic import smoke:

```bash
bash scripts/run_generic_import_smoke.sh
```

Checks:

```text
fake .pt/.npy/.npz features
build_generic_feature_manifest.py
import_generic_feature_store.py
validate_wsi_feature_store.py
inspect_wsi_feature_store.py
```

---

### `run_wsi_synthetic_e2e_smoke.sh`

Full synthetic WSI pipeline smoke:

```bash
bash scripts/run_wsi_synthetic_e2e_smoke.sh
```

Checks:

```text
create_synthetic_wsi_feature_store.py
validate_wsi_feature_store.py
split_wsi_feature_store.py
train_wsi_abmil.py
extract_wsi_abmil_attention.py
train_wsi_attention_forecaster.py
evaluate_wsi_forecaster_pruning.py
evaluate_wsi_abmil_pruning_agreement.py
plot_wsi_pruning_curves.py
create_pruned_wsi_feature_store.py
```

Useful environment variables:

```text
WORKDIR
FEATURE_DIM
HIDDEN_DIM
N_SLIDES
MIN_TILES
MAX_TILES
N_CLASSES
BATCH_SIZE
ABMIL_EPOCHS
FORECASTER_EPOCHS
KEEP_RATIOS
PRUNED_KEEP_RATIO
DEVICE
SEED
```

Example:

```bash
WORKDIR=/tmp/eaf_wsi_smoke \
FEATURE_DIM=8 \
N_SLIDES=8 \
ABMIL_EPOCHS=1 \
FORECASTER_EPOCHS=1 \
DEVICE=cpu \
bash scripts/run_wsi_synthetic_e2e_smoke.sh
```

---

## 8. Legacy tile-level scripts

The following scripts are not WSI-specific and should not be deleted without checking the original EAF tile-level pipeline:

```text
train_classifier.py
train_forecaster.py
finetune_pruned.py
```

They likely correspond to the original EAF stages:

```text
tile classifier
tile forecaster
fine-tuning on pruned tiles
```

The `ablations/` directory should also be preserved unless its tests and references are removed.

---

## 9. Safe cleanup

Safe to remove from `scripts/`:

```text
scripts/__pycache__/
```

Command:

```bash
rm -rf scripts/__pycache__
```

Do not commit generated runtime files such as:

```text
*.h5
checkpoints/
results/
reports/
logs/
```

unless the repository intentionally tracks test fixtures.
