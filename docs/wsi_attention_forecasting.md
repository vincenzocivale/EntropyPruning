# WSI Tile Attention Forecasting

This document describes the WSI feature-level extension of EAF.

The goal is to predict which tiles of a whole-slide image are likely to be important for a downstream WSI classifier, then prune low-value tiles before later computation.

The current implementation starts from precomputed tile features. It does not extract tiles or encoder features directly from raw `.svs`, `.ndpi`, `.tiff`, or equivalent WSI files.

---

## 1. Pipeline overview

The implemented WSI pipeline is:

```text
precomputed tile features
  -> EAF HDF5 WSI feature store
  -> inspect / validate
  -> persistent train/val/test split
  -> train ABMIL teacher
  -> extract ABMIL tile attention
  -> train tile attention forecaster
  -> evaluate pruning quality
  -> evaluate ABMIL full-vs-pruned agreement
  -> plot pruning curves
  -> create pruned feature store
```

The central idea is:

```text
ABMIL teacher attention = target tile importance
forecaster prediction   = learned proxy for target attention
top-k predicted tiles   = retained tiles after pruning
```

The implementation currently validates the mechanism at feature level. True compute savings depend on future experiments where the forecaster uses cheaper or earlier features than the downstream WSI classifier.

---

## 2. Core data contract

The canonical internal WSI object is `WSIBag`.

```text
WSIBag
  slide_id: str
  tile_features: Tensor [n_tiles, feature_dim]
  coords: optional Tensor [n_tiles, 2] or [n_tiles, 4]
  label: optional int / float / Tensor
  attention: optional Tensor [n_tiles]
  metadata: optional dict
```

The canonical feature store is `H5WSIFeatureStore`.

A feature store for ABMIL teacher training needs:

```text
tile_features
label
```

A feature store for attention forecaster training needs:

```text
tile_features
attention
```

Coordinates are optional for model training but recommended for traceability, visualization, and downstream slide-level analysis.

---

## 3. Recommended experiment layout

```text
data/
  features_real.h5
  features_real_abmil_attention.h5
  features_real_pruned_keep_0.10.h5

manifests/
  trident.csv
  generic.csv

splits/
  exp001/
    train.txt
    val.txt
    test.txt
    split_summary.json

checkpoints/
  exp001_abmil/
    best_abmil_classifier.pt
    training_summary.json
  exp001_forecaster/
    best_wsi_tile_attention_forecaster.pt
    training_summary.json

results/
  exp001_forecaster_pruning.csv
  exp001_abmil_pruning_agreement.csv

reports/
  features_real_inspection.json
  exp001_pruning/
    summary.json
    *.png
```

---

## 4. Importing feature stores

There are two supported import paths.

### 4.1 TRIDENT-style HDF5 features

Use this path when features are stored as per-slide HDF5 files, for example:

```text
trident_processed/
  20x_256px_0px_overlap/
    features_uni_v1/
      slide_001.h5
      slide_002.h5
    patches/
      slide_001.h5
      slide_002.h5
```

Build a manifest:

```bash
python scripts/build_trident_manifest.py \
  --features-dir trident_processed/20x_256px_0px_overlap/features_uni_v1 \
  --coords-dir trident_processed/20x_256px_0px_overlap/patches \
  --labels-csv labels.csv \
  --output-manifest manifests/trident.csv \
  --require-coords \
  --require-labels
```

Import it:

```bash
python scripts/import_trident_feature_store.py \
  --manifest manifests/trident.csv \
  --output-feature-store data/features_trident_eaf.h5 \
  --feature-dim 1024
```

If the HDF5 dataset names differ from defaults, pass explicit names:

```bash
python scripts/import_trident_feature_store.py \
  --manifest manifests/trident.csv \
  --output-feature-store data/features_trident_eaf.h5 \
  --feature-dim 1024 \
  --feature-dataset features \
  --coords-dataset coords
```

### 4.2 Generic `.pt`, `.pth`, `.npy`, `.npz` features

Use this path when each slide already has a feature file:

```text
features/
  slide_001.pt
  slide_002.npy
  slide_003.npz

coords/
  slide_001.pt
  slide_002.npy
  slide_003.npz
```

Build a manifest:

```bash
python scripts/build_generic_feature_manifest.py \
  --features-dir features \
  --coords-dir coords \
  --labels-csv labels.csv \
  --output-manifest manifests/generic.csv \
  --require-coords \
  --require-labels
```

Import it:

```bash
python scripts/import_generic_feature_store.py \
  --manifest manifests/generic.csv \
  --output-feature-store data/features_generic_eaf.h5 \
  --feature-dim 1024
```

For `.pt` or `.npz` files containing multiple arrays, pass explicit keys:

```bash
python scripts/import_generic_feature_store.py \
  --manifest manifests/generic.csv \
  --output-feature-store data/features_generic_eaf.h5 \
  --feature-dim 1024 \
  --feature-key features \
  --coords-key coords
```

---

## 5. Inspecting and validating feature stores

Before training, inspect the imported store:

```bash
python scripts/inspect_wsi_feature_store.py \
  --feature-store data/features_real.h5 \
  --output-json reports/features_real_inspection.json
```

The inspection report includes:

```text
n_slides
feature_dim distribution
tile count min/mean/median/max
coords coverage
attention coverage
label coverage
label distribution
metadata source distribution
quality flags
```

Then validate the store:

```bash
python scripts/validate_wsi_feature_store.py \
  --feature-store data/features_real.h5 \
  --feature-dim 1024 \
  --require-coords
```

For a store that should already contain teacher attention:

```bash
python scripts/validate_wsi_feature_store.py \
  --feature-store data/features_real_abmil_attention.h5 \
  --feature-dim 1024 \
  --require-coords \
  --require-attention
```

---

## 6. Creating persistent train/val/test splits

Create a reproducible split:

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

Output:

```text
splits/exp001/train.txt
splits/exp001/val.txt
splits/exp001/test.txt
splits/exp001/split_summary.json
```

Training scripts use `train.txt` and `val.txt`. Evaluation scripts should use `test.txt`.

The `--split-dir` argument in training CLIs is equivalent to:

```text
--train-slide-ids-file splits/exp001/train.txt
--val-slide-ids-file splits/exp001/val.txt
```

---

## 7. Training the ABMIL teacher

Train the teacher on the raw feature store:

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
checkpoints/exp001_abmil/best_abmil_classifier.pt
checkpoints/exp001_abmil/training_summary.json
```

The ABMIL teacher is used for two purposes:

```text
1. baseline WSI classifier
2. source of tile attention targets
```

---

## 8. Extracting ABMIL attention

After teacher training, extract attention into a new feature store:

```bash
python scripts/extract_wsi_abmil_attention.py \
  --input-feature-store data/features_real.h5 \
  --output-feature-store data/features_real_abmil_attention.h5 \
  --abmil-checkpoint checkpoints/exp001_abmil/best_abmil_classifier.pt \
  --batch-size 4 \
  --device cuda
```

The output store has the same slide-level bags, plus:

```text
attention [n_tiles]
```

for each slide.

---

## 9. Training the tile attention forecaster

Train the forecaster to predict ABMIL attention from tile features:

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
checkpoints/exp001_forecaster/best_wsi_tile_attention_forecaster.pt
checkpoints/exp001_forecaster/training_summary.json
```

The training objective is KL divergence between teacher attention and predicted attention. Attention targets are normalized over valid tiles.

---

## 10. Evaluating pruning quality

### 10.1 Attention-level pruning evaluation

This measures how well the forecaster selects high-attention tiles.

```bash
python scripts/evaluate_wsi_forecaster_pruning.py \
  --feature-store data/features_real_abmil_attention.h5 \
  --forecaster-checkpoint checkpoints/exp001_forecaster/best_wsi_tile_attention_forecaster.pt \
  --slide-ids-file splits/exp001/test.txt \
  --keep-ratios 0.05 0.1 0.25 0.5 1.0 \
  --output-csv results/exp001_forecaster_pruning.csv \
  --device cuda
```

Metrics include:

```text
Spearman correlation
top-k overlap
NDCG@k
attention mass retained
oracle attention mass
relative attention mass retained
```

### 10.2 ABMIL full-vs-pruned agreement

This measures whether ABMIL predictions are preserved after pruning.

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

Metrics include:

```text
full_accuracy
pruned_accuracy
prediction_agreement
mean_logit_cosine_similarity
mean_prob_kl_full_to_pruned
attention_mass_retained
```

---

## 11. Plotting pruning curves

```bash
python scripts/plot_wsi_pruning_curves.py \
  --forecaster-pruning-csv results/exp001_forecaster_pruning.csv \
  --abmil-agreement-csv results/exp001_abmil_pruning_agreement.csv \
  --output-dir reports/exp001_pruning \
  --overwrite
```

Outputs:

```text
reports/exp001_pruning/*.png
reports/exp001_pruning/summary.json
```

---

## 12. Creating a pruned feature store

Create a new HDF5 store containing only top predicted tiles:

```bash
python scripts/create_pruned_wsi_feature_store.py \
  --input-feature-store data/features_real_abmil_attention.h5 \
  --output-feature-store data/features_real_pruned_keep_0.10.h5 \
  --forecaster-checkpoint checkpoints/exp001_forecaster/best_wsi_tile_attention_forecaster.pt \
  --keep-ratio 0.10 \
  --device cuda
```

Selected tile indices are sorted back into original slide order before writing. The output metadata records pruning details.

The pruned store can be used as input to downstream WSI training:

```bash
python scripts/train_wsi_abmil.py \
  --feature-store data/features_real_pruned_keep_0.10.h5 \
  --output-dir checkpoints/exp001_abmil_pruned_keep_0.10 \
  --feature-dim 1024 \
  --hidden-dim 256 \
  --n-classes 2 \
  --epochs 20 \
  --batch-size 4 \
  --device cuda \
  --split-dir splits/exp001
```

---

## 13. Synthetic smoke pipeline

For a complete local sanity check:

```bash
bash scripts/run_wsi_synthetic_e2e_smoke.sh
```

This runs:

```text
synthetic feature store creation
validation
persistent splitting
ABMIL teacher training
ABMIL attention extraction
forecaster training
pruning evaluation
agreement evaluation
plotting
pruned feature store creation
```

Environment variables can override defaults:

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

## 14. Recommended first real experiment: `exp001`

A recommended first real experiment:

```text
exp001
  data/features_real.h5
  reports/features_real_inspection.json
  splits/exp001
  checkpoints/exp001_abmil
  data/features_real_abmil_attention.h5
  checkpoints/exp001_forecaster
  results/exp001_forecaster_pruning.csv
  results/exp001_abmil_pruning_agreement.csv
  reports/exp001_pruning
  data/features_real_pruned_keep_0.10.h5
```

Recommended first keep ratios:

```text
0.05
0.10
0.25
0.50
1.00
```

Do not tune on the test split. Use `train.txt` and `val.txt` for model selection. Reserve `test.txt` for final pruning evaluation.

---

## 15. Current limitations

The current WSI implementation is feature-level only.

Not yet implemented:

```text
raw WSI tiling
raw image encoder inference
multi-resolution feature extraction
CLAM / DSMIL / TransMIL teacher wrappers
formal early-feature vs late-feature dual-store contract
full experiment runner
full report generator
```

The current teacher is ABMIL. Results should be interpreted as ABMIL-teacher-specific.

For true EAF-style savings, future experiments should distinguish:

```text
cheap / early features used by the forecaster
expensive / late features used by the teacher
```

The current implementation can still validate the main pruning mechanism with a single feature representation.
