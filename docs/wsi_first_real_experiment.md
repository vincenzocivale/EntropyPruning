# WSI First Real Experiment (`exp001`)

This runbook documents the first real paired-store WSI experiment for the
tile-importance pipeline:

```text
early tile features
  -> WSI tile-importance forecaster
  -> predicted tile importance
  -> pruning top-k
  -> materialized late/pruned feature store
  -> downstream WSI evaluation
```

The forecaster predicts one scalar per tile: `tile_importance_target`. For
storage compatibility that value is physically written into the HDF5
`attention [N]` field, but semantically it is a generic tile-importance
target, not a transformer self-attention matrix.

Use `/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python` for validation
and runs in this repository.

---

## 1. Inputs required for `exp001`

You need:

```text
1. Early feature store        (cheap features used as forecaster input)
2. Late feature store         (expensive features materialized after pruning)
3. Tile-importance target     (one scalar per tile, aligned by slide/coords)
4. Split directory            (train.txt, val.txt, test.txt)
```

Expected store properties:

```text
- same slide ids across paired stores
- same tile coordinates between aligned slides when using --alignment-mode coords
- early store feature dim matches --input-feature-dim
- target store contains per-tile attention [N] / tile_importance_target
```

Recommended alignment mode for independently produced stores:

```text
--alignment-mode coords
```

Use `index` only when both stores were created by the same code path and tile
order is guaranteed identical.

---

## 2. Recommended first real experiment: Mode A

Mode A is the controlled first run and should be preferred before using a
WSI foundation model target.

```text
late/full feature store
  -> train ABMIL teacher
  -> extract ABMIL attention target
early feature store
  + ABMIL attention target
  -> train tile-importance forecaster
  -> evaluate pruning
  -> create pruned late feature store
```

### Step 1. Train an ABMIL teacher on the full late store

Use the existing legacy WSI ABMIL pipeline on the full-resolution or
late-layer store intended for downstream use.

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/train_wsi_abmil.py \
  --feature-store data/features_late.h5 \
  --output-dir checkpoints/exp001_abmil \
  --feature-dim 1024 \
  --epochs 20 \
  --batch-size 4 \
  --split-dir splits/exp001 \
  --device cuda
```

### Step 2. Extract the ABMIL tile-importance target

This produces a target store with the same tile features as the late store
plus per-tile scalar attention in the HDF5 `attention` field.

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/extract_wsi_abmil_attention.py \
  --input-feature-store data/features_late.h5 \
  --output-feature-store data/features_abmil_attention.h5 \
  --abmil-checkpoint checkpoints/exp001_abmil/best_abmil_classifier.pt \
  --device cuda \
  --overwrite
```

### Step 3. Validate early/input and target paired stores

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/validate_wsi_paired_feature_stores.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_abmil_attention.h5 \
  --input-feature-dim 384 \
  --alignment-mode coords \
  --require-coords \
  --require-attention
```

### Step 4. Train the tile-importance forecaster

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/train_wsi_importance_forecaster.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_abmil_attention.h5 \
  --output-dir checkpoints/exp001_importance_forecaster \
  --input-feature-dim 384 \
  --hidden-dim 256 \
  --n-heads 4 \
  --n-layers 2 \
  --dropout 0.1 \
  --loss kl \
  --top-k 10 \
  --epochs 20 \
  --batch-size 4 \
  --split-dir splits/exp001 \
  --alignment-mode coords \
  --device cuda
```

Primary output:

```text
checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt
```

### Step 5. Evaluate pruning quality

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/evaluate_wsi_importance_pruning.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_abmil_attention.h5 \
  --forecaster-checkpoint checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt \
  --slide-ids-file splits/exp001/test.txt \
  --keep-ratios 0.05 0.10 0.25 0.50 1.0 \
  --alignment-mode coords \
  --output-csv results/exp001_importance_pruning.csv \
  --device cuda
```

Primary output:

```text
results/exp001_importance_pruning.csv
```

### Step 6. Materialize the pruned late feature store

This is the main artifact needed by downstream WSI evaluation.

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/create_pruned_wsi_feature_store.py \
  --selection-feature-store data/features_layer2.h5 \
  --materialize-feature-store data/features_late.h5 \
  --output-feature-store data/features_late_pruned_keep_0.10.h5 \
  --forecaster-checkpoint checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt \
  --keep-ratio 0.10 \
  --alignment-mode coords \
  --device cuda
```

Primary output:

```text
data/features_late_pruned_keep_0.10.h5
```

### Step 7. Run downstream WSI evaluation

Use the pruned late store with the downstream WSI model or classifier you
already use on full late features. This repository does not define a new
downstream evaluation CLI for `exp001`; it prepares the pruned store needed
by that existing evaluation path.

---

## 3. Alternate real experiment: Mode B

Mode B is for precomputed external tile-importance targets, including
WSI-FM or TRIDENT-produced artifacts.

```text
TRIDENT or other backend
  -> early feature store
  -> late feature store
  -> precomputed tile importance target
EAF
  -> import target
  -> validate paired stores
  -> train importance forecaster
  -> evaluate pruning
  -> materialize pruned late store
```

Important constraint:

```text
TRIDENT is optional.
EAF consumes artifacts that are already on disk.
```

EAF does not require a live TRIDENT installation for this pipeline. It only
consumes HDF5 / `.pt` / `.pth` / `.npy` / `.npz` artifacts that were
precomputed elsewhere.

### Step 1. Build and import the target manifest

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/build_wsi_importance_manifest.py \
  --targets-dir external/tile_importance \
  --coords-dir external/coords \
  --labels-csv labels.csv \
  --output-manifest manifests/exp001_importance.csv \
  --target-source gigapath_wsi_fm \
  --require-coords

/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/import_wsi_importance_targets.py \
  --manifest manifests/exp001_importance.csv \
  --output-feature-store data/features_wsi_importance.h5 \
  --require-coords \
  --overwrite
```

### Step 2. Validate paired stores

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/validate_wsi_paired_feature_stores.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_wsi_importance.h5 \
  --input-feature-dim 384 \
  --alignment-mode coords \
  --require-coords \
  --require-attention
```

### Step 3. Train, evaluate, materialize

Use the same three commands from Mode A, replacing the target store with
`data/features_wsi_importance.h5`.

---

## 4. Minimal `exp001` command set

If the feature stores and `splits/exp001` already exist, the minimum paired
run is:

```bash
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/validate_wsi_paired_feature_stores.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_abmil_attention.h5 \
  --input-feature-dim 384 \
  --alignment-mode coords \
  --require-coords \
  --require-attention

/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/train_wsi_importance_forecaster.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_abmil_attention.h5 \
  --output-dir checkpoints/exp001_importance_forecaster \
  --input-feature-dim 384 \
  --hidden-dim 256 \
  --n-heads 4 \
  --n-layers 2 \
  --dropout 0.1 \
  --loss kl \
  --top-k 10 \
  --epochs 20 \
  --batch-size 4 \
  --split-dir splits/exp001 \
  --alignment-mode coords \
  --device cuda

/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/evaluate_wsi_importance_pruning.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_abmil_attention.h5 \
  --forecaster-checkpoint checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt \
  --slide-ids-file splits/exp001/test.txt \
  --keep-ratios 0.05 0.10 0.25 0.50 1.0 \
  --alignment-mode coords \
  --output-csv results/exp001_importance_pruning.csv \
  --device cuda

/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python scripts/create_pruned_wsi_feature_store.py \
  --selection-feature-store data/features_layer2.h5 \
  --materialize-feature-store data/features_late.h5 \
  --output-feature-store data/features_late_pruned_keep_0.10.h5 \
  --forecaster-checkpoint checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt \
  --keep-ratio 0.10 \
  --alignment-mode coords \
  --device cuda
```

---

## 5. Expected outputs

```text
checkpoints/exp001_importance_forecaster/
results/exp001_importance_pruning.csv
data/features_late_pruned_keep_0.10.h5
```

Optional supporting outputs:

```text
reports/...
logs/...
```

These are runtime artifacts and should not be committed.

---

## 6. Common failure modes

```text
- slide ids differ between stores
- coords missing in one of the two paired stores
- duplicate coords in a slide
- target store imported with the wrong target key
- input feature dim does not match --input-feature-dim
- split dir references slide ids absent from the paired stores
```

Validate before training, especially when stores were generated by different
backends or jobs.
