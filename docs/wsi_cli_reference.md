# WSI CLI Reference

This document lists the WSI-related command line tools in `scripts/`. See
`docs/wsi_tile_importance_forecasting.md` for the paired-store tile-importance
pipeline this reference documents alongside the legacy single-store ABMIL
pipeline (`docs/wsi_attention_forecasting.md`).

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

### `build_wsi_importance_manifest.py`

Builds a manifest for importing precomputed tile-importance targets (ABMIL
attention exported elsewhere, a WSI foundation model tile score, or any other
precomputed per-tile value) from a directory of per-slide target files.
Target files may be TRIDENT-style per-slide HDF5, or generic `.pt`, `.pth`,
`.npy`, `.npz` — this script (and the importer below) never require TRIDENT
to be installed; they only expect TRIDENT-*style* per-slide artifacts.

Output manifest columns:

```text
slide_id,target_path,coords_path,label,target_source,target_type
```

Example:

```bash
python scripts/build_wsi_importance_manifest.py \
  --targets-dir trident_processed/.../tile_importance_gigapath \
  --coords-dir trident_processed/.../patches \
  --labels-csv labels.csv \
  --output-manifest manifests/gigapath_importance.csv \
  --target-glob "*.h5" \
  --target-source gigapath_wsi_fm \
  --require-coords
```

Common options:

```text
--target-glob
--target-source
--target-type
--require-coords
--require-labels
--absolute-paths
--overwrite
```

---

### `import_wsi_importance_targets.py`

Imports a tile-importance manifest into an EAF `H5WSIFeatureStore` usable as
a `--target-feature-store`. Since a target store has no real per-tile
embedding, `tile_features` is set to the target vector reshaped to
`[n_tiles, 1]` — this makes accidentally using a target store as an *input*
feature store fail loudly on a feature-dim mismatch instead of silently
succeeding.

```bash
python scripts/import_wsi_importance_targets.py \
  --manifest manifests/gigapath_importance.csv \
  --output-feature-store data/features_wsi_importance.h5 \
  --target-key attention \
  --coords-key coords \
  --require-coords \
  --overwrite
```

Common options:

```text
--target-key           dataset/array key for the importance value; auto-detected
                        from common names (attention, importance,
                        tile_importance, score, scores) when omitted
--coords-key            dataset/array key for coordinates
--require-coords
--normalize             none | sum | minmax | softmax (default: none)
--created-by
--trident-job-dir
--patch-encoder
--slide-encoder
--mag
--patch-size
--overwrite
```

Every imported bag's metadata records `target_source`, `target_type`,
`target_key`, `coords_source`, `created_by`, `normalization`,
`trident_job_dir`, `patch_encoder`, `slide_encoder`, `mag`, and `patch_size`
(unset fields are stored as `null`). Validate the result with:

```bash
python scripts/validate_wsi_feature_store.py \
  --feature-store data/features_wsi_importance.h5 \
  --require-attention
```

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

### `create_synthetic_wsi_importance_stores.py`

Creates *paired* synthetic early/late/importance-target HDF5 stores in one
deterministic pass, for tests and smoke runs of the tile-importance pipeline
(§4). The importance target is correlated with the late-feature store:
`importance_i = softmax(w^T late_feature_i + noise_i)`. All three stores
share identical slide ids, tile counts, and coordinates by construction —
three independent `create_synthetic_wsi_feature_store.py` calls with
different `--feature-dim` values cannot guarantee this, because the shared
RNG is consumed differently by each dimension and desynchronizes per-slide
tile counts across runs.

```bash
python scripts/create_synthetic_wsi_importance_stores.py \
  --early-output /tmp/features_layer2.h5 \
  --late-output /tmp/features_late.h5 \
  --importance-output /tmp/features_synthetic_importance.h5 \
  --n-slides 8 \
  --early-feature-dim 8 \
  --late-feature-dim 16 \
  --min-tiles 4 \
  --max-tiles 8 \
  --noise-std 0.1 \
  --seed 0
```

Common options:

```text
--n-classes
--noise-std       std of Gaussian noise added before the importance softmax
--seed
--overwrite
```

The importance-target store follows the same convention as
`import_wsi_importance_targets.py`: `tile_features` is the target reshaped
to `[n_tiles, 1]`, so accidentally using it as an *input* feature store
fails loudly on a feature-dim mismatch instead of silently succeeding.

---

### `create_pruned_wsi_feature_store.py`

Creates a feature store containing only the top predicted tiles per slide.

Legacy single-store mode (tiles are selected and materialized from the same
store):

```bash
python scripts/create_pruned_wsi_feature_store.py \
  --input-feature-store data/features_real_abmil_attention.h5 \
  --output-feature-store data/features_real_pruned_keep_0.10.h5 \
  --forecaster-checkpoint checkpoints/exp001_forecaster/best_wsi_tile_attention_forecaster.pt \
  --keep-ratio 0.10 \
  --device cuda
```

Selection/materialize mode: tiles are *selected* using one store (e.g.
early-layer features consumed by the forecaster) and the corresponding tiles
are *materialized* from a different store (e.g. late-layer features for a
downstream WSI model). `--materialize-feature-store` defaults to
`--selection-feature-store` when omitted. `--input-feature-store` is
mutually exclusive with `--selection-feature-store`/`--materialize-feature-store`.

```bash
python scripts/create_pruned_wsi_feature_store.py \
  --selection-feature-store data/features_layer2.h5 \
  --materialize-feature-store data/features_late.h5 \
  --output-feature-store data/features_late_pruned_keep_0.10.h5 \
  --forecaster-checkpoint checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt \
  --keep-ratio 0.10 \
  --alignment-mode coords \
  --device cuda
```

`--alignment-mode` only matters when the materialize store differs from the
selection store: `index` (default) requires identical tile counts (and, if
both sides have coords, identical tile order); `coords` aligns tiles by exact
coordinate match and requires coords on both sides. Selected tile indices are
always sorted back into the selection store's original tile order before
writing.

Pruned-store metadata in selection/materialize mode: `pruned_by`,
`forecaster_checkpoint`, `keep_ratio`, `n_tiles_original`, `n_tiles_kept`,
`selection_feature_store`, `materialize_feature_store`, `alignment_mode`, and
`target_source` when the forecaster checkpoint carries one. The legacy
single-store mode keeps its original metadata keys (`pruned_by`,
`forecaster_checkpoint`, `pruning_keep_ratio`, `pruning_input_n_tiles`,
`pruning_output_n_tiles`, `pruning_preserved_original_tile_order`) unchanged.

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

### `train_wsi_importance_forecaster.py`

Generalized replacement for `train_wsi_attention_forecaster.py`. Predicts a
scalar tile-importance score (`WSITileImportanceForecaster` — a backward
compatible alias, `WSITileAttentionForecaster`, is kept for existing code)
from paired input/target feature stores, so early-layer input features and a
precomputed importance target (ABMIL attention, an imported WSI-FM tile
score, or anything else) can live in independently produced stores.
`train_wsi_attention_forecaster.py` is unaffected and keeps working exactly
as before; it is not rewired to call this script.

```bash
python scripts/train_wsi_importance_forecaster.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_wsi_importance.h5 \
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

A single fused store is also supported via `--feature-store` (sets both
input and target to the same store), mutually exclusive with
`--input-feature-store`/`--target-feature-store`.

Loss options (`--loss`):

```text
kl        KL divergence between predicted and target distributions (default)
mse       MSE between predicted and target distributions
topk_bce  binary cross-entropy treating the target top-k tiles as positives
kl+rank   kl + --rank-weight * a top-k-vs-bottom-k margin ranking loss
```

Other options:

```text
--alignment-mode index|coords   how input/target tiles are paired (see
                                 docs/wsi_paired_feature_store_audit.md)
--require-coords
--target-source                 explicit target provenance label; inferred
                                 from target-store metadata when omitted
--target-smoothing              non-negative value added to the target
                                 before normalization; 0.0 (default) raises
                                 an explicit error on an all-zero target on
                                 a slide, a positive value turns it into a
                                 uniform distribution instead
--rank-weight, --rank-margin    only used by --loss kl+rank
```

Important outputs:

```text
best_wsi_tile_importance_forecaster.pt
training_summary.json
```

The checkpoint metadata includes `model_type`, `input_feature_dim`,
`hidden_dim`, `n_heads`, `n_layers`, `loss`, `target_type`, `target_source`,
`input_feature_store`, `target_feature_store`, `alignment_mode`, plus
`train_slide_ids`/`val_slide_ids`/`top_k`/`seed`/`split_files`.
`training_summary.json` additionally reports `best_val_loss`,
`best_val_metrics` (including top-k overlap/NDCG@k), `target_entropy_mean`,
`target_source`, and `seed`.
`load_wsi_tile_importance_forecaster_checkpoint` reads checkpoints written by
either this script or the legacy `train_wsi_attention_forecaster.py`.

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

### `evaluate_wsi_importance_pruning.py`

Generalized replacement for `evaluate_wsi_forecaster_pruning.py`. Evaluates
tile-importance pruning quality from paired input/target feature stores (an
early-features selection store and an independently produced tile-importance
target store), so pruning can be evaluated even when the target did not come
from the same store as the forecaster's input.
`evaluate_wsi_forecaster_pruning.py` is unaffected and keeps working exactly
as before.

```bash
python scripts/evaluate_wsi_importance_pruning.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_wsi_importance.h5 \
  --forecaster-checkpoint checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt \
  --slide-ids-file splits/exp001/test.txt \
  --keep-ratios 0.05 0.10 0.25 0.50 1.0 \
  --alignment-mode coords \
  --output-csv results/exp001_importance_pruning.csv \
  --device cuda
```

A single fused store is also supported via `--feature-store` (sets both
input and target to the same store), mutually exclusive with
`--input-feature-store`/`--target-feature-store`.

Main metrics:

```text
spearman
top-k overlap
ndcg@k
target importance mass retained
oracle importance mass retained
relative importance mass retained
selected tile count (n_tiles_kept)
effective keep ratio
```

ABMIL agreement evaluation is optional and off by default. Pass
`--abmil-checkpoint` to additionally evaluate full-vs-pruned ABMIL
prediction agreement (`full_accuracy`, `pruned_accuracy`,
`prediction_agreement`, `mean_logit_cosine_similarity`,
`mean_prob_kl_full_to_pruned`) using the same selection-store tile features
the forecaster scored; without it, only target-importance recovery metrics
are reported. This mirrors `evaluate_wsi_abmil_pruning_agreement.py`'s
metrics but does not require an ABMIL checkpoint for the new pipeline.

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

### `run_wsi_tile_importance_synthetic_smoke.sh`

Full synthetic tile-importance (paired-store) pipeline smoke. Does not
require a real dataset or a TRIDENT installation. See
`docs/wsi_tile_importance_forecasting.md` §6 for a step-by-step walkthrough.

```bash
bash scripts/run_wsi_tile_importance_synthetic_smoke.sh
```

Checks:

```text
create_synthetic_wsi_importance_stores.py  (paired early/late/importance stores)
validate_wsi_feature_store.py              (each store individually)
validate_wsi_paired_feature_stores.py      (early/importance pair)
split_wsi_feature_store.py
train_wsi_importance_forecaster.py         (paired stores, --alignment-mode coords)
evaluate_wsi_importance_pruning.py
create_pruned_wsi_feature_store.py         (--selection-feature-store early, --materialize-feature-store late)
validate_wsi_feature_store.py              (pruned late store)
```

Useful environment variables:

```text
WORKDIR
EARLY_FEATURE_DIM
LATE_FEATURE_DIM
HIDDEN_DIM
N_SLIDES
MIN_TILES
MAX_TILES
N_CLASSES
NOISE_STD
BATCH_SIZE
FORECASTER_EPOCHS
KEEP_RATIOS
PRUNED_KEEP_RATIO
DEVICE
SEED
```

Example:

```bash
WORKDIR=/tmp/eaf_wsi_importance_smoke \
N_SLIDES=8 \
FORECASTER_EPOCHS=1 \
DEVICE=cpu \
bash scripts/run_wsi_tile_importance_synthetic_smoke.sh
```

Output layout under `WORKDIR`:

```text
features_layer2.h5
features_late.h5
features_synthetic_importance.h5
features_late_pruned_keep_0.10.h5
splits/{train,val,test}.txt, split_summary.json
checkpoints/importance_forecaster/{best_wsi_tile_importance_forecaster.pt,training_summary.json}
results/wsi_importance_pruning.csv
reports/{paired_validation.json,pruned_store_validation.json}
```

---

## 8. Importance target providers

`src/models/wsi/importance_providers.py` defines `WSIImportanceProvider`, a
small interface for obtaining a tile-importance target from a `WSIBag`:

```python
class WSIImportanceProvider(ABC):
    name: str
    def compute_tile_importance(self, bag: WSIBag) -> tuple[torch.Tensor, dict]: ...
```

Implementations:

```text
PrecomputedImportanceProvider     looks up bag.slide_id in a target
                                   WSIFeatureStore (e.g. one produced by
                                   import_wsi_importance_targets.py)
ABMILImportanceProvider           wraps a trained ABMILClassifier and
                                   returns its attention output
TridentSlideEncoderImportanceProvider
                                   documented stub: most TRIDENT slide
                                   encoders do not expose tile-level
                                   attention, so this raises
                                   NotImplementedError rather than
                                   guessing. TRIDENT is imported lazily
                                   inside compute_tile_importance only,
                                   never at module import time, so this
                                   module has no hard TRIDENT dependency.
```

Use these when writing new target-generation code; use
`import_wsi_importance_targets.py` when the target was already computed by
an external pipeline (TRIDENT-based or not).

---

## 9. Legacy tile-level scripts

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

## 10. Safe cleanup

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
