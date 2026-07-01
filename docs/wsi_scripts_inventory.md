# WSI Scripts Inventory and Cleanup Notes

This file explains which scripts are currently part of the WSI feature-level pipeline and which files can be safely cleaned.

---

## 1. Keep: WSI data import

```text
build_trident_manifest.py
import_trident_feature_store.py
build_generic_feature_manifest.py
import_generic_feature_store.py
build_wsi_importance_manifest.py
import_wsi_importance_targets.py
create_synthetic_wsi_feature_store.py
create_synthetic_wsi_importance_stores.py
```

Purpose:

```text
external feature files
  -> manifest
  -> EAF HDF5 WSI feature store
```

These are required for real-data and synthetic-data entry points.

---

## 2. Keep: WSI store QA and split management

```text
validate_wsi_feature_store.py
inspect_wsi_feature_store.py
split_wsi_feature_store.py
```

Purpose:

```text
feature store validation
feature store statistics
persistent train/val/test split creation
```

These should be run before experiments.

---

## 3. Keep: WSI teacher and attention target

```text
train_wsi_abmil.py
extract_wsi_abmil_attention.py
```

Purpose:

```text
train ABMIL teacher
extract tile attention targets
```

These are required for the WSI attention forecasting workflow.

---

## 4. Keep: WSI forecaster and pruning

```text
train_wsi_attention_forecaster.py
train_wsi_importance_forecaster.py
evaluate_wsi_forecaster_pruning.py
evaluate_wsi_importance_pruning.py
evaluate_wsi_abmil_pruning_agreement.py
plot_wsi_pruning_curves.py
create_pruned_wsi_feature_store.py
```

Purpose:

```text
train tile attention forecaster (legacy, single fused store, KL loss only)
train tile importance forecaster (paired stores, configurable loss)
evaluate attention-level pruning quality (legacy, single fused store)
evaluate tile-importance pruning quality (paired stores, optional ABMIL agreement)
evaluate prediction preservation after pruning
plot pruning curves
materialize pruned feature stores (legacy single-store, or selection/materialize stores)
```

`train_wsi_importance_forecaster.py` generalizes
`train_wsi_attention_forecaster.py`; `evaluate_wsi_importance_pruning.py`
generalizes `evaluate_wsi_forecaster_pruning.py`; `create_pruned_wsi_feature_store.py`
gained `--selection-feature-store`/`--materialize-feature-store` support.
The legacy scripts and legacy CLI flags are kept and unmodified.

These are the core EAF-style WSI pruning tools.

---

## 5. Keep: smoke scripts

```text
run_trident_import_smoke.sh
run_generic_import_smoke.sh
run_wsi_synthetic_e2e_smoke.sh
run_wsi_tile_importance_synthetic_smoke.sh
```

Purpose:

```text
fast sanity checks for import paths and end-to-end WSI pipeline
run_wsi_tile_importance_synthetic_smoke.sh additionally covers the
paired-store tile-importance pipeline (see docs/wsi_tile_importance_forecasting.md)
```

These scripts are intentionally committed because tests call them and because they document executable workflows.

---

## 6. Keep unless intentionally removing legacy tile-level EAF

```text
train_classifier.py
train_forecaster.py
finetune_pruned.py
ablations/
```

These are not WSI-specific, but they appear to be the original tile-level EAF pipeline.

Do not delete them unless you are deliberately removing or archiving the original EAF workflow.

---

## 7. Safe to remove

```text
scripts/__pycache__/
```

Command:

```bash
rm -rf scripts/__pycache__
```

Also remove accidental runtime files from repository root:

```text
features.h5
*.tmp
local checkpoints
local reports
local results
```

Check with:

```bash
git status --short
```

---

## 8. Recommended `.gitignore` patterns

The repository should not track generated experiment outputs by default.

Suggested patterns:

```text
logs/
checkpoints/
results/
reports/
data/*.h5
*.h5
__pycache__/
.pytest_cache/
```

Be careful with `data/*.h5` if the repository intentionally tracks small test fixtures.
