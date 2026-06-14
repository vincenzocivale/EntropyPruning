# Training Pipeline

Complete walkthrough of the three-phase EAF training process.

---

## Prerequisites

```bash
source .venv/bin/activate

# Thunder data must be downloaded:
thunder download crc --base-data-folder /path/to/thunder/data
```

Set a convenience variable:
```bash
DATA=/path/to/thunder/data
MODEL=uni          # or hoptimus0, virchow, dinov2base, etc.
DATASET=crc        # or break_his, mhist, patch_camelyon, etc.
```

---

## Phase 1 — Base Classifier

Fine-tune a classification head on top of the backbone using the chosen adaptation strategy.

```bash
python scripts/train_classifier.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --adaptation lora \
    --lora-r 8 \
    --lora-alpha 32 \
    --epochs 20 \
    --batch-size 8 \
    --lr-backbone 1e-5 \
    --lr-head 1e-3 \
    --weight-decay 0.01 \
    --warmup-steps 100 \
    --label-smoothing 0.1 \
    --early-stopping-patience 3
```

**Early stopping:** Ferma il training se la metrica scelta non migliora per N epoche consecutive (default: 3).
Imposta `--early-stopping-patience 0` per disabilitarlo.

**Output:** `checkpoints/$DATASET/${MODEL}_lora/best_model.pt`

The script prints a classification report on the test split at the end.

### Choosing `--prune-layer` early

The source layer used for feature extraction in Phase 2 should be decided now.
For a ViT-L (24 blocks), layer 2 is a good starting point.
For H-optimus (40 blocks), layer 5–8.

---

## Phase 2 — AttentionForecaster

### Step 2a — Feature extraction

The first time you run Phase 2, `collect_and_save_dataset` is called automatically.
It runs the Phase 1 model in eval mode and caches, for each image:
- Patch embeddings at the source layer(s): shape `(P, D)`, float16
- CLS attention weights at the target layer: shape `(P,)`, float16

```bash
python scripts/train_forecaster.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --adaptation lora \            # must match Phase 1
    --layers-source 2 \            # which layers to extract embeddings from
    --layer-target 23 \            # which layer to predict attention for (default: last)
    --epochs 30 \
    --hidden 256 \
    --n-heads 4 \
    --n-layers 2 \
    --lr 1e-4 \
    --wandb-project eaf-forecaster
```

**Feature cache:** `checkpoints/$DATASET/${DATASET}_${MODEL}_features.h5`
**Output:** `checkpoints/$DATASET/${MODEL}_forecaster/forecaster_src02_tgt23.pt`

If the cache already exists (from a previous run), feature extraction is skipped.

### Step 2b — Multiple source layers

You can extract from multiple layers and train one forecaster per layer in a single run:

```bash
python scripts/train_forecaster.py \
    ... \
    --layers-source 2 4 8 12
```

This trains four forecasters: `forecaster_src02_tgt23.pt`, `forecaster_src04_tgt23.pt`, etc.

### Monitoring forecaster quality

Watch `val/rho` in W&B (Spearman rank correlation against ground-truth attention).
A good forecaster typically reaches `rho > 0.5` on the validation set.
`test/delta_vs_norm` shows the improvement over a simple token-norm baseline.

---

## Phase 3 — Pruned Fine-tuning

Load Phase 1 weights into `GenericLoRAWithForecasterPruning`, freeze the forecaster, and fine-tune.

```bash
python scripts/finetune_pruned.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --prune-layer 2 \
    --keep-ratio 0.1 \            # keep 10% of patches
    --epochs 20 \
    --batch-size 16 \
    --lr-backbone 1e-4 \
    --lr-head 1e-3 \
    --wandb-project eaf-pruning \
    --eval-baseline \             # also evaluate unpruned model for comparison
    --early-stopping-patience 3
```

**Early stopping:** Ferma il training se la metrica non migliora per N epoche consecutive (default: 3).

**Output:** `checkpoints/$DATASET/${MODEL}_pruned/best_${MODEL}_prune2_keep10.pt`

### `--keep-ratio` sweep

```bash
for RATIO in 0.05 0.1 0.2 0.3 0.5; do
    python scripts/finetune_pruned.py \
        --model-name $MODEL --dataset-name $DATASET \
        --base-data-folder $DATA \
        --prune-layer 2 --keep-ratio $RATIO \
        --epochs 20
done
```

---

## Evaluation

Reload any pruned checkpoint and evaluate on the test set:

```bash
python scripts/evaluate_pruned_checkpoints.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --prune-layers 2 \
    --keep-ratios 0.05 0.1 0.2 0.3 0.5 \
    --output-csv results/eval_${MODEL}_${DATASET}.csv
```

The CSV contains accuracy, F1-macro, TAR@FAR, ms/img, and GFLOPs for each configuration.

---

## Checkpoint naming conventions

```
checkpoints/
└── {dataset}/
    ├── {model}_{adaptation}/
    │   └── best_model.pt                           ← Phase 1
    ├── {model}_forecaster/
    │   └── forecaster_src{L:02d}_tgt{T:02d}.pt    ← Phase 2
    ├── {dataset}_{model}_features.h5               ← Phase 2 cache
    └── {model}_pruned/
        └── best_{model}_prune{L}_keep{k}.pt        ← Phase 3
```

---

## Unsupervised, multi-dataset forecaster (universal EAF)

An alternative to per-dataset Phase 1 + Phase 2: train **one**
`AttentionForecaster` on a **frozen, pretrained** foundation model (no
Phase-1 fine-tuning) using the FM's own last-layer CLS->patch attention as
the target, computed across a **merged corpus of THUNDER datasets**. This
gives a dataset-independent, "linear-pruning"-style token-importance
predictor.

### Corpus

16 THUNDER classification datasets (excludes the 4 segmentation-only configs
`ocelot`, `pannuke`, `segpath_epithelial`, `segpath_lymphocytes`):

```
bach bracs break_his ccrcc crc esca mhist patch_camelyon
spider_breast spider_colorectal spider_skin spider_thorax
tcga_crc_msi tcga_tils tcga_uniform wilds
```

Download any missing datasets:

```bash
thunder download <dataset> --make-splits --base-data-folder $DATA
```

### Step 1 — Build per-dataset attention caches

```bash
python scripts/build_unsupervised_cache.py \
    --model-name $MODEL \
    --base-data-folder $DATA \
    --layer-source 2
```

For each dataset, extracts `emb_layer{source}` and `attn_layer{last}` from
the unmodified pretrained backbone (wrapped as a frozen, uncheckpointed
`LinearProbingClassifier` -- the head is unused) and writes
`checkpoints/unsupervised/{dataset}_{model}_attn_features.h5`. Skips datasets
whose `data_splits/{name}.json` is missing (prints the `thunder download`
command) and skips datasets whose cache is already valid, so the sweep is
resumable.

### Step 2 — Train the universal forecaster

```bash
python scripts/train_forecaster_unsupervised.py \
    --model-name $MODEL \
    --base-data-folder $DATA \
    --layer-source 2 \
    --epochs 30 \
    --wandb-project eaf-forecaster
```

Builds/reuses the same per-dataset caches as Step 1, concatenates them via
`MultiH5ForecastDataset`, and trains one `AttentionForecaster` with the same
KL-divergence loss as Phase 2. Reports an aggregate test Spearman `rho`
(forecaster vs. token-norm baseline) **and** a per-dataset breakdown.

**Output:**
`checkpoints/unsupervised/{model}_forecaster/forecaster_src{L:02d}_attn{T:02d}_universal.pt`
+ `results_forecaster_src{L:02d}_attn{T:02d}_universal.json`.

### Step 3 — Phase 3 hookup

Point `finetune_pruned.py` at the universal checkpoint via `--forecaster-ckpt`
to fine-tune any single target dataset with a dataset-independent pruner:

```bash
python scripts/finetune_pruned.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --prune-layer 2 \
    --keep-ratio 0.1 \
    --forecaster-ckpt checkpoints/unsupervised/${MODEL}_forecaster/forecaster_src02_attn23_universal.pt
```

---

## Tips

**Out of memory in Phase 1** — reduce `--batch-size` or switch to `--adaptation linear_probing` (no backbone gradients).

**Phase 2 cache is stale** — delete the `.h5` file; it will be regenerated on the next run.

**Phase 3 not improving over baseline** — try a lower `--keep-ratio` (less aggressive pruning), a different `--prune-layer`, or more `--epochs`.

**Forecaster `val/rho` stays near 0** — the source layer may be too early (features not yet discriminative). Try a later source layer.
