# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EntropyPruning is a research project implementing attention entropy-aware token pruning for Vision Transformers (ViTs), applied to histological image classification. The backbone is the UNI (Universal Image) ViT-L model, enhanced with LoRA adapters. An `AttentionForecaster` predicts which tokens to prune at intermediate layers based on source-layer embeddings, enabling efficient inference without recomputing all attention.

## Environment Setup

```bash
conda create -n trident --file environment.yml
# or update existing:
conda env update -n trident --file environment.yml
conda activate trident
```

Key dependencies: PyTorch 2.6.0, transformers 4.51.3, timm 1.0.25, peft (LoRA), wandb, h5py, fvcore.

## Three-Phase Training Pipeline

### Phase 1 — Classifier fine-tuning

```bash
python scripts/train_classifier.py \
    --data-dir /path/to/dataset \
    --img-size 224 \
    --batch-size 8 \
    --epochs 20 \
    --lr-head 1e-3 \
    --lr-backbone 1e-5 \
    --output-dir /raid/DATASETS/checkpoints-Attention-Pruning/<dataset>/uni_finetuned
```

Trains `UNILoRAClassifier` (pretrained UNI + LoRA + linear head). Checkpoint saved as `best_model.pt`.

### Phase 2 — Attention forecaster training

```bash
python scripts/train_forecaster.py \
    --data-dir /path/to/dataset \
    --layers-source 2 \
    --layer-target 23 \
    --hidden 256 \
    --n-heads 4 \
    --n-layers 2 \
    --epochs 30 \
    --lr 1e-4 \
    --wandb-project attention-forecaster
```

Automatically extracts embeddings and attention maps from the frozen Phase 1 classifier via hooks, caches them to HDF5 at `/raid/DATASETS/data_cache/`, then trains `AttentionForecaster` to minimize KL divergence. Validates using Spearman rho.

### Phase 3 — Pruned model fine-tuning

```bash
python scripts/finetune_pruned.py \
    --data-dir /path/to/dataset \
    --classifier-ckpt /path/to/best_model.pt \
    --forecaster-ckpt /path/to/forecaster_src02_tgt23.pt \
    --prune-layer 2 \
    --keep-ratio 0.1 \
    --epochs 20 \
    --lr-head 1e-3 \
    --lr-backbone 1e-4 \
    --wandb-project pruned-finetuning
```

Wires the forecaster into the classifier via a hook at `prune-layer`. Uses a straight-through estimator (soft masking) for differentiable pruning during training.

### Layer ablation study

```bash
python scripts/ablations/layer_ablation.py \
    --data-dir /path/to/dataset \
    --layers-source 2 \
    --layers-target 23 22 21 20 \
    --keep-ratio 0.1 \
    --wandb-project layer-ablation-targets
```

## Architecture

```
src/
├── models/
│   ├── classifier.py          # UNILoRAClassifier — pretrained UNI + LoRA + linear head
│   ├── forecaster.py          # AttentionForecaster — transformer-based attention predictor
│   └── pruned_classifier.py   # UNILoRAWithForecasterPruning — classifier with pruning hook
├── data/
│   ├── dataset.py             # HistologicalImageDataset (HuggingFace, RGB/RGBA handling)
│   ├── loaders.py             # build_loaders() — train/val/test dataloaders, weighted sampling
│   ├── h5_dataset.py          # H5ForecastDataset — loads HDF5 embeddings for forecaster
│   └── transforms.py          # ImageNet-normalized augmentations
├── collection/
│   └── extract_features.py    # collect_and_save_dataset() — hook-based HDF5 extraction
├── evaluation/
│   ├── metrics.py             # evaluate() — accuracy, F1-macro, TAR@FAR
│   └── benchmark.py           # benchmark_model() — inference time and FLOPs via fvcore
├── baselines/
│   ├── cropr/                 # CropR baseline
│   ├── evit/                  # EfficientViT baseline
│   ├── dynamic_vit/           # DynamicViT baseline
│   └── ucb/                   # UCB-based dynamic pruning
└── utils.py                   # set_seed(), get_device()
scripts/
├── train_classifier.py
├── train_forecaster.py
├── finetune_pruned.py
└── ablations/layer_ablation.py
notebooks/                     # Analysis and visualization
```

## Data & Checkpoint Layout

`data/` is local-only and ignored by Git. It holds external Thunder data by
symlink and local TCGA/WSI inputs; raw slides and reusable TRIDENT outputs are
kept once per cohort, while checkpoints, logs, rankings, and generated WSI
feature stores are disposable run artifacts. See `docs/data_layout.md` for the
canonical directory layout and placement rules.

The tile-level forecaster cache uses the following HDF5 layout:

```
{dataset}_forecaster_dataset.h5
└── /{split}/
    ├── labels          [n_samples]
    ├── emb_layer{i}    [n_samples × 196 × 1024]
    └── attn_layer{i}   [n_samples × 196]
```

**Checkpoints** (`/raid/DATASETS/checkpoints-Attention-Pruning/{dataset}/`):
```
uni_finetuned/best_model.pt
forecaster/forecaster_src{src:02d}_tgt{tgt:02d}.pt
pruned_finetuned/pruned_models/
```

## WSI EAF Training Data Policy

The separate WSI-level EAF work (Tile-EAF / WSI-EAF, `src/wsi_pipeline/`,
`src/data/wsi/`, `scripts/eaf.py`) uses a different, larger-scale pretraining
corpus than the classifier/forecaster pipeline above. The canonical runtime
root is `$EAF_WSI_ROOT` (see `docs/data_layout.md`).

**The EAF training corpus is HISTAI + GTEx + HEST only. TCGA is explicitly
excluded from EAF pretraining.** Raw TCGA slides remain untouched under
`sources/gdc/tcga` (recoverable there if ever needed), but the curated
TCGA-derived pretraining datasets (`tcga_eaf_multicohort_v1`,
`tcga_eaf_thunder_clean_v1`, and the merged `eaf_multisource_clean_v1`) were
deleted from `$EAF_WSI_ROOT` on 2026-08-07 — they are not kept on disk even
as an ablation/non-EAF reserve; regenerate from raw via
`scripts/wsi_data/build_tcga_thunder_clean_inventory.py` and
`build_eaf_multisource_manifest.py` if a TCGA-inclusive ablation is ever
needed. TCGA exclusion exists because many downstream THUNDER benchmarks
(`catalog/benchmark_registry.csv`) are themselves TCGA-derived, so including
TCGA in EAF pretraining would risk leakage into those evaluations.

`eaf_wsi_pretrain_strict_v1` (`src/data/wsi/corpora.py:build_strict_corpus`,
`python scripts/eaf.py data build-strict`) is the logical union of HISTAI +
GTEx + HEST — no duplicated pixels, TCGA excluded by construction via a
leakage guard — and is the manifest that EAF training should read. See
`docs/offline_eaf_pipeline.md` and `docs/refactor_migration.md` for the
offline teacher-cache pipeline and the full dataset-layout rationale.

## Key Design Details

- **Layer indexing**: 0-based; layer 23 is the final transformer block; layer 2 is the canonical early prune point used in ablations.
- **Token count**: 196 patch tokens (14×14 grid) from 224×224 images with 16×16 patches.
- **Pruning training**: Straight-through estimator keeps gradients flowing through the discrete pruning mask during fine-tuning.
- **LoRA**: Applied to UNI backbone; only LoRA params + classification head are updated in Phases 1 and 3.
- **Reproducibility**: `set_seed(42)` called at start of each script.
- **Mixed precision**: `torch.amp.autocast()` used throughout for memory efficiency.
- **Gradient clipping**: `clip_grad_norm_(model.parameters(), 1.0)` applied during training.

## Experiment Tracking

All three phases log to Weights & Biases. Pass `--wandb-project <name>` to each script. Phase 2 tracks KL divergence and Spearman rho; Phase 3 tracks accuracy, F1-macro, TAR@FAR, inference time, and FLOPs.
