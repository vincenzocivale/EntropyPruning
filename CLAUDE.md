# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EAF (Entropy-based Attention Forecasting) is a ViT token pruning framework for efficient whole-slide image (WSI) inference. It uses an **AttentionForecaster** that predicts patch importance from early transformer layers, allowing efficient in-place pruning of low-importance patches before later layers process them.

The `wsi-eval` branch integrates TRIDENT (WSI loading/encoding) and Patho-Bench (benchmark datasets) to evaluate pruning effectiveness on real WSI tasks across multiple organs and cancer types.

## Setup

```bash
# Create environment
conda env create -f environment_wsi.yml
conda activate eaf-wsi

# Install TRIDENT and Patho-Bench (no-deps to preserve timm==1.0.25)
pip install -e /path/to/TRIDENT --no-deps
pip install -e /path/to/Patho-Bench
```

**Note:** The environment specifies `timm==1.0.25` for backbone compatibility. TRIDENT's pyproject.toml lists `timm==0.9.16`, but using `--no-deps` preserves our version since TRIDENT's encoder loading is compatible.

## Quick Start

```bash
# Full 3-phase pipeline: forecaster training + distillation + WSI evaluation
python scripts/run_wsi_pipeline.py \
  --encoder uni_v1 \
  --dataset TCGA-BRCA \
  --task subtype \
  --wsi-dir /path/to/wsis \
  --output-dir ./results \
  --prune-layer 4 \
  --keep-ratio 0.5

# Individual phases can be run with --skip-phase1, --skip-phase2 flags
python scripts/run_wsi_pipeline.py ... --skip-phase1
```

## Architecture

The pruning pipeline has three phases:

### Phase 1 — AttentionForecaster Training
**Script:** `scripts/wsi_train_forecaster.py`

Trains a lightweight attention predictor on cached WSI patch embeddings. The forecaster learns to predict final-block CLS-to-patch attention weights from intermediate-layer embeddings, enabling patch importance scoring.

**Key models:** `src/models/forecaster.py` (AttentionForecaster), `src/models/backbone_adapter.py` (TRIDENT encoder integration)

### Phase 2 — Pruned Model Distillation  
**Script:** `scripts/wsi_distill_pruned.py`

Fine-tunes the pruned model using knowledge distillation. Teacher is the full non-pruned model, student is the pruned encoder with LoRA adaptation plus the forecaster.

**Key models:** `src/models/pruned_classifier.py` (GenericLoRAWithForecasterPruning)

### Phase 3 — WSI-Level Evaluation
**Script:** `scripts/wsi_evaluate.py`

Loads full WSI slides, extracts tiles on-demand at a given magnification, runs inference through the pruned model, and aggregates tile predictions for whole-slide classification. Evaluates accuracy/F1 against Patho-Bench ground truth.

**Data loading:** `src/data/wsi_tile_dataset.py` (streams tiles from WSI files)

## Source Structure

- **`src/models/`**
  - `forecaster.py` — AttentionForecaster: MLP that predicts attention weights
  - `backbone_adapter.py` — Wraps TRIDENT encoders for patch extraction and LoRA support
  - `pruned_classifier.py` — Classification head + forecaster + pruning logic
  - `classifier.py` — Standard (unpruned) classification baseline

- **`src/data/`**
  - `wsi_tile_dataset.py` — WSIPatchDataset: streams tiles from WSI files; supports caching
  - `h5_dataset.py` — HDF5-based cached feature datasets (for Phase 1 forecaster training)
  - `thunder_loaders.py` — Patho-Bench dataset loaders via TRIDENT/Patho-Bench

- **`src/evaluation/`**
  - `benchmark.py` — Evaluation metrics and WSI-level aggregation
  - `metrics.py` — Classification metrics (accuracy, F1, AUC, etc.)

- **`src/baselines/`** — Comparison token-pruning methods (CROP-R, EVit, DynamicViT, UCB)

## Configuration

All scripts support command-line arguments for:
- **Model:** `--encoder` (e.g., `uni_v1`, `virchow`), `--model-name` (for backbones)
- **Data:** `--dataset`, `--task`, `--wsi-dir`, `--mag` (magnification), `--patch-size`, `--tiles-per-wsi`
- **Training:** `--epochs-phase1`, `--epochs-phase2`, `--keep-ratio`, `--prune-layer`, `--batch-size`
- **Logging:** `--wandb-project` (optional, requires W&B API key)

## Common Development Commands

```bash
# Run a single phase manually
python scripts/wsi_train_forecaster.py --encoder uni_v1 --dataset TCGA-BRCA --task subtype ...

# Evaluate without training (requires checkpoints)
python scripts/wsi_evaluate.py --encoder uni_v1 --dataset TCGA-BRCA --task subtype ...

# Quick debugging: smaller dataset slice
python scripts/run_wsi_pipeline.py ... --tiles-per-wsi 16 --epochs-phase1 3 --epochs-phase2 2
```

## Key Dependencies

- **TRIDENT** — WSI loading, tile extraction, foundation model encoders (UNI, Virchow, etc.)
- **Patho-Bench** — WSI datasets and task definitions (TCGA, BACH, BreakHIS, etc.)
- **PyTorch, Lightning** — Training and distributed computation
- **PEFT** — LoRA fine-tuning layers
- **W&B** (optional) — Experiment tracking

## Notes

- No unit tests in repository; validation is via WSI-level evaluation accuracy
- Output checkpoints and logs are stored in `./results/` by default (configurable)
- Feature caching (HDF5) happens automatically during Phase 1 extraction; can be reused across experiments
- The forecaster is layer-pair-specific; a new forecaster must be trained for each (source_layer, target_layer) combination
