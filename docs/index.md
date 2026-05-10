# EAF — Documentation

**EAF** (Entropy-based Attention Forecasting) is a three-phase framework that trains an **AttentionForecaster** to predict which patch tokens to prune from a Vision Transformer at inference time, reducing compute while preserving whole-slide image (WSI) classification accuracy on histopathology tasks.

The pipeline is built on top of [TRIDENT](https://github.com/jlevy44/TRIDENT) for WSI loading, tile encoding with foundation models (UNI, Virchow, etc.), and [Patho-Bench](https://github.com/jlevy44/Patho-Bench) for benchmark datasets.

---

## Contents

| Document | Description |
|---|---|
| [Setup](setup.md) | Environment setup with conda (eaf-wsi) and TRIDENT installation |
| [Architecture](architecture.md) | System design, three-phase pipeline, class hierarchy, data flow |
| [Training Pipeline](training_pipeline.md) | Step-by-step Phase 1 → 2 → 3 execution guide |
| [Run an Experiment](run_experiment.md) | Quick-start commands for common encoder + dataset combinations |
| **API Reference** | |
| [AttentionForecaster](api/forecaster.md) | Attention prediction model and training interface |
| [Pruned Model](api/pruned_model.py) | `GenericLoRAWithForecasterPruning` and inference pruning logic |
| [BackboneAdapter](api/backbone_adapter.md) | TRIDENT encoder interface |

---

## Quick start

```bash
# 1. Activate environment
conda activate eaf-wsi

# 2. Full 3-phase pipeline: forecaster training + distillation + WSI evaluation
python scripts/run_wsi_pipeline.py \
    --encoder uni_v1 \
    --dataset TCGA-BRCA \
    --task subtype \
    --wsi-dir /path/to/wsis \
    --output-dir ./results \
    --prune-layer 4 \
    --keep-ratio 0.5

# Or run individual phases:
python scripts/wsi_train_forecaster.py \
    --encoder uni_v1 \
    --wsi-dir /path/to/wsis \
    --prune-layer 4 \
    --output-dir ./results

python scripts/wsi_distill_pruned.py \
    --encoder uni_v1 \
    --wsi-dir /path/to/wsis \
    --forecaster-ckpt ./results/forecaster_*.pt \
    --prune-layer 4 \
    --keep-ratio 0.5 \
    --output-dir ./results

python scripts/wsi_evaluate.py \
    --encoder uni_v1 \
    --dataset TCGA-BRCA \
    --task subtype \
    --checkpoint ./results/best_*.pt \
    --output-dir ./results
```

---

## Supported encoders (TRIDENT)

| Name | Architecture | Embed Dim | Blocks | Patches |
|---|---|---|---|---|
| `uni_v1` | ViT-L/16 | 1024 | 24 | 196 |
| `virchow` | ViT-H/14 | 1280 | 32 | 256 |
| `virchow2` | ViT-H/14 | 1280 | 32 | 256 |
| `hoptimus0` | ViT-g/14 | 1536 | 40 | 256 |
| `hiboul` | Custom | 1536 | 40 | 256 |

---

## Dataset support

Patho-Bench datasets accessible via TRIDENT:

- **TCGA** (multiple organs: BRCA, LUAD, KIRC, COAD, etc.)
  - Classification tasks: `subtype`, `mutational_status`, organ-specific tasks
- **BACH** (Breast histology)
- **BreakHIS** (Breast cancer)
- **CAMELYON16/17** (Lymph node metastasis detection)

Specify via `--dataset` and `--task` arguments.

---

## Architecture overview

The three-phase pruning pipeline:

### Phase 1 — AttentionForecaster Training
Self-supervised training on WSI tiles. Learns to predict final-block CLS-to-patch attention from intermediate-layer embeddings.

**Script:** `scripts/wsi_train_forecaster.py`  
**Input:** WSI files (tiles extracted on-the-fly via TRIDENT)  
**Output:** Trained forecaster checkpoint  
**Validation:** Periodic on validation tile batches (not per-WSI)

### Phase 2 — Pruned Model Distillation
Fine-tune the pruned student model via knowledge distillation. Teacher (non-pruned) and student (pruned with LoRA) both process the same tiles; loss minimizes CLS embedding divergence.

**Script:** `scripts/wsi_distill_pruned.py`  
**Input:** Forecaster checkpoint, WSI files  
**Output:** Pruned model checkpoint with LoRA weights  
**Validation:** Per-epoch on validation tile batches (for early stopping)

### Phase 3 — WSI-Level Evaluation
Evaluate on full WSI slides. Extract tile embeddings, aggregate to slide level (mean pool), train linear classifier, report accuracy/F1 on Patho-Bench ground truth.

**Script:** `scripts/wsi_evaluate.py`  
**Input:** Pruned checkpoint, full WSI slides, labels  
**Output:** Per-slide predictions, evaluation metrics (accuracy, F1, AUC)

---

## Key design decisions

- **TRIDENT integration**: Encoders loaded via `encoder_factory()`, WSI processing via TRIDENT's `load_wsi()`, `WSIPatcher`, and segmentation models.
- **Tile-based training**: Phases 1 & 2 train on sampled tiles per WSI per epoch (no caching), reducing memory overhead and enabling infinite tile diversity.
- **LoRA-only pruning**: Phase 2 always uses LoRA on the backbone, independent of the initial encoder architecture.
- **Straight-through estimator**: During Phase 2 training, pruning masks use hard selection at inference but allow gradients to flow for LoRA updates.

---

## Environment

- **Conda:** `eaf-wsi` (created from `environment_wsi.yml`)
- **Key packages:** PyTorch, TRIDENT, Patho-Bench, PyTorch Lightning, PEFT (LoRA), W&B (optional)
- **GPU:** 24+ GB VRAM recommended for ViT-L/H/g backbones
