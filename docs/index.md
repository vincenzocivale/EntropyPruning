# EAF — Documentation

**EAF** (Entropy-guided Attention-based token pruning Framework) is a three-phase pipeline that trains an *AttentionForecaster* to predict which patch tokens to prune from a Vision Transformer at inference time, reducing compute while preserving classification accuracy on histopathology images.

The pipeline is built on top of [THUNDER](https://github.com/MICS-Lab/thunder) for foundation model loading and benchmark dataset access.

---

## Contents

| Document | Description |
|---|---|
| [Setup](setup.md) | Environment setup and installation |
| [Machine Paths (Nanopore-PC)](machine_paths.md) | Path critici dataset, pesi, checkpoint su questa macchina |
| [Architecture](architecture.md) | System design, class hierarchy, data flow |
| [Adaptation Strategies](adaptation_strategies.md) | Choosing between linear probing, LoRA, full fine-tuning, BitFit |
| [Training Pipeline](training_pipeline.md) | Step-by-step Phase 1 → 2 → 3 guide |
| [Training Performance Bottlenecks](performance_bottlenecks.md) | CPU/RAM-aware bottlenecks and resource-friendly run settings |
| [Avviare un esperimento](run_experiment.md) | Comandi pronti all'uso dato encoder + dataset |
| [Thunder Integration](thunder_integration.md) | How Thunder models and datasets are used |
| **API Reference** | |
| [Classifiers](api/classifiers.md) | `BaseClassifier`, all strategy classes, `build_classifier` |
| [BackboneAdapter](api/backbone_adapter.md) | `ThunderBackboneAdapter` |
| [Forecaster](api/forecaster.md) | `AttentionForecaster` |
| [Pruned Models](api/pruned_model.md) | `GenericLoRAWithForecasterPruning`, `FrozenPrunedLinearProbe`, `DistilledPrunedBackbone` |

---

## Quick start

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Phase 1 — train base classifier (LoRA, default)
python scripts/train_classifier.py \
    --model-name uni --dataset-name crc \
    --base-data-folder /path/to/thunder/data

# 3. Phase 2 — train AttentionForecaster
python scripts/train_forecaster.py \
    --model-name uni --dataset-name crc \
    --base-data-folder /path/to/thunder/data \
    --layers-source 2

# 4. Phase 3 — fine-tune pruned model
python scripts/finetune_pruned.py \
    --model-name uni --dataset-name crc \
    --base-data-folder /path/to/thunder/data \
    --prune-layer 2 --keep-ratio 0.1
```

---

## Supported models

Timm-based (fully supported):

| Name | Architecture | embed\_dim | n\_blocks | n\_patches |
|---|---|---|---|---|
| `uni` | ViT-L/16 | 1024 | 24 | 196 |
| `uni2h` | ViT-H/14 | 1536 | 24 | 256 |
| `hoptimus0`, `hoptimus1` | ViT-g/14 | 1536 | 40 | 256 |
| `virchow` | ViT-H/14 | 1280 | 32 | 256 |
| `virchow2` | ViT-H/14 | 1280 | 32 | 256 |
| `kaiko_vitb16` | ViT-B/16 | 768 | 12 | 196 |
| `kaiko_vitb8` | ViT-B/8 | 768 | 12 | 784 |
| `dinov2base`, `dinov2large` | DINOv2 | 768 / 1024 | 12 / 24 | 256 |

HuggingFace-based (not yet supported — `NotImplementedError`):
`phikon`, `phikon2`, `hiboul`, `hiboub`
