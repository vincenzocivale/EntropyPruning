# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch: `wsi-eval` (NEW)

**Status**: Replaces `thunder` dependency with TRIDENT. Adds WSI-level evaluation pipeline.

**3-Phase WSI Pipeline**:
- **Phase 1** (`scripts/wsi_train_forecaster.py`): Train AttentionForecaster on WSI tiles (self-supervised, no labels)
- **Phase 2** (`scripts/wsi_distill_pruned.py`): Distillation fine-tune of pruned encoder (teacher = non-pruned, student = pruned + LoRA)
- **Phase 3** (`scripts/wsi_evaluate.py`): In-memory WSI classification comparison (Patho-Bench task, no HDF5)

**Setup**: `conda env create -f environment_wsi.yml` then `pip install -e /path/to/TRIDENT --no-deps && pip install -e /path/to/Patho-Bench`

**Quick start**:
```bash
python scripts/run_wsi_pipeline.py \
  --encoder uni_v1 --dataset TCGA-BRCA --task subtype \
  --wsi-dir /path/to/wsis --output-dir ./results \
  --prune-layer 4 --keep-ratio 0.5
```

Key files: `src/data/wsi_tile_dataset.py` (streaming tiles), `src/models/backbone_adapter.py` (TRIDENT support), all scripts use `encoder_factory` from TRIDENT (no `thunder` imports).

## Repository Structure

This workspace contains two independent projects for computational pathology:

- **`EntropyPruning/`** — ViT token pruning framework for efficient histopathology inference. Has its own `CLAUDE.md` with full details.
- **`thunder/`** — THUNDER benchmark (NeurIPS 2025 Spotlight): evaluates 23 foundation models across 16 datasets and 9+ task types. Published as `thunder-bench` on PyPI.

---

## EntropyPruning

See `EntropyPruning/CLAUDE.md` for the full training pipeline, architecture, and dataset paths. Summary:

- **Environment:** `conda env create -f EntropyPruning/environment.yml && conda activate trident`
- **Three-phase pipeline:** (1) LoRA fine-tune classifier, (2) train AttentionForecaster, (3) fine-tune pruned model
- **Datasets** at `/raid/DATASETS/`; HDF5 cache at `/raid/DATASETS/data_cache/`; checkpoints at `/raid/DATASETS/checkpoints-Attention-Pruning/`

---

## THUNDER (`thunder/`)

### Setup

```bash
cd thunder
pip install -e ".[dev]"
```

### Common Commands

```bash
# Run a benchmark (CLI)
thunder benchmark <model> <dataset> <task>

# Examples
thunder benchmark phikon break_his knn_classification
thunder benchmark uni crc linear_probing
thunder benchmark keep spider_breast zero_shot_vlm

# Download a dataset
thunder download <dataset>

# Run tests
pytest

# Lint
black src/ tests/
isort src/ tests/

# Build docs
mkdocs serve
```

### Python API

```python
from thunder import benchmark
benchmark("phikon", "break_his", "knn")

from thunder.models import get_model_from_name
model, transforms = get_model_from_name("uni")
```

### Architecture

**Entry point:** `src/thunder/main.py` (Typer CLI) → `src/thunder/benchmark.py` (core logic)

**Task types** (`src/thunder/tasks/`):
- `knn_classification` — k-NN on frozen embeddings
- `linear_probing` — linear head on frozen embeddings
- `simple_shot` — few-shot learning
- `image_retrieval` — similarity-based retrieval
- `adversarial_attack` — PGD robustness evaluation
- `transformation_invariance` — RandStain augmentation robustness
- `alignment_scoring` — feature alignment metrics
- `segmentation` — dense prediction
- `zero_shot_vlm` — VLM zero-shot classification (VLMs only: CONCH, KEEP, PLIP, MUSK, TITAN)

**Model zoo** (`src/thunder/models/pretrained_models.py`): 23 models including UNI, Virchow, H-optimus, Phikon, Hibou, DINOv2, CLIP, PLIP, CONCH, KEEP. Some require accepting HuggingFace usage terms before use.

**Configuration system:** Hydra YAML configs in `src/thunder/config/`. Subcategories: `task/`, `model/`, `dataset/`, `adaptation/` (LoRA vs frozen), `data_loading/` (online / image pre-loading / embedding pre-loading), `wandb/`.

**Data loading modes** (`data_loading` config key):
- `online_loading` — stream images on demand
- `image_pre_loading` — load all images to RAM
- `embedding_pre_loading` — use pre-computed HDF5 embeddings (fastest)

**Custom models/datasets:** Provide a Python class implementing the model interface or a YAML dataset config. See `examples/` for `dinov2small.py`, `resnet.py`, `bracs_3classes.yaml`.

**Datasets** (`src/thunder/datasets/`): 16 base datasets (BACH, BRACS, BreakHIS, CCRCC, CRC, ESCA, MHist, OCELOT, PanNuke, Patch-Camelyon, SegPath variants, TCGA variants, WILDS) plus SPIDER variants (breast, colorectal, skin, thorax).

**Test suite:** `tests/test_benchmark.py`, `tests/test_datasets.py`, `tests/test_models.py`. CI runs on Python 3.10–3.13 with offline wandb (`WANDB_MODE=offline`).
