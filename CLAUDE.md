# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This branch (`cropr-evaluation`) evaluates **CropR** — a progressive token pruning method — applied to pre-trained Thunder histopathology foundation models.  CropR inserts lightweight cross-attention scorer modules between transformer blocks and progressively drops the lowest-scoring patch tokens, reducing compute while preserving classification accuracy.

## Environment

```bash
conda env create -f environment.yml && conda activate trident
```

## Paths

- **Base data folder:** `/dune/DATASETS/EAF_results/datasets` (contains `data_splits/` and dataset dirs)
- **Pretrained model weights:** `/dune/DATASETS/EAF_results/pretrained_ckpts/`
- **Output checkpoints:** `/dune/DATASETS/EAF_results/checkpoints/`

Set the required env var before any command:
```bash
export THUNDER_BASE_DATA_FOLDER=/dune/DATASETS/EAF_results
```

## Available models and datasets

**Models (pretrained weights present):** `uni`, `uni2h`

**Datasets (data split present):** `patch_camelyon`, `spider_colorectal`, `tcga_crc_msi`

## Training

```bash
export THUNDER_BASE_DATA_FOLDER=/dune/DATASETS/EAF_results

python scripts/train_cropr.py \
  --model-name uni \
  --dataset-name patch_camelyon \
  --base-data-folder /dune/DATASETS/EAF_results/datasets \
  --pruning-rate 8 \
  --epochs 30 \
  --batch-size 32 \
  --output-dir /dune/DATASETS/EAF_results/checkpoints/patch_camelyon/uni_cropr_pr8
```

Key flags:
- `--freeze-backbone` — train only CropR modules + head (backbone frozen); default uses LoRA
- `--pruning-rate N` — tokens removed per block; with UNI (196 patches, 24 blocks, 23 CropR modules) `--pruning-rate 8` leaves ~12 tokens before the final block
- `--wandb-project <name>` — enable W&B logging (omit to skip)
- `--lora-r / --lora-alpha` — LoRA rank/alpha for backbone adaptation (ignored with `--freeze-backbone`)

Supported models (timm-based, `num_prefix_tokens=1`): `uni`, `uni2h`, `hoptimus0`, `hoptimus1`, `virchow`, `virchow2`, `h0mini`, `kaiko_vit*`, `dinov2base`, `dinov2large`.

Supported datasets: any Thunder classification dataset (`crc`, `mhist`, `break_his`, `patch_camelyon`, etc.) except `bracs`.

## Architecture

### `src/models/cropr_classifier.py` — `CroprClassifier`

Wraps a pre-trained timm ViT backbone and inserts one `Cropr` module after each transformer block except the last.

**Forward pass:**
1. `patch_embed` → `_pos_embed` → `patch_drop` → `norm_pre`
2. For each block i (0..n-2): run block → `Cropr[i]` prunes lowest-scoring patch tokens → continue
3. Run final block (no pruning) → `norm` → pool → `fc_norm` → classification head

**Training output:** `[main_logits, aux_0, ..., aux_{n-2}]` — `train_cropr.py` sums CE losses over all heads.
**Eval output:** `main_logits` only (Cropr intermediate heads return `None` at inference).

**Backbone adaptation:** frozen (`--freeze-backbone`) or LoRA (`target_modules: qkv, proj, fc1, fc2`). The `raw_backbone` property always returns the unwrapped timm model for block-level access, regardless of peft wrapping.

### `src/baselines/cropr/cropr.py` — `Cropr` (original, unmodified)

Cross-attention scorer + pruning module.  `scores[:, 0] = math.inf` protects the CLS token at position 0.  During training uses `x.detach()` so auxiliary loss gradients do not flow into the backbone; only the main classification loss reaches backbone weights.

### `src/data/thunder_loaders.py` — `build_thunder_loaders`

Wraps Thunder's `get_data()` into `(train_loader, val_loader, test_loader, class_names, n_classes)`. Training uses `WeightedRandomSampler` for class balance.

### `src/evaluation/metrics.py` — `evaluate`

`evaluate(model, loader, device, far_threshold)` → `{acc, f1_macro, tar_at_far, threshold}`. Calls `model(x)` and expects a plain logit tensor — compatible with `CroprClassifier` in eval mode.
