# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**EAF — Early Attention Forecaster**: a ViT token-pruning method for efficient histopathology tile
encoding. A small *AttentionForecaster* is trained to predict, from an early Transformer block, the
final-block `[CLS]`→patch attention; at inference the top-`k` patches are kept and the deep blocks run on
a much shorter sequence. `metodo_EAF_paper.md` is the original paper spec (ECCV 2026 submission).

**Alignment with the paper:** Phase 1 = the paper's *Stage 1* — full-backbone LoRA fine-tuning (qkv/proj/
fc1/fc2 of every block) + a fresh classification head, trained on full (un-pruned) sequences. Phase 2
distills this LoRA-adapted backbone's final-block CLS→patch attention. Phase 3 = the paper's *Stage 3*:
the pruning-aware classifier (same LoRA config + head) is **warm-started from Phase 1's converged
adapters** and fine-tuned with pruning active after `--prune-layer` — use a reduced LR (e.g.
`--lr-backbone 1e-5 --lr-head 1e-4`) and fewer epochs for this gentle adaptation step. If no Phase-1
checkpoint exists, Phases 2/3 fall back to a frozen-pretrained-backbone teacher / cold-start LoRA,
respectively.

This repository contains **only the EAF method** (no comparison baselines). It depends on the external
[`thunder`](https://github.com/MICS-Lab/thunder) package for foundation-model loading and benchmark
dataset access (`from thunder.models.pretrained_models import get_model_from_name`).

## Environment

- **Working env on this machine:** `conda activate eaf_env` (torch 2.5.1, `thunder`, `peft`, `h5py` present).
- `environment.yml` recreates the env (declared name: `trident`); `thunder` must be pip-installed manually
  (see the commented note at the bottom of `environment.yml`).
- Run any command via: `conda run -n eaf_env python scripts/<script>.py ...`.

## Three-phase pipeline

All scripts take `--model-name` (thunder model, e.g. `uni`, `virchow2`), `--dataset-name` (thunder dataset,
e.g. `break_his`, `crc`), and `--base-data-folder` (contains `data_splits/` + `datasets/`; populate with
`thunder download <dataset>`). Checkpoints are written under `checkpoints/<dataset>/...`.

```bash
# Phase 1 — full-backbone LoRA (qkv/proj/fc1/fc2, every block) + a fresh classification
#           head, trained on full sequences with cross-entropy (paper's Stage 1)
python scripts/train_phase1_lora.py --model-name uni --dataset-name break_his \
    --base-data-folder <DATA>

# Phase 2 — train the AttentionForecaster to distill the Phase-1 LoRA-adapted backbone's
#           final-block CLS→patch attention (caches features on first run; falls back to
#           the frozen pretrained backbone if no Phase-1 checkpoint is found)
python scripts/train_forecaster.py  --model-name uni --dataset-name break_his \
    --base-data-folder <DATA> --layers-source 2

# Phase 3 — pruning-aware fine-tuning: forecaster frozen, top-k pruning after a block,
#           LoRA + head warm-started from Phase 1, fine-tuned with a reduced LR (paper's
#           Stage 1 -> Stage 3 warm start)
python scripts/finetune_pruned.py   --model-name uni --dataset-name break_his \
    --base-data-folder <DATA> --prune-layer 2 --keep-ratio 0.2 \
    --lr-backbone 1e-5 --lr-head 1e-4 --epochs 10

# Inference-only evaluation of saved pruned checkpoints
python scripts/evaluate_pruned_checkpoints.py --model-name uni --dataset-name break_his \
    --base-data-folder <DATA> --keep-ratios 0.1 0.2
```

Phase-1 checkpoints are written to
`checkpoints/<dataset>/<model>_lora_phase1/best_<model>_<dataset>_phase1.pt` and contain the LoRA adapters
+ head (`{"backbone": ..., "head": ...}`). Phases 2 and 3 auto-detect this file by default; override with
`--phase1-ckpt` on either script (`--lora-r`/`--lora-alpha` must match Phase 1 for the state dicts to be
compatible), or skip Phase 1 entirely to fall back to a frozen-pretrained-backbone teacher (Phase 2) /
cold-start LoRA (Phase 3, use the higher default LR/epochs in that case).

## Architecture (`src/`)

- `models/backbone_adapter.py` — `ThunderBackboneAdapter`: uniform access (embed_dim, blocks, prefix tokens)
  over timm-based thunder backbones.
- `models/lora_utils.py` — `wrap_lora`: shared LoRA config (qkv/proj/fc1/fc2, all blocks) used by both
  Phase 1 and Phase 3 so their state dicts are key-compatible.
- `models/lora_classifier.py` — `GenericLoRAClassifier` (Phase 1: full-backbone LoRA + head, trained on
  full sequences), `load_lora_adapted_backbone` (builds the frozen LoRA-adapted Phase-2 teacher) and
  `load_lora_adapted_weights` (warm-starts Phase 3's backbone+head from a Phase-1 checkpoint).
- `models/extractor.py` — `FrozenBackbone`: thin frozen wrapper used only to extract the teacher attention
  in Phase 2 (exposes `.adapter`/`.raw_backbone` + a forward that fires the attention hooks); `raw_backbone`
  unwraps a `peft.LoraModel` if the teacher is Phase-1 LoRA-adapted.
- `models/forecaster.py` — `AttentionForecaster`: light self-attn + learnable-query cross-attn → per-patch score.
- `models/pruned_classifier.py` — `GenericLoRAWithForecasterPruning`: runs blocks up to `prune_layer`, scores
  patches with the frozen forecaster, keeps top-`keep_ratio` (CLS/register tokens always kept), runs the rest.
  Same LoRA config + head as `GenericLoRAClassifier`, so it can be warm-started from a Phase-1 checkpoint.
- `collection/extract_features.py` — `collect_and_save_dataset`: hooks attention to cache patch embeddings
  (source layer) and `[CLS]`→patch attention (target layer) to HDF5 for Phase-2 training.
- `data/` — `thunder_loaders.py` (thunder datasets → `(image, label)` loaders, weighted sampler),
  `h5_dataset.py` (`H5ForecastDataset` for cached features/targets).
- `evaluation/` — `metrics.py` (acc, F1-macro, TAR@FAR), `benchmark.py` (latency + FLOPs).
- `losses.py` — Phase-2 distillation loss: KL divergence between the forecaster's softmax and the
  L1-normalized teacher CLS→patch attention.
- `utils.py` — seeding, device, optimizer builder (`build_optimizer`, used by Phase 1's
  `trainable_backbone_params`/`head` split), grad-norm, results IO.

## Tests

`pytest -q` — CPU-only smoke tests (`tests/`): core imports, the distillation loss (differentiability,
`kl` ≡ `F.kl_div`, NaN-safety), the pruned forward pass, and Phase1→Phase3 LoRA checkpoint
warm-start compatibility (`test_phase1_checkpoint_warm_starts_phase3`). No GPU/thunder/dataset
required (uses `vit_tiny` / synthetic tensors).
