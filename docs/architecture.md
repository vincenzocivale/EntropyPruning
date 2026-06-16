# Architecture

## Overview

EAF trains a lightweight **AttentionForecaster** that, given intermediate patch embeddings from an early transformer block, predicts the CLS-to-patch attention weights at the final block.
At inference time, only the top-`k` scoring patches are kept after the prune layer, reducing the sequence length (and thus compute) of all subsequent blocks.

```
Input image
    │
    ▼
Foundation model (blocks 0 → prune_layer)
    │
    ├──► AttentionForecaster ──► importance scores
    │                                    │
    │         ┌──────────────────────────┘
    │         ▼
    │    Keep top-k patches (prune patches with low scores)
    │         │
    ▼         ▼
Foundation model (blocks prune_layer+1 → end)
    │
    ▼
Classification head → logits
```

---

## Three-phase training pipeline

### Phase 1 — Base classifier

Train a classification head (+ optional backbone adaptation) on the full-token forward pass.
This gives the backbone the task-specific signal needed for Phase 2 feature extraction.

**Script:** `scripts/train_classifier.py`
**Output:** `checkpoints/{dataset}/{model}_{adaptation}/best_model.pt`

### Phase 2 — AttentionForecaster

1. **Feature extraction** — run the Phase 1 model in eval mode over all splits and cache, for each sample, the patch embeddings at `layers_source` and the CLS attention map at `layer_target`. Saved as HDF5.
2. **Forecaster training** — train `AttentionForecaster` on the cached features using KL divergence against the ground-truth CLS attention. One forecaster per `(source_layer, target_layer)` pair.

**Script:** `scripts/train_forecaster.py`
**Output:** `checkpoints/{dataset}/{model}_forecaster/forecaster_src{L:02d}_tgt{T:02d}.pt`
**Cache:** `checkpoints/{dataset}/{dataset}_{model}_features.h5`

### Phase 3 — Pruned fine-tuning

Reload Phase 1 weights into `GenericLoRAWithForecasterPruning`, freeze the forecaster, and fine-tune the backbone (LoRA) + head under the pruning constraint. Uses a straight-through estimator during training for differentiability.

**Script:** `scripts/finetune_pruned.py`
**Output:** `checkpoints/{dataset}/{model}_pruned/best_{model}_prune{L}_keep{k}.pt`

---

## Class hierarchy

### Classifiers (`src/models/classifier.py`)

```
BaseClassifier (ABC)
├── raw_backbone          → timm VisionTransformer (block-level hook access)
├── trainable_backbone_params → params with backbone lr
├── head                  → LayerNorm → Dropout → Linear
│
├── LinearProbingClassifier   backbone frozen, no_grad in forward
├── LoRAClassifier            peft LoRA on qkv / proj / fc1 / fc2
├── FullFinetuneClassifier    all backbone weights trained
└── BitFitClassifier          only bias terms trained
```

`build_classifier(strategy, backbone, adapter, n_classes, **kwargs)` is the factory entry point.

### Backbone adapter (`src/models/backbone_adapter.py`)

`ThunderBackboneAdapter` wraps any timm `VisionTransformer` returned by Thunder's `get_model_from_name` and exposes:

| Attribute | Source |
|---|---|
| `embed_dim` | `model.embed_dim` |
| `n_blocks` | `len(model.blocks)` |
| `n_patches` | `model.patch_embed.num_patches` |
| `num_prefix_tokens` | `model.num_prefix_tokens` (CLS + register tokens) |

### Forecaster (`src/models/forecaster.py`)

`AttentionForecaster(embed_dim, hidden, n_heads, n_layers, dropout)`

Architecture:
```
patch embeddings (B, N, D)
    │
Linear(D → hidden)          — input projection
    │
TransformerEncoder × n_layers  — self-attention over patches
    │
MultiheadAttention × n_layers  — cross-attention: learnable CLS queries attend to patches
    │
Linear(hidden*2 → 128) → GELU → Dropout → Linear(128 → 1)
    │
scores (B, N)               — per-patch importance
```

Trained with KL divergence against ground-truth CLS attention.
Evaluated with Spearman rank correlation (higher = better ranking of important patches).

### Pruned models (`src/models/pruned_classifier.py`)

Two classes share the same pruning mechanism:

**`GenericLoRAWithForecasterPruning`** — Phase 3. Wraps the backbone with LoRA, keeps LoRA + head trainable, freezes the forecaster. Uses a straight-through estimator during training for differentiable pruning.

**`FrozenPrunedLinearProbe`** — Linear probing experiment. Both backbone and forecaster are fully frozen at all times; only a single `nn.Linear` head is optimised. Overrides `train()` to keep backbone and forecaster in eval mode even during head training. Supports multi-layer source: hooks capture patch embeddings at each source block and concatenate them along the feature dimension before feeding to the forecaster.

Pruning forward pass (shared logic):
```
x (B, N_total, D)
    │
prefix = x[:, :num_prefix_tokens]   — CLS + register tokens, always kept
patches = x[:, num_prefix_tokens:]  — spatial patches, subject to pruning
    │
[if multi-layer: concatenate captured embeddings from earlier source blocks]
    │
scores = forecaster(patches / concat_embs)   — (B, N_patches), no grad
    │
keep top-k patches
    │
cat([prefix, kept_patches], dim=1)  — shortened sequence for remaining blocks
```

---

## Data flow

```
Thunder
  get_model_from_name(model_name)
    ├── raw_backbone  ──→ ThunderBackboneAdapter ──→ embed_dim, n_blocks, n_patches, num_prefix_tokens
    └── transform

Thunder
  get_data(dataset_name, base_data_folder)
    └── PatchDataset + _TupleDataset ──→ DataLoader yielding (imgs, labels)

Phase 1:  build_classifier(strategy, backbone, adapter, n_classes)
Phase 2:  collect_and_save_dataset → HDF5 cache → H5ForecastDataset → forecaster training
Phase 3:  GenericLoRAWithForecasterPruning(backbone, adapter, forecaster, prune_layer, keep_ratio)
Exp:      FrozenPrunedLinearProbe(backbone, adapter, forecaster, n_classes, layers_source, keep_ratio)
```

---

## HDF5 cache format (Phase 2)

```
{split}/
    labels              (N,)          int32
    emb_layer{L}        (N, P, D)     float16   — patch embeddings at source layer L
    attn_layer{L}       (N, P)        float16   — CLS attention to patches at target layer L
```

Where `P = adapter.n_patches`, `D = adapter.embed_dim`.
Register tokens are excluded — only spatial patch tokens are stored.

A single cache file can hold embeddings from multiple source layers (e.g. `emb_layer1` through `emb_layer5`), allowing the same HDF5 to be reused for experiments with different `layers_source` configurations without re-extraction.

When `H5ForecastDataset` is given a list of source layers, it concatenates their embeddings along the feature dimension: the returned `emb` tensor has shape `(P, len(layers_source) × D)`. The forecaster must be initialised with the matching `embed_dim`.

---

## Key design decisions

**`raw_backbone` interface** — all classifier variants expose `model.raw_backbone` pointing to the underlying timm `VisionTransformer`. `collect_and_save_dataset` and `GenericLoRAWithForecasterPruning` use `model.raw_backbone.blocks[i]` for attention hooks, making the code model-agnostic regardless of whether LoRA or another adapter wraps the backbone.

**Register tokens** — models like `uni2h` and `hoptimus0/1` prepend register tokens in addition to the CLS token (`num_prefix_tokens > 1`). All spatial operations (pruning, forecasting, embedding extraction) use `x[:, num_prefix_tokens:]` to exclude them correctly.

**`embed_dim` from adapter** — the forecaster and classification head always use `adapter.embed_dim` (the raw CLS token dimension) rather than Thunder's `emb_dim` config value, which may differ (e.g. Virchow concatenates CLS + avg-pooled patches for its Thunder embedding).
