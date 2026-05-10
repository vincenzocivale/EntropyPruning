# Architecture

## Overview

EAF is a three-phase pruning framework for WSI classification:

1. **Phase 1** — Train an **AttentionForecaster** to predict which patch tokens are important by learning to replicate the final-block CLS-to-patch attention weights from intermediate-layer embeddings.
2. **Phase 2** — Fine-tune the pruned backbone via knowledge distillation, matching CLS embeddings to the non-pruned encoder while keeping top-k tokens.
3. **Phase 3** — Evaluate on full WSI slides: extract tile embeddings, aggregate to slide level, and measure classification accuracy.

```
Phase 1: Tile-level forecaster training (self-supervised)
    Input: WSI tiles
    ↓
    Backbone (frozen) → intermediate embeddings → AttentionForecaster → importance scores
    ↓
    Loss: KL divergence(predicted attention, ground-truth CLS attention)
    Output: forecaster.pt

Phase 2: Pruned model distillation (LoRA fine-tuning on tiles)
    Input: Forecaster + WSI tiles
    ↓
    Student (pruned + LoRA) vs Teacher (full model)
    Loss: cosine_distance(student CLS embedding, teacher CLS embedding)
    Output: best_pruned.pt

Phase 3: Slide-level evaluation
    Input: Pruned model + full WSI slides + labels
    ↓
    Extract embeddings per tile → Mean-pool to slide level → Linear classifier
    Output: accuracy, F1, AUC on Patho-Bench ground truth
```

---

## Phase 1 — AttentionForecaster Training

### Goal
Train a lightweight MLP-based transformer that predicts the importance of each patch token, as measured by its CLS-attention weight at the final transformer block.

### Data flow
```
WSI file
    ↓ (TRIDENT: WSIPatcher + Otsu segmentation)
    Tiles sampled on-the-fly (no caching)
    ↓
    Foundation model backbone (frozen, device=cuda)
    ├─ Blocks 0 → prune_layer (run forward)
    └─ Register hook at prune_layer to capture patch embeddings (B, N_patches, D)
    ├─ Register hook at target_layer to capture CLS-to-patch attention (B, N_patches)
    ↓
    AttentionForecaster forward
    ├─ Input: patch embeddings from prune_layer
    └─ Output: predicted importance scores (B, N_patches)
    ↓
    KL divergence loss w.r.t. ground-truth attention from target_layer
    ↓
    Backprop → Update forecaster only (backbone frozen)
```

### Validation strategy
- **Per epoch**, not per WSI
- Validation loss computed on all validation tile batches
- Spearman rank correlation measured (how well predicted ranking matches true ranking)
- Early stopping based on validation loss

### Implementation
- **Script:** `scripts/wsi_train_forecaster.py`
- **Input args:**
  - `--encoder {uni_v1, virchow, ...}` — TRIDENT encoder name
  - `--wsi-dir /path/to/wsis` — directory with `.svs`, `.ndpi`, etc.
  - `--prune-layer {int}` — source layer for embeddings (typically 4–8 for ViT-L)
  - `--target-layer {int}` — target layer for attention (default: n_blocks - 1)
  - `--epochs {int}` — training epochs (default: 30)
- **Output:**
  - `forecaster_{encoder}_src{prune_layer}_tgt{target_layer}.pt`
  - `results.json` with train/val metrics

---

## Phase 2 — Pruned Model Distillation

### Goal
Fine-tune the pruned backbone (with LoRA) to match CLS embeddings from the non-pruned teacher while maintaining the forecaster-based pruning.

### Data flow
```
Teacher (non-pruned, frozen)          Student (pruned + LoRA, trainable)
    ↓                                      ↓
    WSI tiles (same batch)
    ↓                                      ↓
    Forward all blocks                     Forward up to prune_layer
    ↓                                      ↓
    Extract CLS token (B, D)               Apply forecaster pruning
    ↓                                      ↓
    (no gradient)                          Forward remaining blocks (prune_layer+1 → end)
                                          ↓
                                          Extract CLS token (B, D)
    ↓                                      ↓
    ────────────────────────────────────────
    Loss: 1 - cosine_similarity(teacher_cls, student_cls)
    ↓
    Backprop → Update LoRA params only
```

### Validation strategy
- **Per epoch**, on validation tile batches
- Loss should decrease over time
- Best checkpoint saved when validation loss improves
- No per-WSI evaluation (same as Phase 1)

### Implementation
- **Script:** `scripts/wsi_distill_pruned.py`
- **Input args:**
  - `--encoder {uni_v1, ...}`
  - `--wsi-dir /path/to/wsis`
  - `--forecaster-ckpt /path/to/forecaster.pt` — Phase 1 output
  - `--prune-layer {int}` — must match Phase 1
  - `--keep-ratio {float}` — fraction of tokens to retain (e.g., 0.5 = 50% kept)
  - `--epochs {int}` — training epochs (default: 20)
- **Output:**
  - `best_{encoder}_prune{layer}_keep{ratio}.pt`
  - `results_phase2.json` with distillation loss

---

## Phase 3 — WSI-Level Evaluation

### Goal
Evaluate the pruned encoder on full WSI slides. Extract embeddings for all tiles, aggregate to slide level, train a linear classifier on aggregated embeddings, and measure accuracy against Patho-Bench ground truth.

### Data flow
```
Full WSI file (high-res)
    ↓
    TRIDENT: Otsu segmentation → WSIPatcher (all tissue-covered tiles)
    ↓
    Batch process tiles through pruned encoder:
    ├─ Apply pruning (forecaster selects top-k patches)
    ├─ Extract CLS embedding (B, D)
    ↓
    Collect all CLS embeddings for this slide → (n_tiles, D)
    ↓
    Mean-pool → slide-level embedding (D,)
    ↓
    Linear logistic regression classifier
    ├─ Train: on training slides (mean-pooled embeddings)
    ├─ Eval: on validation/test slides
    ↓
    Metrics: accuracy, F1, AUC vs Patho-Bench labels
```

### Implementation
- **Script:** `scripts/wsi_evaluate.py`
- **Input args:**
  - `--encoder {uni_v1, ...}`
  - `--dataset {TCGA-BRCA, BACH, ...}` — Patho-Bench dataset
  - `--task {subtype, ...}` — classification task for this dataset
  - `--checkpoint /path/to/pruned_model.pt` — Phase 2 output
  - `--mag {int}` — magnification (default: 20)
- **Output:**
  - `predictions.csv` with per-slide predictions
  - `metrics.json` with accuracy, F1, AUC
  - Plots (if enabled)

---

## Class Hierarchy

### AttentionForecaster
```
AttentionForecaster(nn.Module)
  ├─ Linear(embed_dim → hidden)
  ├─ TransformerEncoder (n_layers)
  │   └─ MultiheadAttention (self-attention over patches)
  ├─ MultiheadAttention (cross-att: learnable CLS queries → patches)
  └─ MLP head → (B, N) importance scores
```

**Training:** KL divergence against ground-truth CLS attention weights.  
**Evaluation:** Spearman rank correlation (how well ranking is preserved).

### BackboneAdapter
Wraps a timm `VisionTransformer` from TRIDENT:
```
BackboneAdapter(nn.Module)
  ├─ model: timm.VisionTransformer
  └─ Properties:
      ├─ embed_dim: token embedding dimension
      ├─ n_blocks: number of transformer blocks
      ├─ n_patches: number of spatial patches
      ├─ num_prefix_tokens: CLS + register tokens
      └─ get_blocks(): accessor for block-level hooks
```

### GenericLoRAWithForecasterPruning
Combines pruned backbone with forecaster for inference:
```
GenericLoRAWithForecasterPruning(nn.Module)
  ├─ backbone: TRIDENT encoder with LoRA adapters
  ├─ forecaster: AttentionForecaster (frozen after Phase 1)
  ├─ head: classification head (not used during Phase 2 distillation)
  └─ forward(x) with pruning:
      1. x (B, N_total, D) through backbone.blocks[0:prune_layer]
      2. Forecaster scores top-k important patches
      3. Hard mask: keep only top-k
      4. Forward through backbone.blocks[prune_layer+1:end]
      5. CLS token (and optionally head) output
```

---

## Data Flow: TRIDENT Integration

### Encoder Loading
```python
from trident.patch_encoder_models import encoder_factory

enc = encoder_factory("uni_v1")  # Returns EncoderOutput namedtuple
backbone = enc.model              # timm.VisionTransformer
transform = enc.eval_transforms   # Preprocessing pipeline
```

### WSI Loading and Tiling
```python
from trident import load_wsi
from trident.wsi_objects.WSIPatcher import WSIPatcher
from trident.segmentation_models import segmentation_model_factory

wsi = load_wsi("slide.svs")
otsu = segmentation_model_factory("otsu")
mask_gdf = wsi.segment_tissue(otsu, target_mag=1.25)

patcher = WSIPatcher(
    wsi,
    patch_size=256,     # output tile size in pixels
    dst_mag=20,         # magnification
    mask=mask_gdf,      # tissue mask
    pil=True            # return PIL images
)

# Iterate over patches
for i in range(len(patcher)):
    tile, coords, region_props = patcher[i]  # PIL image
```

### Embedding Extraction
```python
# Transform and encode
tile_tensor = transform(tile).unsqueeze(0)  # (1, 3, H, W)

# Forward through frozen encoder
with torch.no_grad():
    cls_emb = enc.model(tile_tensor)[..., 0, :]  # (1, D)
```

---

## Key Design Decisions

### 1. **Tile-based training (not slide-level caching)**
- Phases 1 & 2 sample tiles on-the-fly from WSI files each epoch
- No HDF5 caching (unlike prior THUNDER-based pipelines)
- Pros: lower memory, infinite diversity, easier to scale
- Cons: slower I/O; requires WSI directory at training time

### 2. **Frozen backbone in Phase 1**
- Backbone is purely frozen; only forecaster learns
- Forecaster is lightweight (few parameters)
- Enables self-supervised training (no labels needed)

### 3. **LoRA-only adaptation in Phase 2**
- Pruned model uses LoRA on backbone (independent of initial encoding strategy)
- Keeps parameter count low for distillation
- Forecaster remains frozen

### 4. **Straight-through estimator for gradients**
- During Phase 2 training, pruning selection is hard (binary mask)
- During backprop, mask is treated as differentiable for LoRA updates
- At inference, hard pruning (no approximation)

### 5. **Prefix tokens (CLS + registers) always kept**
- CLS token is essential for classification
- Register tokens (in some encoders) are always retained
- Pruning applies only to spatial patch tokens

---

## Attention Hook Mechanics (Phase 1)

The forecaster is trained by replicating the transformer's attention computation inside a hook:

```python
def make_tgt_hook(layer_idx):
    def hook(m, x, y):
        # m: Attention module at layer_idx
        # x: input to attention (B, N, C)
        x_in = x[0] if isinstance(x, tuple) else x
        B, N, C = x_in.shape
        
        # Recompute attention (same as module, but with QK norm for v2 models)
        qkv = m.qkv(x_in).reshape(B, N, 3, m.num_heads, m.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = m.q_norm(q), m.k_norm(k)  # For ViT-v2
        
        attn = (q @ k.transpose(-2, -1)) * m.scale
        attn = attn.softmax(-1)
        
        # Extract CLS-to-patch attention (index 0 = CLS token)
        cls_attn = attn[:, :, 0, num_prefix:].mean(1)  # Mean over heads
        cache[f"attn_{layer_idx}"] = cls_attn.detach().cpu()
    
    return hook
```

This ensures Phase 1 training sees the exact same attention distribution as the full model's final block.
