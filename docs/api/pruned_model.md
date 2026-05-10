# API — GenericLoRAWithForecasterPruning

**Module:** `src/models/pruned_classifier.py`  
**Import:** `from src.models import GenericLoRAWithForecasterPruning`

---

## `GenericLoRAWithForecasterPruning`

```python
GenericLoRAWithForecasterPruning(
    backbone: torch.nn.Module,
    adapter: BackboneAdapter,
    n_classes: int,
    forecaster: torch.nn.Module,
    prune_layer: int,
    keep_ratio: float,
    lora_r: int = 8,
    lora_alpha: int = 32,
)
```

Phase 2 model used in pruned model distillation. Wraps a TRIDENT backbone with LoRA adapters and a frozen `AttentionForecaster`. Applies token pruning at a specified layer and forward-passes the reduced token sequence through remaining layers.

The backbone **always uses LoRA** in this phase, regardless of Phase 1 strategy. The forecaster is frozen and acts as a static pruning scorer.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backbone` | `nn.Module` | — | Raw timm `VisionTransformer` (from TRIDENT's `encoder_factory`) |
| `adapter` | `BackboneAdapter` | — | Adapter wrapping the backbone (provides `embed_dim`, `n_blocks`, `num_prefix_tokens`) |
| `n_classes` | `int` | — | Number of output classes. Not used during distillation (Phase 2) but needed for classification in Phase 3. |
| `forecaster` | `nn.Module` | — | Trained `AttentionForecaster`. **Must be frozen before passing in** (no gradients). |
| `prune_layer` | `int` | — | Block index where pruning is applied (0-indexed). Must be < `adapter.n_blocks`. |
| `keep_ratio` | `float` | — | Fraction of spatial patch tokens to retain (e.g., `0.5` = keep 50% of patches, prune 50%). |
| `lora_r` | `int` | 8 | LoRA rank (number of low-rank updates). Higher = more capacity but slower. |
| `lora_alpha` | `int` | 32 | LoRA scaling factor. Higher = stronger LoRA effects. |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `backbone` | PEFT `LoraModel` | The backbone wrapped with LoRA adapters. |
| `raw_backbone` | `nn.Module` | Direct reference to the underlying timm `VisionTransformer` (`self.backbone.model`). Used for block-level access. |
| `adapter` | `BackboneAdapter` | Architecture metadata. |
| `forecaster` | `AttentionForecaster` | Frozen importance scorer. |
| `head` | `nn.Sequential` | Classification head: `LayerNorm → Dropout → Linear(embed_dim → n_classes)` |

### Forward Pass

The forward pass implements token pruning at `prune_layer`:

```
x (B, N_total, D) — input tiles (e.g., transformed image tensors)
    │
[Forward blocks 0 → prune_layer-1]
    │
x (B, N_total, D) at prune_layer
    │
    ├─ Split prefix: prefix = x[:, :num_prefix_tokens, :]  (CLS + registers, always kept)
    │
    ├─ Extract patches: patches = x[:, num_prefix_tokens:, :]
    │
    ├─ Score patches: scores = forecaster(patches)  (B, N_patches), no grad
    │
    ├─ Keep top-k: k = max(1, int(N_patches * keep_ratio))
    │   - Training: soft mask via sigmoid (straight-through estimator for gradients)
    │   - Inference: hard selection of top-k indices
    │
    └─ Prune: kept_patches = patches[kept_idx]  (B, k, D)
    │
    Merge: x_pruned = cat([prefix, kept_patches], dim=1)  (B, num_prefix_tokens+k, D)
    │
[Forward blocks prune_layer+1 → end]
    │
x_final (B, num_prefix_tokens+k, D)
    │
Extract CLS: cls_token = x_final[:, 0, :]  (B, D)
    │
[Classification head] → logits (B, n_classes)
```

**Key points:**
- **Prefix tokens always kept:** CLS and register tokens are never pruned
- **Straight-through estimator during training:** Pruning is hard (binary selection) but gradients flow for LoRA updates
- **Hard pruning at inference:** No approximation; exact top-k selection

### Example: Phase 2 Distillation

```python
from src.models import AttentionForecaster, GenericLoRAWithForecasterPruning, BackboneAdapter
from trident.patch_encoder_models import encoder_factory

device = "cuda"

# Load encoder
enc = encoder_factory("uni_v1")
backbone = enc.model.to(device)
transform = enc.eval_transforms
adapter = BackboneAdapter(backbone)

# Load and freeze forecaster (from Phase 1)
forecaster = AttentionForecaster(embed_dim=adapter.embed_dim).to(device)
forecaster.load_state_dict(torch.load("path/to/forecaster.pt", map_location=device))
forecaster.eval()
for p in forecaster.parameters():
    p.requires_grad_(False)

# Create pruned student model
student = GenericLoRAWithForecasterPruning(
    backbone=backbone,
    adapter=adapter,
    n_classes=2,  # dummy; not used in distillation
    forecaster=forecaster,
    prune_layer=4,
    keep_ratio=0.5,
    lora_r=8,
    lora_alpha=32,
).to(device)

# During Phase 2 training:
# - Teacher (non-pruned) forward
teacher_x = backbone(tiles)  # forward all blocks
teacher_cls = teacher_x[:, 0, :]  # CLS token

# - Student (pruned) forward
student_x = student(tiles)  # forward with pruning at layer 4
# For distillation, we extract CLS from the pruned model
# This is typically done inside the student's forward or by custom logic

# - Loss: cosine similarity between CLS embeddings
loss = 1.0 - F.cosine_similarity(teacher_cls, student_cls).mean()

# - Backprop updates LoRA params only
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Example: Phase 3 Inference

```python
# Load checkpoint from Phase 2
checkpoint = torch.load("path/to/best_pruned.pt")
student.load_state_dict(checkpoint)
student.eval()

# Inference on new tiles
with torch.no_grad():
    # Forward with pruning
    logits = student(tiles)  # (B, n_classes)
    probs = F.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
```

### Checkpoint Loading

Checkpoints saved from Phase 2 include both LoRA weights and (frozen) forecaster weights.

```python
# Load Phase 2 checkpoint
state_dict = torch.load("best_pruned.pt")

# The state dict contains:
# - backbone.base_model.model.* (timm weights + LoRA)
# - forecaster.* (frozen, but included)
# - head.* (classification head)

student.load_state_dict(state_dict)
```

### Token Reduction Impact

For a ViT-L with 196 spatial patches:

| `keep_ratio` | Patches kept | Sequence length post-prune | FLOP reduction |
|---|---|---|---|
| 1.0 | 196 | 197 (196 + 1 CLS) | None (baseline) |
| 0.9 | 176 | 177 | ~17% |
| 0.7 | 137 | 138 | ~30% |
| 0.5 | 98 | 99 | ~49% |
| 0.3 | 59 | 60 | ~69% |

Quadratic layers (attention, fully-connected) benefit most from token reduction.

### Pruning mechanics

The forecaster outputs scores for each spatial patch. The top-k patches by score are selected:

```python
# In pruning forward hook:
scores = forecaster(patches)          # (B, N_patches)
k = max(1, int(N_patches * keep_ratio))
top_k_indices = torch.topk(scores, k, dim=1)[1]  # (B, k)

# Hard selection at inference
kept_patches = patches[batch_idx, top_k_indices[batch_idx]]
```

During training (Phase 2), a soft mask (sigmoid) is applied for differentiability, but at inference, pruning is hard and exact.

### Integration with distillation

Phase 2 trains `GenericLoRAWithForecasterPruning` as the student against a frozen non-pruned teacher:

**Teacher:** Full backbone (no pruning)  
**Student:** This model (with pruning at `prune_layer`)  
**Loss:** Cosine distance between CLS embeddings

The forecaster is frozen during Phase 2, so only LoRA params update. This forces the LoRA layers to adapt the pruned representation to match the full representation.

### Hyperparameter tuning

- **`keep_ratio`:** Controls aggressiveness of pruning. Lower = faster but riskier. Typical range: 0.3–0.9.
- **`prune_layer`:** Earlier layer = more aggressive (smaller effective sequence length). Typical range: 2–8 for ViT-L (24 blocks).
- **`lora_r`:** LoRA rank. Higher = more expressive but slower. Typical: 4–16.

A good starting point: `keep_ratio=0.5`, `prune_layer=4–6`, `lora_r=8`.
