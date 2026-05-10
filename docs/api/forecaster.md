# API — AttentionForecaster

**Module:** `src/models/forecaster.py`  
**Import:** `from src.models import AttentionForecaster`

---

## `AttentionForecaster`

```python
AttentionForecaster(
    embed_dim: int = 1024,
    hidden: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.2,
)
```

A lightweight transformer-based attention predictor that learns to approximate the CLS-to-patch attention weights at a target layer, using only embeddings from a source layer.

Given patch embeddings from an early backbone layer, it outputs an importance score for each patch that correlates with the CLS token's attention to that patch at the final block.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `embed_dim` | 1024 | Dimension of input patch embeddings (from source layer). Must match `adapter.embed_dim`. |
| `hidden` | 256 | Hidden dimension for internal transformer layers. |
| `n_heads` | 4 | Number of attention heads in self- and cross-attention. |
| `n_layers` | 2 | Number of `(self-attention, cross-attention)` layer pairs. |
| `dropout` | 0.2 | Dropout rate throughout the model. |

### Architecture

```
Input: patch_embeddings (B, N_patches, embed_dim)
    │
Linear(embed_dim → hidden)                      — input projection
    │
[TransformerEncoderLayer] × n_layers
    │ Self-attention: patches attend to patches
    │
[MultiheadAttention] × n_layers                 — Cross-attention:
    │    learnable_queries: (1, 1, hidden)      — learned CLS token
    │    keys/values: patch features
    │
Concatenate [self_out, cross_out]               — (B, 1, hidden*2)
    │
Linear(hidden*2 → 128) → GELU → Dropout → Linear(128 → 1)
    │
Squeeze → importance_scores (B, N_patches)
```

### Methods

#### `forward`

```python
def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor
```

**Input:**
- `patch_embeddings` — Tensor of shape `(B, N_patches, embed_dim)`
  - Spatial patches only (CLS and register tokens already excluded)
  - Typically extracted via hook at `prune_layer` in Phase 1

**Output:**
- `scores` — Tensor of shape `(B, N_patches)`
  - Per-patch importance scores (unnormalized logits)
  - Higher scores indicate more important patches

### Training

The forecaster is trained in Phase 1 with KL divergence against ground-truth CLS attention:

```python
# Ground truth: CLS-to-patch attention at target_layer
target_attn = cache[f"attn_{target_layer}"]  # (B, N_patches)

# Prediction
pred_scores = forecaster(patch_embeddings)   # (B, N_patches)

# Loss
loss = F.kl_div(
    pred_scores.log_softmax(dim=-1),
    target_attn.softmax(dim=-1),
    reduction="batchmean"
)
```

### Evaluation

During validation and testing, the forecaster's ranking quality is measured by **Spearman rank correlation** (`rho`):

```python
def spearman_correlation(y_pred, y_true) -> float:
    """Measure how well predicted ranking matches true ranking."""
    r_pred = y_pred.argsort().argsort()         # rank indices
    r_true = y_true.argsort().argsort()
    # compute correlation on ranks...
    return correlation
```

- **Perfect ranking:** `rho = 1.0` (predicted order matches true order exactly)
- **Random ranking:** `rho ≈ 0.0`
- **Baseline (norm):** `rho ≈ 0.2–0.4` (ranking by embedding norm)
- **Good forecaster:** `rho > 0.6` (strong correlation with true importance)

### Usage Example

```python
from src.models import AttentionForecaster, BackboneAdapter
from trident.patch_encoder_models import encoder_factory

# Load encoder
enc = encoder_factory("uni_v1")
backbone = enc.model.to(device)
adapter = BackboneAdapter(backbone)

# Create forecaster
forecaster = AttentionForecaster(
    embed_dim=adapter.embed_dim,  # 1024 for UNI
    hidden=256,
    n_heads=4,
    n_layers=2,
).to(device)

# During Phase 1 training:
# - Input: patch embeddings from source layer (prune_layer)
patch_emb = ...  # shape (B, N_patches, embed_dim)

# - Output: importance scores
scores = forecaster(patch_emb)  # shape (B, N_patches)

# - Compare with ground truth (captured from target layer)
target_attn = ...  # shape (B, N_patches)
loss = F.kl_div(scores.log_softmax(-1), target_attn.softmax(-1))
```

### Checkpoint Format

Saved as: `{output_dir}/forecaster_{encoder}_{src_layer:02d}_{tgt_layer:02d}.pt`

Example: `forecaster_uni_v1_src04_tgt23.pt`

**Important:** A forecaster is tied to its training configuration:
- `embed_dim` must match the source layer's embedding dimension
- It cannot be transferred across different backbones or source layers
- Train a separate forecaster for each `(source_layer, target_layer)` pair

### Implementation details

- The forecaster operates on **spatial patches only** — CLS and register tokens are excluded
- Input validation: embeddings should have shape `(B, N, D)` where `N = adapter.n_patches`
- No position encoding: the forecaster relies on self-attention to learn spatial relationships
- Fully differentiable: can be integrated into end-to-end pipelines (though Phase 1 keeps backbone frozen)
