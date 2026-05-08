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
    dropout: float = 0.1,
)
```

Predicts per-patch importance scores from intermediate patch embeddings.
Given patch embeddings from an early backbone layer, it outputs a score for each patch that approximates the CLS-to-patch attention weight at a later (typically final) layer.

### Parameters

| Argument | Default | Description |
|---|---|---|
| `embed_dim` | 1024 | Dimension of the input patch embeddings. **Must match `adapter.embed_dim`** of the backbone used in Phase 2 feature extraction. |
| `hidden` | 256 | Internal transformer dimension. |
| `n_heads` | 4 | Number of attention heads in self- and cross-attention layers. |
| `n_layers` | 2 | Number of `(self-attention, cross-attention)` layer pairs. |
| `dropout` | 0.1 | Dropout rate throughout the module. |

### Architecture

```
Input: patch_embeddings (B, N, embed_dim)
    │
Linear(embed_dim → hidden)                     — input projection
    │
[TransformerEncoderLayer(hidden, n_heads)] × n_layers   — self-attention over patches
    │
[MultiheadAttention(hidden, n_heads)]  × n_layers       — cross-attention:
    │    queries = learnable CLS token (1, 1, hidden)         learned from data
    │    keys/values = patch features
    │
Concat([self-attn out, cross-attn out], dim=-1)         — (B, 1, hidden*2)
    │
Linear(hidden*2 → 128) → GELU → Dropout → Linear(128 → 1)
    │
Squeeze → scores (B, N)
```

### `forward`

```python
def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor
```

- **Input:** `patch_embeddings` — shape `(B, N, embed_dim)`, spatial patches only (no CLS/register tokens)
- **Output:** `scores` — shape `(B, N)`, per-patch importance scores (unnormalised)

### Training objective

KL divergence between the softmax of the predicted scores and the ground-truth CLS attention distribution:

```python
loss = F.kl_div(scores.log_softmax(-1), target_attention, reduction='batchmean')
```

`target_attention` is the mean CLS-to-patch attention over all heads at the target layer, stored in the Phase 2 HDF5 cache.

### Evaluation metric

Spearman rank correlation (`val/rho`) between predicted scores and ground-truth attention.
A model that perfectly ranks patches by their CLS attention weight gets `rho = 1.0`.
Baseline: rank by patch embedding norm — typically `rho ≈ 0.2–0.4`.
Good forecaster: `rho > 0.5`.

### Example

```python
from src.models import AttentionForecaster, ThunderBackboneAdapter
from thunder.models.pretrained_models import get_model_from_name

_, _, _ = get_model_from_name("uni", "cpu")  # just to get adapter
adapter = ThunderBackboneAdapter(_[0] if False else raw_backbone)

forecaster = AttentionForecaster(
    embed_dim=adapter.embed_dim,  # 1024 for UNI
    hidden=256,
    n_heads=4,
    n_layers=2,
).to(device)

# During Phase 2 inference:
patch_emb = ...  # (B, 196, 1024) from H5ForecastDataset
scores = forecaster(patch_emb)  # (B, 196)
```

### Checkpoint

Saved as `checkpoints/{dataset}/{model}_forecaster/forecaster_src{L:02d}_tgt{T:02d}.pt`

One checkpoint per `(source_layer, target_layer)` pair. A forecaster is tied to a specific `embed_dim` — it cannot be loaded for a different backbone.
