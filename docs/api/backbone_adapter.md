# API — BackboneAdapter

**Module:** `src/models/backbone_adapter.py`  
**Import:** `from src.models.backbone_adapter import BackboneAdapter`

---

## `BackboneAdapter`

```python
BackboneAdapter(model: torch.nn.Module)
```

Wraps a timm `VisionTransformer` loaded via TRIDENT's `encoder_factory()` and exposes architecture metadata needed by the EAF pipeline.

The adapter normalizes access to backbone properties across different foundation models (UNI, Virchow, Hoptimus, etc.), handling differences in embedding dimensions, block counts, and prefix token conventions (CLS + register tokens).

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `torch.nn.Module` | A timm `VisionTransformer` instance, typically loaded via TRIDENT's `encoder_factory()` |

### Properties

| Property | Type | Source | Description |
|---|---|---|---|
| `model` | `nn.Module` | — | Reference to the wrapped timm backbone |
| `embed_dim` | `int` | `model.embed_dim` | Token embedding dimension (CLS, patches, etc.). Determines forecaster input size. |
| `n_blocks` | `int` | `len(model.blocks)` | Number of transformer blocks in the network. |
| `n_patches` | `int` | `model.patch_embed.num_patches` | Number of spatial patches (excludes CLS and register tokens). |
| `num_prefix_tokens` | `int` | `model.num_prefix_tokens` | Count of prefix tokens (CLS + register tokens). Always at the start of the sequence. |

### Methods

#### `get_blocks() → torch.nn.ModuleList`

Returns the list of transformer blocks from the backbone.

```python
blocks = adapter.get_blocks()
# Use in loops
for layer_idx, block in enumerate(blocks):
    x = block(x)
```

Equivalent to `model.blocks`.

### Properties at a glance

Quick reference for supported encoders:

| Encoder | `embed_dim` | `n_blocks` | `n_patches` | `num_prefix_tokens` |
|---|---|---|---|---|
| `uni_v1` | 1024 | 24 | 196 | 1 (CLS) |
| `virchow` | 1280 | 32 | 256 | 1 (CLS) |
| `virchow2` | 1280 | 32 | 256 | 5 (CLS + 4 regs) |
| `hoptimus0` | 1536 | 40 | 256 | 5 (CLS + 4 regs) |
| `hiboul` | 1536 | 40 | 256 | 5 (CLS + 4 regs) |

### Usage Example

```python
from src.models.backbone_adapter import BackboneAdapter
from trident.patch_encoder_models import encoder_factory

# Load encoder via TRIDENT
enc = encoder_factory("uni_v1")
backbone = enc.model.to(device)

# Wrap with adapter
adapter = BackboneAdapter(backbone)

# Access architecture info
print(f"Embed dim: {adapter.embed_dim}")        # 1024
print(f"Blocks: {adapter.n_blocks}")            # 24
print(f"Patches: {adapter.n_patches}")          # 196
print(f"Prefix tokens: {adapter.num_prefix_tokens}")  # 1

# Access blocks for forward pass
blocks = adapter.get_blocks()
for layer_idx, block in enumerate(blocks):
    x = block(x)
    if layer_idx == 4:  # prune_layer
        features = x[:, adapter.num_prefix_tokens:, :]  # extract patches only
        break

# Use with forecaster
from src.models import AttentionForecaster

forecaster = AttentionForecaster(embed_dim=adapter.embed_dim, ...)
```

### Design rationale

- **Normalization:** Different encoders have different prefix token conventions. The adapter normalizes this so downstream code (forecaster, pruning) doesn't need model-specific logic.
- **Forward-compatible:** If a new encoder is added to TRIDENT, only the validation in `BackboneAdapter.__init__` needs to be extended; the rest of the pipeline works unchanged.
- **Minimal overhead:** The adapter is a thin wrapper with no learnable parameters or forward pass — it's purely a metadata interface.

### Prefix tokens (CLS + registers)

Some newer models (e.g., Virchow2, Hoptimus) prepend extra "register" tokens along with the CLS token:

```
x = [CLS, REG1, REG2, ..., patch1, patch2, ..., patchN]
     ^                    ^
     num_prefix_tokens    start of spatial patches

# When extracting patch embeddings (excluding prefix tokens):
patches = x[:, adapter.num_prefix_tokens:, :]
```

The forecaster and pruning logic always exclude prefix tokens by using `x[:, adapter.num_prefix_tokens:, :]`.

### Validation

The adapter validates that the input model has the required attributes:

```python
try:
    adapter = BackboneAdapter(model)
except AttributeError as e:
    print(f"Model does not have required attribute: {e}")
```

Required attributes:
- `model.blocks` (torch.nn.ModuleList)
- `model.embed_dim` (int)
- `model.patch_embed.num_patches` (int)
- `model.num_prefix_tokens` (int)

If your model lacks these, it won't be compatible with EAF.

### Integration with Phase 1, 2, 3

**Phase 1 (forecaster training):**
- Adapter is used to determine `embed_dim` for forecaster initialization
- `get_blocks()` used for hook registration at prune and target layers

**Phase 2 (distillation):**
- Adapter needed to initialize `GenericLoRAWithForecasterPruning`
- `num_prefix_tokens` ensures pruning doesn't affect CLS/register tokens

**Phase 3 (evaluation):**
- Adapter's `embed_dim` used for embedding aggregation (mean pooling over tiles)
- Property values logged for reproducibility
