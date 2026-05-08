# API — ThunderBackboneAdapter

**Module:** `src/models/backbone_adapter.py`
**Import:** `from src.models import ThunderBackboneAdapter`

---

## `ThunderBackboneAdapter`

```python
ThunderBackboneAdapter(model: nn.Module)
```

Wraps a raw timm `VisionTransformer` returned by Thunder's `get_model_from_name` and exposes architecture metadata needed by the rest of the EAF pipeline.

Raises `NotImplementedError` for HuggingFace-based models (phikon, hibou).

### Attributes

| Attribute | Type | Source attribute | Description |
|---|---|---|---|
| `model` | `nn.Module` | — | Reference to the wrapped timm model. |
| `embed_dim` | `int` | `model.embed_dim` | CLS token dimension (e.g. 1024 for UNI). |
| `n_blocks` | `int` | `len(model.blocks)` | Number of transformer blocks. |
| `n_patches` | `int` | `model.patch_embed.num_patches` | Spatial patch count (excl. prefix tokens). |
| `num_prefix_tokens` | `int` | `model.num_prefix_tokens` | CLS + register token count prepended to the sequence. |

### Methods

```python
adapter.get_blocks()               # → model.blocks
adapter.get_attn_module(block_idx) # → model.blocks[block_idx].attn  (validates .qkv exists)
```

### Detection

A model is accepted if it has all of:
- `model.blocks`
- `model.embed_dim`
- `model.patch_embed.num_patches`
- `model.num_prefix_tokens`

### Example

```python
raw_backbone, transform, _ = get_model_from_name("hoptimus0", "cuda")
adapter = ThunderBackboneAdapter(raw_backbone)

print(adapter.embed_dim)          # 1536
print(adapter.n_blocks)           # 40
print(adapter.n_patches)          # 256
print(adapter.num_prefix_tokens)  # 5  (1 CLS + 4 register tokens)
```

### Reference values

| Model | `embed_dim` | `n_blocks` | `n_patches` | `num_prefix_tokens` |
|---|---|---|---|---|
| `uni` | 1024 | 24 | 196 | 1 |
| `uni2h` | 1536 | 24 | 256 | 9 |
| `hoptimus0`, `hoptimus1` | 1536 | 40 | 256 | 5 |
| `virchow` | 1280 | 32 | 256 | 1 |
| `virchow2` | 1280 | 32 | 256 | 5 |
| `kaiko_vitb16` | 768 | 12 | 196 | 1 |
| `kaiko_vitb8` | 768 | 12 | 784 | 1 |
| `dinov2base` | 768 | 12 | 256 | 1 |
| `dinov2large` | 1024 | 24 | 256 | 1 |
