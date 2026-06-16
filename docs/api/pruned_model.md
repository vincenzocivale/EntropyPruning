# API — Pruned Models

**Module:** `src/models/pruned_classifier.py`
**Import:** `from src.models import GenericLoRAWithForecasterPruning, FrozenPrunedLinearProbe`

---

## `GenericLoRAWithForecasterPruning`

```python
GenericLoRAWithForecasterPruning(
    backbone: nn.Module,
    adapter: ThunderBackboneAdapter,
    n_classes: int,
    forecaster: nn.Module,
    prune_layer: int,
    keep_ratio: float,
    lora_r: int = 8,
    lora_alpha: int = 32,
    dropout: float = 0.1,
)
```

Phase 3 model. Wraps a backbone with LoRA adapters and a frozen `AttentionForecaster`.
At each forward pass, patches are scored and pruned at `prune_layer` before the remaining blocks are executed.

The backbone always uses LoRA regardless of the Phase 1 adaptation strategy.

### Parameters

| Argument | Description |
|---|---|
| `backbone` | Raw timm model from `get_model_from_name`. |
| `adapter` | `ThunderBackboneAdapter` for the same backbone. |
| `n_classes` | Number of output classes. |
| `forecaster` | Trained `AttentionForecaster`, **must be frozen before passing in**. |
| `prune_layer` | Block index where pruning is applied (0-indexed). Must be < `adapter.n_blocks`. |
| `keep_ratio` | Fraction of spatial patch tokens to keep (e.g. `0.1` = 10%). |
| `lora_r`, `lora_alpha` | LoRA hyperparameters. |
| `dropout` | Classification head dropout. |

### Attributes

| Attribute | Description |
|---|---|
| `backbone` | peft `LoraModel` wrapping the timm model. |
| `raw_backbone` | `self.backbone.model` — timm model (property). |
| `head` | `LayerNorm → Dropout → Linear`. |
| `forecaster` | Frozen `AttentionForecaster`. |

### Pruning forward pass

At `prune_layer`, the standard block forward is replaced by a hook that:

1. Runs the original block: `x = orig_block(x)`
2. Splits tokens: `prefix = x[:, :num_prefix_tokens]` (CLS + registers), `patches = x[:, num_prefix_tokens:]`
3. Scores patches: `scores = forecaster(patches)` — no grad
4. Keeps top-`k` patches: `k = max(1, int(N * keep_ratio))`
5. **Training:** soft mask via sigmoid (straight-through estimator for differentiability)
6. **Inference:** hard selection of top-k indices
7. Returns `cat([prefix, kept_patches], dim=1)` — shortened sequence

All subsequent blocks operate on the reduced sequence, reducing FLOP count quadratically with the sequence length.

### Loading Phase 1 checkpoint

```python
model = GenericLoRAWithForecasterPruning(
    backbone=raw_backbone, adapter=adapter,
    n_classes=n_classes, forecaster=forecaster,
    prune_layer=2, keep_ratio=0.1,
)
# Load Phase 1 LoRA weights (strict=False because forecaster keys are new)
missing, unexpected = model.load_state_dict(
    torch.load("checkpoints/.../best_model.pt", map_location=device),
    strict=False,
)
```

If Phase 1 used `LoRAClassifier`, backbone key names match exactly.
If Phase 1 used a non-LoRA strategy, the backbone starts from pretrained weights (Phase 1 head weights are loaded).

### Example

```python
from src.models import AttentionForecaster, GenericLoRAWithForecasterPruning, ThunderBackboneAdapter
from thunder.models.pretrained_models import get_model_from_name

raw_backbone, _, _ = get_model_from_name("uni", "cuda")
adapter = ThunderBackboneAdapter(raw_backbone)

# Load and freeze forecaster
forecaster = AttentionForecaster(embed_dim=adapter.embed_dim).to(device)
forecaster.load_state_dict(torch.load("forecaster_src02_tgt23.pt"))
forecaster.eval()
for p in forecaster.parameters():
    p.requires_grad_(False)

# Build pruned model
model = GenericLoRAWithForecasterPruning(
    backbone=raw_backbone, adapter=adapter,
    n_classes=9, forecaster=forecaster,
    prune_layer=2, keep_ratio=0.1,
).to(device)
```

### Evaluating GFLOPs

Use `src/evaluation/benchmark.py`:

```python
from src.evaluation import benchmark_model

bench = benchmark_model(model, test_loader, device, label="pruned keep=10%")
print(bench["ms_per_img"])   # inference latency
print(bench["gflops"])       # FLOPs via fvcore
```

---

## `FrozenPrunedLinearProbe`

```python
FrozenPrunedLinearProbe(
    backbone: nn.Module,
    adapter: ThunderBackboneAdapter,
    forecaster: nn.Module,
    n_classes: int,
    layers_source: int | list[int],
    keep_ratio: float = 0.5,
)
```

Linear probing model for the unsupervised EAF experiment. The backbone and forecaster are frozen at construction and kept in eval mode at all times (via `train()` override). Only the `head` (a single `nn.Linear`) receives gradients.

Supports **multi-layer source**: when `layers_source` contains more than one index, patch embeddings from each source block are captured via forward hooks and concatenated along the feature dimension before being fed to the forecaster. Pruning is applied after `max(layers_source)`.

### Parameters

| Argument | Description |
|---|---|
| `backbone` | Raw timm ViT (will be frozen). |
| `adapter` | `ThunderBackboneAdapter` for the same backbone. |
| `forecaster` | `AttentionForecaster` (will be frozen). Its `embed_dim` must equal `adapter.embed_dim × len(layers_source)`. |
| `n_classes` | Number of output classes. |
| `layers_source` | Block index (int) or list of block indices whose patch embeddings are fed to the forecaster. |
| `keep_ratio` | Fraction of spatial patch tokens to retain. |

### Attributes

| Attribute | Description |
|---|---|
| `head` | `nn.Linear(embed_dim, n_classes)` — the only trainable component. |
| `prune_layer` | `max(layers_source)` — block where pruning is applied. |
| `layers_source` | Sorted list of source block indices. |

### Example

```python
from src.models import AttentionForecaster, FrozenPrunedLinearProbe, ThunderBackboneAdapter
from src.collection.unsupervised_cache import build_frozen_model
from src.evaluation.metrics import evaluate
from thunder.models.pretrained_models import get_model_from_name
import torch

raw_backbone, transform, _ = get_model_from_name("uni", "cuda")
_, adapter = build_frozen_model("uni", raw_backbone, device)

# Load a universal forecaster (embed_dim inferred from weights)
state = torch.load("forecaster_uni_src02_attn23_universal.pt", map_location=device)
forecaster = AttentionForecaster(
    embed_dim=state["input_proj.weight"].shape[1],
    hidden=state["input_proj.weight"].shape[0],
).to(device)
forecaster.load_state_dict(state)

model = FrozenPrunedLinearProbe(
    backbone=raw_backbone,
    adapter=adapter,
    forecaster=forecaster,
    n_classes=9,
    layers_source=[2],
    keep_ratio=0.5,
).to(device)

# Train only the head
opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
# ... training loop ...

metrics = evaluate(model, test_loader, device)
# metrics: {acc, f1_macro, auroc, tar_at_far, threshold}
```

### Notes

- `evaluate()` in `src/evaluation/metrics.py` now returns `auroc` in addition to `acc`, `f1_macro`, `tar_at_far`, and `threshold`. Binary classification uses the positive-class score; multiclass uses OvR macro.
- The backbone is shared across model instances when running multiple experiments in sequence — `to(device)` moves it only once.
- `train()` sets only `head.training = True`; backbone and forecaster remain in eval mode regardless.
