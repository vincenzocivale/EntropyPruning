# API — GenericLoRAWithForecasterPruning

**Module:** `src/models/pruned_classifier.py`
**Import:** `from src.models import GenericLoRAWithForecasterPruning`

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
