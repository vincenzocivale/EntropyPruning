# API — Classifiers

**Module:** `src/models/classifier.py`
**Imports:** `from src.models import BaseClassifier, LoRAClassifier, LinearProbingClassifier, FullFinetuneClassifier, BitFitClassifier, build_classifier, STRATEGIES`

---

## `BaseClassifier`

Abstract base class for all EAF adaptation strategies.

```python
class BaseClassifier(nn.Module, ABC)
```

### Abstract interface

| Member | Type | Description |
|---|---|---|
| `raw_backbone` | `property → nn.Module` | Underlying timm `VisionTransformer`. Used for block-level forward hooks. |
| `trainable_backbone_params` | `property → List[Parameter]` | Backbone params that receive the backbone learning rate. Empty for frozen strategies. |
| `forward(x)` | `method` | Returns class logits `(B, n_classes)`. |

### Required attributes (set by subclasses)

| Attribute | Type | Description |
|---|---|---|
| `adapter` | `ThunderBackboneAdapter` | Architecture metadata. |
| `head` | `nn.Sequential` | `LayerNorm → Dropout → Linear`. |

---

## `LinearProbingClassifier`

```python
LinearProbingClassifier(backbone, adapter, n_classes, dropout=0.1)
```

Backbone is frozen. `torch.no_grad()` is applied inside `forward()` for efficiency.
`trainable_backbone_params` is always empty.
`raw_backbone` = `self.backbone` (timm model directly, no peft wrapping).

---

## `LoRAClassifier`

```python
LoRAClassifier(backbone, adapter, n_classes, lora_r=8, lora_alpha=32, dropout=0.1)
```

Applies peft LoRA to `["qkv", "proj", "fc1", "fc2"]` in every transformer block.
After construction:
- `self.backbone` — peft `LoraModel`
- `self.raw_backbone` = `self.backbone.model` — original timm model
- `self.trainable_backbone_params` — LoRA matrices only (`requires_grad=True`)

**Checkpoint keys** use the peft prefix `backbone.base_model.model.*`.

---

## `FullFinetuneClassifier`

```python
FullFinetuneClassifier(backbone, adapter, n_classes, dropout=0.1)
```

All backbone weights receive gradients.
`raw_backbone` = `self.backbone`.
`trainable_backbone_params` = all backbone parameters.

---

## `BitFitClassifier`

```python
BitFitClassifier(backbone, adapter, n_classes, dropout=0.1)
```

All non-bias backbone weights are frozen. Only bias terms (in attention, FFN, LayerNorm) are trained.
`raw_backbone` = `self.backbone`.
`trainable_backbone_params` = bias parameters only.

---

## `build_classifier`

```python
def build_classifier(
    strategy: str,
    backbone: nn.Module,
    adapter: ThunderBackboneAdapter,
    n_classes: int,
    **kwargs,
) -> BaseClassifier
```

Factory function. `strategy` must be one of `STRATEGIES`.

```python
STRATEGIES = ["linear_probing", "lora", "full", "bitfit"]
```

`**kwargs` are forwarded to the strategy constructor:
- `lora` accepts: `lora_r`, `lora_alpha`, `dropout`
- all others accept: `dropout`

### Example

```python
from thunder.models.pretrained_models import get_model_from_name
from src.models import ThunderBackboneAdapter, build_classifier
from src.utils import build_optimizer

raw_backbone, transform, _ = get_model_from_name("uni", "cuda")
adapter = ThunderBackboneAdapter(raw_backbone)

model = build_classifier("lora", raw_backbone, adapter, n_classes=9, lora_r=16)
model = model.to(device)

optimizer = build_optimizer(model, lr_backbone=1e-5, lr_head=1e-3, weight_decay=0.01)
```
