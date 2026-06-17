# API — Pruned Models

**Module:** `src/models/pruned_classifier.py`
**Import:** `from src.models import GenericLoRAWithForecasterPruning, FrozenPrunedLinearProbe, DistilledPrunedBackbone`

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

LoRA adapters are scoped to blocks **strictly after** `prune_layer` only (`post_prune_lora_targets`). Blocks at or before `prune_layer` see the same tokens regardless of pruning, so they have nothing to compensate for and stay frozen at their pretrained weights — only the blocks that actually run on the shortened sequence are fine-tuned. This holds regardless of the Phase 1 adaptation strategy.

### Parameters

| Argument | Description |
|---|---|
| `backbone` | Raw timm model from `get_model_from_name`. |
| `adapter` | `ThunderBackboneAdapter` for the same backbone. |
| `n_classes` | Number of output classes. |
| `forecaster` | Trained `AttentionForecaster`, **must be frozen before passing in**. |
| `prune_layer` | Block index where pruning is applied (0-indexed). Must leave at least one block to adapt: `prune_layer < adapter.n_blocks - 1`. |
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

If Phase 1 used `LoRAClassifier`, head weights and the LoRA weights of blocks **after** `prune_layer` match exactly and are loaded. Phase 1 LoRA weights for blocks at or before `prune_layer` have no corresponding parameter here (those blocks carry no adapter — see scoping above) and show up in `unexpected`; this is expected, not a bug. Those blocks start from the plain pretrained weights instead.
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
- `scripts/linear_probe_pruned_eaf.py` accepts `--backbone-ckpt` to load a non-default `state_dict` onto the backbone before probing (e.g. the output of `DistilledPrunedBackbone` distillation below) and `--backbone-tag` to label the resulting CSV rows.

---

## `DistilledPrunedBackbone`

```python
DistilledPrunedBackbone(
    backbone: nn.Module,
    adapter: ThunderBackboneAdapter,
    forecaster: nn.Module,
    prune_layer: int,
    keep_ratio: float,
    lora_r: int = 8,
    lora_alpha: int = 32,
)
```

Phase 3, "Approach 3": dataset-agnostic counterpart to `GenericLoRAWithForecasterPruning`. There is no classification head — `forward(x)` returns the CLS embedding `(B, embed_dim)`. Training target is the CLS token the same backbone would have produced *without* pruning (a frozen "teacher" copy of the backbone), not class labels, so one run produces a single backbone usable across every downstream dataset.

Only blocks strictly after `prune_layer` carry LoRA adapters (see `post_prune_lora_targets`). Blocks at or before `prune_layer` are identical between teacher and student — they see the same tokens either way — so they stay completely frozen, unlike `GenericLoRAWithForecasterPruning` which LoRA-adapts the whole backbone.

### Parameters

| Argument | Description |
|---|---|
| `backbone` | Raw timm model from `get_model_from_name`. Will host LoRA adapters on post-`prune_layer` blocks only. |
| `adapter` | `ThunderBackboneAdapter` for the same backbone. |
| `forecaster` | Trained `AttentionForecaster` (typically the **universal** one — see `train_forecaster_unsupervised.py` — since the whole point is to stay dataset-agnostic). Frozen internally. |
| `prune_layer` | Block index where pruning is applied (0-indexed). Must leave at least one block to distill (`prune_layer < adapter.n_blocks - 1`). |
| `keep_ratio` | Fraction of spatial patch tokens to keep. |
| `lora_r`, `lora_alpha` | LoRA hyperparameters for the post-prune blocks. |

### `post_prune_lora_targets(adapter, prune_layer) -> list[str]`

Returns the exact peft `target_modules` list (`"blocks.{i}.attn.qkv"`, `.attn.proj`, `.mlp.fc1`, `.mlp.fc2"` for `i > prune_layer`). Exact names rather than bare suffixes (`"qkv"`, `"proj"`, ...) so block 5 is never matched by a rule meant for block 15 — peft checks list membership/exact-suffix match, see `peft.tuners.tuners_utils.check_target_module_exists`.

### Training recipe (`scripts/distill_pruned.py`)

```python
teacher = raw_backbone_copy.to(device).eval()  # untouched, frozen
for p in teacher.parameters():
    p.requires_grad_(False)

student = DistilledPrunedBackbone(
    backbone=another_raw_backbone_copy, adapter=adapter, forecaster=forecaster,
    prune_layer=2, keep_ratio=0.1,
).to(device)

teacher_cls = teacher.forward_features(imgs)[:, 0]          # no_grad
student_cls = student(imgs)
loss = mse_weight * F.mse_loss(student_cls, teacher_cls) \
     + cosine_weight * (1 - F.cosine_similarity(student_cls, teacher_cls, dim=-1).mean())
```

Teacher and student must be **separate backbone instances** (two calls to `get_model_from_name`) — reusing one object for both would mean the "teacher" forward also runs through the (partially trained) LoRA weights once the first optimizer step lands, defeating the purpose of a fixed target.

### Merging after training

```python
merged_backbone = student.backbone.merge_and_unload()   # call ONCE, after training ends
torch.save(merged_backbone.state_dict(), "distilled_uni_prune2_keep10.pt")
```

`merge_and_unload()` folds the LoRA deltas into the base weights **in place** and removes the adapter layers, turning `student.backbone` back into a plain timm module — do not call it mid-training, it permanently destroys the LoRA structure. The resulting `state_dict()` has the exact same keys as a vanilla `get_model_from_name` backbone, so it can be loaded directly:

```python
raw_backbone, transform, _ = get_model_from_name("uni", "cuda")
raw_backbone.load_state_dict(torch.load("distilled_uni_prune2_keep10.pt"))
# then build FrozenPrunedLinearProbe(backbone=raw_backbone, ...) as usual
```

or via `scripts/linear_probe_pruned_eaf.py --backbone-ckpt distilled_uni_prune2_keep10.pt --backbone-tag distilled` for the per-dataset linear-probing sweep.
