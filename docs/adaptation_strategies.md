# Adaptation Strategies

EAF supports four strategies for adapting the backbone during Phase 1 (base classifier training).
The choice affects how many parameters are trained, GPU memory usage, training speed, and typically also final accuracy.

---

## Comparison

| Strategy | CLI flag | Trainable backbone params | Memory | Speed | Typical use case |
|---|---|---|---|---|---|
| Linear Probing | `linear_probing` | 0 | ★☆☆ | ★★★ | Quick baseline; frozen features already strong |
| BitFit | `bitfit` | bias terms only (~0.1%) | ★★☆ | ★★★ | Lightweight, often competitive with LoRA |
| LoRA | `lora` *(default)* | LoRA matrices (~1–2%) | ★★☆ | ★★☆ | Best accuracy/param trade-off for pathology FMs |
| Full Fine-tuning | `full` | all backbone weights (100%) | ★★★ | ★☆☆ | Small datasets risk forgetting; use tiny lr |

---

## Linear Probing

```bash
python scripts/train_classifier.py \
    --adaptation linear_probing \
    --lr-head 1e-3
```

The backbone is completely frozen — no gradient flows through it. Only the `LayerNorm → Dropout → Linear` head is trained.

**When to use:**
- Rapid benchmarking when you need a quick result.
- The pretrained features are already highly discriminative for your task.
- Very limited GPU memory.

**Note:** Because the backbone is frozen, the Phase 2 feature cache is identical to what you would extract from the pretrained model without any fine-tuning. You can share the Phase 2 HDF5 cache across linear probing runs on the same model/dataset combination.

---

## LoRA (default)

```bash
python scripts/train_classifier.py \
    --adaptation lora \
    --lora-r 8 \
    --lora-alpha 32 \
    --lr-backbone 1e-5 \
    --lr-head 1e-3
```

Low-rank adapter matrices are injected into four projection layers per block: `qkv`, `proj`, `fc1`, `fc2`. The base weights remain frozen; only the LoRA matrices and the classification head receive gradients.

**Key hyperparameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--lora-r` | 8 | Rank of the low-rank decomposition. Higher = more capacity, more params. |
| `--lora-alpha` | 32 | Scaling factor; effective scale = `lora_alpha / lora_r`. |
| `--lr-backbone` | 1e-5 | Learning rate for LoRA matrices. |

**When to use:**
- The default choice for pathology foundation models.
- Good balance between expressivity and regularization.
- Compatible with Phase 3 pruned fine-tuning (which always uses LoRA).

**Implementation note:** After construction, `model.backbone` is a peft `LoraModel`. `model.raw_backbone` (= `model.backbone.model`) gives access to the original timm blocks for forward hooks.

---

## Full Fine-tuning

```bash
python scripts/train_classifier.py \
    --adaptation full \
    --lr-backbone 1e-6 \
    --lr-head 1e-3
```

All backbone parameters receive gradients. Use a very small backbone learning rate to avoid catastrophic forgetting of the pretrained representation.

**When to use:**
- Large, diverse downstream dataset.
- You want to maximise final accuracy and have the GPU budget.
- Works best with aggressive learning rate warmup and cosine decay.

**Warning:** For ViT-L (UNI, 307M params) or ViT-g (H-optimus, 1.1B params), full fine-tuning requires > 40 GB VRAM at reasonable batch sizes. Consider gradient checkpointing if memory is tight.

---

## BitFit

```bash
python scripts/train_classifier.py \
    --adaptation bitfit \
    --lr-backbone 5e-5 \
    --lr-head 1e-3
```

Only bias terms in the backbone receive gradients (attention bias, FFN bias, LayerNorm bias/weight). All weight matrices remain frozen.

**When to use:**
- When you want more adaptation than linear probing but cannot afford LoRA memory.
- Surprisingly competitive on tasks where the pretrained bias distribution is a bottleneck.

**Trainable params:** roughly proportional to the number of bias vectors in the transformer.
For ViT-L (1024-dim, 24 blocks): ≈ 500K params.

---

## Choosing a strategy

```
Dataset size < 1k samples?  →  linear_probing (avoid overfitting)
Need a quick baseline?       →  linear_probing
Best accuracy matters?       →  lora (start here) or full (if GPU budget allows)
Memory constrained?          →  linear_probing or bitfit
Ablating adaptation cost?    →  run all four and compare
```

---

## Checkpoint compatibility with Phase 2 & 3

| Phase 1 strategy | Phase 2 feature cache | Phase 3 pruning |
|---|---|---|
| `linear_probing` | Features = pretrained features (backbone not modified) | Phase 3 starts LoRA on pretrained weights + loads Phase 1 head |
| `lora` | Features from LoRA-adapted backbone | Phase 3 loads LoRA weights correctly (matching key names) |
| `full` | Features from fully fine-tuned backbone | Phase 3 applies new LoRA on top; Phase 1 weights loaded as base |
| `bitfit` | Features from bias-adapted backbone | Phase 3 applies LoRA; Phase 1 backbone keys load partially |

Pass `--adaptation {strategy}` to `train_forecaster.py` so it reconstructs the Phase 1 model correctly for feature extraction.
