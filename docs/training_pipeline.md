# Training Pipeline — Step-by-Step Execution

This guide walks through executing the complete EAF pipeline: Phase 1 (forecaster training) → Phase 2 (distillation) → Phase 3 (evaluation) on WSI datasets.

---

## Prerequisites

1. **Environment:** `conda activate eaf-wsi`
2. **WSI files:** Organized in a directory (e.g., `/data/wsis/TCGA-BRCA/`)
3. **Encoder support:** Verify your chosen encoder is available via TRIDENT
4. **GPU:** 24+ GB VRAM recommended for ViT-L/H/g backbones

---

## Full Pipeline (One Command)

For convenience, run all three phases in sequence:

```bash
python scripts/run_wsi_pipeline.py \
    --encoder uni_v1 \
    --dataset TCGA-BRCA \
    --task subtype \
    --wsi-dir /data/wsis/TCGA-BRCA \
    --output-dir ./results \
    --prune-layer 4 \
    --keep-ratio 0.5 \
    --epochs-phase1 30 \
    --epochs-phase2 20 \
    --batch-size 32 \
    --wandb-project my-eaf-experiments
```

This script:
1. Trains forecaster on WSIs
2. Distills pruned model on WSIs
3. Evaluates on Patho-Bench test set

For more granular control, follow the steps below.

---

## Phase 1 — AttentionForecaster Training

### Goal
Train a lightweight attention predictor to rank patches by importance, using only WSI tiles (no labels needed).

### Command

```bash
python scripts/wsi_train_forecaster.py \
    --encoder uni_v1 \
    --wsi-dir /data/wsis/training_slides \
    --prune-layer 4 \
    --target-layer 23 \
    --mag 20 \
    --patch-size 256 \
    --tiles-per-wsi 64 \
    --val-split 0.1 \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir ./results/phase1 \
    --wandb-project my-eaf-experiments
```

### Key arguments

| Argument | Description | Default |
|---|---|---|
| `--encoder` | TRIDENT encoder name (uni_v1, virchow, hoptimus0, ...) | required |
| `--wsi-dir` | Directory with `.svs`, `.ndpi`, `.tif` files | required |
| `--prune-layer` | Source layer for embeddings (typically 4–8 for ViT-L) | required |
| `--target-layer` | Target layer for attention supervision | n_blocks - 1 |
| `--mag` | Tile magnification | 20 |
| `--patch-size` | Tile size in pixels | 256 |
| `--tiles-per-wsi` | Sampled tiles per WSI per epoch | 64 |
| `--val-split` | Fraction of WSIs for validation | 0.1 |
| `--epochs` | Training epochs | 30 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 1e-4 |
| `--weight-decay` | Weight decay | 0.0 |
| `--grad-clip` | Gradient clipping norm | 1.0 |
| `--hidden` | Forecaster hidden dimension | 256 |
| `--n-heads` | Forecaster attention heads | 4 |
| `--n-layers` | Forecaster transformer layers | 2 |
| `--dropout` | Forecaster dropout | 0.2 |
| `--wandb-project` | W&B project (optional) | None |
| `--output-dir` | Where to save checkpoint | required |

### Monitoring

- **Validation metrics:**
  - `val_loss` — KL divergence (should decrease)
  - `val_rho` — Spearman rank correlation (should increase, target > 0.6)
- **Early stopping:** Best checkpoint saved when `val_loss` improves
- **Output:**
  - `forecaster_uni_v1_src4_tgt23.pt` — best model weights
  - `results.json` — training metrics

### Example output
```
Epoch  1 | train_loss=0.5123 | val_loss=0.4891  val_rho=0.6234
Epoch  2 | train_loss=0.4756 | val_loss=0.4523  val_rho=0.6512
...
Epoch 30 | train_loss=0.2134 | val_loss=0.2098  val_rho=0.8234
  → Saved best checkpoint: ./results/phase1/forecaster_uni_v1_src4_tgt23.pt

=======================================================================
Training complete!
  Best val_loss: 0.2098
  Best val_rho: 0.8234
=======================================================================
```

### Troubleshooting

**"No WSI files found"**
- Check: files are `.svs`, `.ndpi`, `.tif`, or `.tiff`
- Check: `--wsi-dir` is correct and readable

**High loss, poor Spearman correlation**
- Try: different source layer (earlier in network may be too noisy)
- Try: longer training (`--epochs 50`)
- Try: larger forecaster (`--hidden 512`, `--n-layers 4`)

**GPU out of memory**
- Reduce `--batch-size` (e.g., 16 or 8)
- Reduce `--tiles-per-wsi` (e.g., 32)
- Use smaller encoder (if possible)

---

## Phase 2 — Pruned Model Distillation

### Goal
Fine-tune the pruned backbone (with LoRA adapters) to match the non-pruned teacher's CLS embeddings while maintaining forecaster-based pruning.

### Prerequisites
- Phase 1 checkpoint: `forecaster_uni_v1_src4_tgt23.pt`
- **Important:** Prune layer must match Phase 1

### Command

```bash
python scripts/wsi_distill_pruned.py \
    --encoder uni_v1 \
    --wsi-dir /data/wsis/training_slides \
    --forecaster-ckpt ./results/phase1/forecaster_uni_v1_src4_tgt23.pt \
    --prune-layer 4 \
    --keep-ratio 0.5 \
    --mag 20 \
    --patch-size 256 \
    --tiles-per-wsi 64 \
    --val-split 0.1 \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --output-dir ./results/phase2 \
    --wandb-project my-eaf-experiments
```

### Key arguments

| Argument | Description | Default |
|---|---|---|
| `--forecaster-ckpt` | Path to Phase 1 forecaster checkpoint | required |
| `--prune-layer` | **Must match Phase 1** | required |
| `--keep-ratio` | Fraction of tokens to keep (0.0–1.0, e.g., 0.5 = 50% kept) | required |
| `--epochs` | Training epochs (typically shorter than Phase 1) | 20 |
| `--weight-decay` | LoRA regularization | 0.01 |
| `--lr` | Learning rate for LoRA | 1e-4 |

### Monitoring

- **Validation metric:** `val_loss` (cosine distance between student and teacher CLS embeddings)
- **Expected behavior:** Loss should decrease monotonically
- **Output:**
  - `best_uni_v1_prune4_keep50.pt` — pruned model with LoRA weights
  - `results_phase2.json` — distillation metrics

### Example output
```
Epoch  1 | train_loss=0.3456 | val_loss=0.3123
Epoch  2 | train_loss=0.3012 | val_loss=0.2876
...
Epoch 20 | train_loss=0.1234 | val_loss=0.1198
  → Saved: ./results/phase2/best_uni_v1_prune4_keep50.pt

=======================================================================
Training complete!
  Best val_loss: 0.1198
=======================================================================
```

### Troubleshooting

**Loss doesn't decrease or plateaus early**
- Try: lower `--keep-ratio` (less aggressive pruning; e.g., 0.7 instead of 0.5)
- Try: higher `--lr` (e.g., 5e-4 instead of 1e-4)
- Try: longer training (`--epochs 40`)

**Training diverges (loss explodes)**
- Reduce `--lr` (e.g., 5e-5)
- Increase `--grad-clip` (e.g., 10.0)
- Reduce `--batch-size`

**GPU out of memory**
- Reduce `--batch-size` or `--tiles-per-wsi`
- If still too large, try a smaller encoder

---

## Phase 3 — WSI-Level Evaluation

### Goal
Evaluate on full Patho-Bench WSI slides: extract tile embeddings, aggregate to slide level, train a linear classifier, and report accuracy/F1/AUC.

### Prerequisites
- Phase 2 checkpoint: `best_uni_v1_prune4_keep50.pt`
- Patho-Bench dataset (auto-downloaded on first access via TRIDENT)

### Command

```bash
python scripts/wsi_evaluate.py \
    --encoder uni_v1 \
    --dataset TCGA-BRCA \
    --task subtype \
    --checkpoint ./results/phase2/best_uni_v1_prune4_keep50.pt \
    --mag 20 \
    --patch-size 256 \
    --batch-size 32 \
    --output-dir ./results/phase3
```

### Key arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Patho-Bench dataset (TCGA-BRCA, TCGA-LUAD, BACH, ...) | required |
| `--task` | Classification task for dataset (e.g., subtype, mutational_status) | required |
| `--checkpoint` | Phase 2 pruned model checkpoint | required |
| `--compare-full` | Also evaluate unpruned encoder for comparison (slower) | False |

### Monitoring

- **Progress:** Slides are processed in batches; progress bar shown
- **Output:** Per-slide predictions and aggregated metrics

### Example output
```
Loading checkpoint: ./results/phase2/best_uni_v1_prune4_keep50.pt
Processing TCGA-BRCA slides...
  Slide 001: pred=LumA, conf=0.92
  Slide 002: pred=LumB, conf=0.87
  ...

=== Metrics (Pruned, keep_ratio=0.5) ===
  Accuracy: 0.8234
  F1 (macro): 0.7956
  AUC: 0.8876
  Speedup: 1.34x (inference 25% faster)

Results saved to:
  - predictions.csv
  - metrics.json
```

---

## Ablation Studies

### Layer Sweep
Test different source layers (prune points):

```bash
for layer in 2 4 6 8 10; do
  echo "=== Layer $layer ==="
  
  # Phase 1
  python scripts/wsi_train_forecaster.py \
      --encoder uni_v1 \
      --wsi-dir /data/wsis/training \
      --prune-layer $layer \
      --target-layer 23 \
      --epochs 30 \
      --output-dir ./ablation/layer_$layer/phase1
  
  # Phase 2
  python scripts/wsi_distill_pruned.py \
      --encoder uni_v1 \
      --wsi-dir /data/wsis/training \
      --forecaster-ckpt ./ablation/layer_$layer/phase1/forecaster_*.pt \
      --prune-layer $layer \
      --keep-ratio 0.5 \
      --epochs 20 \
      --output-dir ./ablation/layer_$layer/phase2
  
  # Phase 3
  python scripts/wsi_evaluate.py \
      --encoder uni_v1 \
      --dataset TCGA-BRCA \
      --task subtype \
      --checkpoint ./ablation/layer_$layer/phase2/best_*.pt \
      --output-dir ./ablation/layer_$layer/phase3
done
```

### Keep-ratio Sweep
Test different compression levels:

```bash
for ratio in 0.3 0.5 0.7 0.9; do
  echo "=== Keep ratio $ratio ==="
  
  python scripts/wsi_distill_pruned.py \
      --encoder uni_v1 \
      --wsi-dir /data/wsis/training \
      --forecaster-ckpt ./results/phase1/forecaster_*.pt \
      --prune-layer 4 \
      --keep-ratio $ratio \
      --epochs 20 \
      --output-dir ./ablation/keep_${ratio}/phase2
  
  python scripts/wsi_evaluate.py \
      --encoder uni_v1 \
      --dataset TCGA-BRCA \
      --task subtype \
      --checkpoint ./ablation/keep_${ratio}/phase2/best_*.pt \
      --output-dir ./ablation/keep_${ratio}/phase3
done
```

Collect results:
```python
import json
import pandas as pd

results = []
for ratio in [0.3, 0.5, 0.7, 0.9]:
    with open(f"./ablation/keep_{ratio}/phase3/metrics.json") as f:
        m = json.load(f)
    results.append({
        "keep_ratio": ratio,
        "accuracy": m["accuracy"],
        "f1": m["f1_macro"],
        "auc": m["auc"],
        "speedup": m.get("speedup", 1.0),
    })

df = pd.DataFrame(results)
print(df)
```

---

## Common Workflows

### Quick Testing
Minimal setup for rapid iteration:

```bash
python scripts/run_wsi_pipeline.py \
    --encoder uni_v1 \
    --dataset TCGA-BRCA \
    --task subtype \
    --wsi-dir /data/wsis/TCGA-BRCA \
    --output-dir ./results \
    --prune-layer 4 \
    --keep-ratio 0.5 \
    --tiles-per-wsi 16 \
    --epochs-phase1 3 \
    --epochs-phase2 2
```

Completes in ~5 minutes, gives quick feedback on setup and hyperparameters.

### Multi-encoder Comparison
Train and evaluate across multiple encoders:

```bash
for encoder in uni_v1 virchow hoptimus0; do
  echo "=== Encoder: $encoder ==="
  
  python scripts/run_wsi_pipeline.py \
      --encoder $encoder \
      --dataset TCGA-BRCA \
      --task subtype \
      --wsi-dir /data/wsis/TCGA-BRCA \
      --output-dir ./results/$encoder \
      --prune-layer 4 \
      --keep-ratio 0.5 \
      --wandb-project eaf-encoders
done
```

### Multi-dataset Evaluation
Test on multiple Patho-Bench datasets:

```bash
for dataset in TCGA-BRCA TCGA-LUAD BACH BreakHIS; do
  # Extract dataset and task
  IFS='-' read -ra parts <<< "$dataset"
  if [[ "$dataset" == "TCGA-"* ]]; then
    ds=${dataset}
    task="subtype"
  else
    ds=$dataset
    task="tumor_grade"  # dataset-specific
  fi
  
  echo "=== $dataset ($task) ==="
  
  python scripts/wsi_evaluate.py \
      --encoder uni_v1 \
      --dataset $ds \
      --task $task \
      --checkpoint ./results/best_*.pt \
      --output-dir ./results/eval_$dataset
done
```

---

## Output Structure

After running the full pipeline, your `results/` directory will contain:

```
results/
├── phase1/
│   ├── forecaster_uni_v1_src4_tgt23.pt    ← Best Phase 1 checkpoint
│   └── results.json                       ← Phase 1 metrics
├── phase2/
│   ├── best_uni_v1_prune4_keep50.pt       ← Best Phase 2 checkpoint
│   └── results_phase2.json                ← Phase 2 distillation loss
├── phase3/
│   ├── predictions.csv                    ← Per-slide predictions
│   ├── metrics.json                       ← Accuracy, F1, AUC, etc.
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── auc_by_subtype.png
└── logs/
    └── wandb_runs/                        ← If --wandb-project enabled
```

### Loading and analyzing results

```python
import json
import pandas as pd

# Phase 1 metrics
with open("results/phase1/results.json") as f:
    p1 = json.load(f)
print(f"Phase 1 best correlation: {p1['best_val_rho']:.4f}")

# Phase 3 predictions
preds = pd.read_csv("results/phase3/predictions.csv")
print(f"Processed {len(preds)} slides")
print(preds.head())

# Phase 3 metrics
with open("results/phase3/metrics.json") as f:
    metrics = json.load(f)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (macro): {metrics['f1_macro']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")

# If speedup computed
if "speedup" in metrics:
    print(f"Inference speedup: {metrics['speedup']:.2f}x")
    print(f"Token reduction: {(1 - keep_ratio) * 100:.1f}%")
```

---

## Tips & Tricks

### Checkpointing intermediate results
If running the full pipeline, intermediate Phase 1 and Phase 2 checkpoints are saved in `--output-dir`. You can resume or skip phases:

```bash
# Skip Phase 1 (use pre-trained forecaster)
python scripts/run_wsi_pipeline.py \
    ... \
    --forecaster-ckpt ./results/phase1/forecaster_*.pt \
    --skip-phase1
```

### W&B integration
If running with `--wandb-project my-project`, all metrics are logged to Weights & Biases.  
Useful for comparing multiple runs:

```bash
# Run 1
python scripts/wsi_train_forecaster.py \
    ... \
    --wandb-project my-eaf \
    --output-dir ./results/run1

# Run 2 (different hyperparameters)
python scripts/wsi_train_forecaster.py \
    ... \
    --lr 5e-4 \
    --hidden 512 \
    --wandb-project my-eaf \
    --output-dir ./results/run2

# View comparison at wandb.ai/my-username/my-eaf
```

### Resuming interrupted training
If a run is interrupted (GPU crash, etc.), restart from the same checkpoint:

```bash
# Phase 2 will continue from last epoch if checkpoint exists
python scripts/wsi_distill_pruned.py \
    ... \
    --output-dir ./results/phase2  ← same as before
    --epochs 40                     ← can extend
```

---

## Debugging Common Issues

### Phase 1: "Hook didn't capture data"
- Check: `--prune-layer` and `--target-layer` are valid (< n_blocks)
- Check: tiles are being loaded (check dataset build logs)

### Phase 2: Student model architecture mismatch
- Check: `--prune-layer` matches Phase 1
- Check: encoder name matches Phase 1

### Phase 3: Very low accuracy
- Try: less aggressive pruning (`--keep-ratio 0.7` or `0.9`)
- Try: different prune layer (earlier layers may be too coarse)
- Check: dataset/task combination is valid for encoder

### All phases: GPU memory issues
- Reduce batch size: `--batch-size 8` or `16`
- Reduce tiles per WSI: `--tiles-per-wsi 32`
- Use a smaller encoder if available

### All phases: Slow training
- Increase workers: `--num-workers 8` (if available)
- Reduce verbosity: remove `--verbose` flag
- Check GPU utilization with `nvidia-smi`
