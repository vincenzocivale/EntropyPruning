# Setup

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU (recommended: ≥ 24 GB VRAM for ViT-L/H/g backbones)
- Git

## Environment setup

### 1. Create conda environment

```bash
cd /path/to/EntropyPruning
conda env create -f environment_wsi.yml
conda activate eaf-wsi
```

The environment is pre-configured with:
- PyTorch (2.1.x) with CUDA 12.1
- TRIDENT (WSI loading, tile extraction, encoders)
- Patho-Bench (benchmark datasets)
- PyTorch Lightning, PEFT (LoRA), W&B
- **timm==1.0.25** (pinned for encoder compatibility)

### 2. Install TRIDENT and Patho-Bench

**Important:** Use `--no-deps` to preserve the timm version.

```bash
# From the parent directory (assumes TRIDENT is a sibling repo)
pip install -e /path/to/TRIDENT --no-deps

# Install TRIDENT's runtime dependencies not already covered
pip install omegaconf hydra-core kornia ijson sentencepiece
pip install opencv-python plotly pydantic typer einops

# Install Patho-Bench
pip install -e /path/to/Patho-Bench
```

Verify TRIDENT:
```bash
python -c "from trident.patch_encoder_models import encoder_factory; print('TRIDENT OK')"
```

## HuggingFace Model Authentication

Several foundation models require accepting terms on HuggingFace before downloading. For each model you plan to use, visit its card and accept the license:

```bash
huggingface-cli login
```

Required for:
- [UNI v1](https://huggingface.co/MahmoodLab/UNI)
- [Virchow](https://huggingface.co/paige-ai/Virchow)
- [H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0)
- [Hiboul](https://huggingface.co/jlevy44/hiboul)

## WSI Data Setup

WSI files (`.svs`, `.ndpi`, `.tiff`, `.tif`) must be organized in a directory. TRIDENT will automatically:
1. Load slides via OpenSlide / ASAP
2. Segment tissue using Otsu thresholding
3. Extract tiles at specified magnification

```bash
# Example directory structure
/path/to/wsis/
├── slide_001.svs
├── slide_002.svs
├── ...
```

Pass `--wsi-dir /path/to/wsis` to all training and evaluation scripts.

## Patho-Bench Datasets

Benchmark datasets are accessed via Patho-Bench (which wraps TRIDENT).  
Datasets are downloaded on-demand when first accessed. Ensure internet access and sufficient disk space.

Supported datasets:
- **TCGA-BRCA, TCGA-LUAD, TCGA-KIRC, TCGA-COAD**, etc. (specify `--dataset TCGA-{ORGAN}`, `--task {task_name}`)
- **BACH** (breast histology)
- **BreakHIS** (breast cancer)
- **CAMELYON16/17** (lymph node metastasis)

## Outputs and Checkpoints

All scripts save results to `--output-dir` (default: `./results`). Structure:

```
results/
├── forecaster_uni_v1_src4_tgt23.pt      (Phase 1 checkpoint)
├── best_uni_v1_prune4_keep50.pt         (Phase 2 checkpoint)
├── predictions.csv                      (Phase 3 predictions)
├── results.json                         (metrics)
```

## Weights & Biases (Optional)

To track experiments in W&B, set `--wandb-project {project_name}` and ensure you're logged in:

```bash
wandb login
```

## Directory structure (after setup)

```
EAF_WSI/
├── EntropyPruning/                 (this repo)
│   ├── environment_wsi.yml         (conda spec)
│   ├── scripts/
│   │   ├── wsi_train_forecaster.py (Phase 1)
│   │   ├── wsi_distill_pruned.py   (Phase 2)
│   │   ├── wsi_evaluate.py         (Phase 3)
│   │   └── run_wsi_pipeline.py     (full pipeline)
│   ├── src/
│   │   ├── models/
│   │   │   ├── forecaster.py
│   │   │   ├── backbone_adapter.py
│   │   │   ├── pruned_classifier.py
│   │   └── data/
│   │       └── wsi_tile_dataset.py
│   ├── docs/
│   └── results/                    (created at runtime)
├── TRIDENT/                        (sibling repo, installed as editable)
└── Patho-Bench/                    (sibling repo, installed as editable)
```
