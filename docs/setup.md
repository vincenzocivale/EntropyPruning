# Setup

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: ≥ 24 GB VRAM for ViT-L/g backbones)
- Git

## Option A — Conda environment (recommended, used on HAL)

```bash
# Create environment
conda create -n eaf_env python=3.11
conda activate eaf_env

# Install PyTorch (cu124, compatible with driver ≥ 550)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install EAF dependencies
pip install peft transformers timm wandb scikit-learn h5py fvcore tqdm matplotlib

# Install Thunder (from the thunder/ sibling or the installed package)
pip install thunder-bench   # from PyPI (published as thunder-bench)
# OR from source:
pip install -e ../thunder --no-deps
pip install omegaconf hydra-core kornia wilds ijson sentencepiece
pip install opencv-python plotly pydantic typer einops einops_exts
```

Run scripts without activating the environment:
```bash
conda run --no-capture-output -n eaf_env python scripts/train_forecaster_unsupervised.py ...
```

## Option B — Virtual environment (.venv)

The project also supports a local `.venv` (used on Nanopore-PC).

```bash
# Create venv (first time only)
python3.10 -m venv .venv

# Activate
source .venv/bin/activate

# Install EAF dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install peft transformers timm wandb scikit-learn h5py fvcore tqdm matplotlib
```

> **Nota CUDA (Nanopore-PC)**: il driver NVIDIA (535.x) supporta CUDA ≤ 12.2.
> Usare `--index-url https://download.pytorch.org/whl/cu121`. Non installare `torch>=2.6.0`.

## Installing Thunder

```bash
pip install -e ../thunder --no-deps

# Install Thunder's runtime dependencies not already covered by EAF:
pip install omegaconf hydra-core kornia wilds ijson sentencepiece
pip install opencv-python plotly pydantic typer einops einops_exts
```

Verify:

```bash
python -c "import thunder; from thunder.models.pretrained_models import get_model_from_name; print('Thunder OK')"
```

## HuggingFace authentication

Several foundation models require accepting usage terms on HuggingFace before downloading.
Run `huggingface-cli login` and accept the model card for each model you plan to use:

- UNI: <https://huggingface.co/MahmoodLab/UNI>
- UNI2-h: <https://huggingface.co/MahmoodLab/UNI2-h>
- H-optimus-0: <https://huggingface.co/bioptimus/H-optimus-0>
- Virchow: <https://huggingface.co/paige-ai/Virchow>

## Thunder data setup

Thunder datasets must be downloaded and split before use.

```bash
# Download a dataset (e.g. CRC)
thunder download crc --base-data-folder /path/to/thunder/data

# This creates:
#   /path/to/thunder/data/datasets/crc/       — image files
#   /path/to/thunder/data/data_splits/crc.json — train/val/test split
```

The `--base-data-folder` path is passed to all EAF scripts via `--base-data-folder`.

## Directory structure (after setup)

```
EAF/
├── EntropyPruning/          ← this repo (branch: thunder-integration)
│   ├── .venv/
│   ├── checkpoints/         ← created at runtime by training scripts
│   ├── docs/
│   ├── scripts/
│   └── src/
└── thunder/                 ← sibling repo, installed as editable package
```
