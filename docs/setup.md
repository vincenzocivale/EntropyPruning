# Setup

## Requirements

- Python 3.10
- CUDA-capable GPU (recommended: ≥ 24 GB VRAM for ViT-L/g backbones)
- Git

## Virtual environment

The project uses a local `.venv` (not the conda `trident` environment, which is kept for reference via `environment.yml`).

```bash
# Create venv (first time only)
python3.10 -m venv .venv

# Activate
source .venv/bin/activate

# Install EAF dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install peft transformers timm wandb scikit-learn h5py fvcore tqdm
```

## Installing Thunder

Thunder must be installed as an editable package from the sibling directory.
Use `--no-deps` to avoid downgrading `timm` (the venv pins `timm==1.0.20`; Thunder's constraint `<=1.0.20` is satisfied).

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
