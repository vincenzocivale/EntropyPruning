# Repository Guidelines

## Project Structure & Module Organization

This repository implements EAF, an entropy-guided token pruning pipeline for Vision Transformer histopathology classifiers. Core Python modules live in `src/`: `models/` contains classifier, backbone adapter, forecaster, and pruned model code; `evaluation/` contains metrics and benchmark utilities; `collection/` contains feature extraction and cache-building helpers; `data/` contains dataset loaders. Entry-point scripts are in `scripts/`, including `train_classifier.py`, `train_forecaster.py`, `train_forecaster_unsupervised.py`, `build_unsupervised_cache.py`, and `finetune_pruned.py`. Documentation is in `docs/`. Generated artifacts (results, checkpoints, wandb logs) are gitignored and should not be committed.

## Build, Test, and Development Commands

Set up the local environment as described in `docs/setup.md`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install peft transformers timm wandb scikit-learn h5py fvcore tqdm
pip install -e ../thunder --no-deps
```

Run the main pipeline with project scripts:

```bash
python scripts/train_classifier.py --model-name uni --dataset-name crc --base-data-folder /path/to/thunder/data
python scripts/train_forecaster.py --model-name uni --dataset-name crc --base-data-folder /path/to/thunder/data --layers-source 2
python scripts/finetune_pruned.py --model-name uni --dataset-name crc --base-data-folder /path/to/thunder/data --prune-layer 2 --keep-ratio 0.1
```

Use `python -m pytest` for tests when test sources are present.

## Coding Style & Naming Conventions

Use Python 3.10, 4-space indentation, and clear module-level imports. Follow existing naming: `snake_case` for functions, variables, files, and script flags; `PascalCase` for model classes such as `AttentionForecaster`; lowercase dataset and model identifiers such as `crc`, `mhist`, `uni`, and `hoptimus0`. Keep CLI scripts thin and place reusable logic in `src/`.

## Testing Guidelines

Add tests under `tests/` with names like `test_imports.py`, `test_losses.py`, or `test_pruned_smoke.py`. Favor fast import, shape, and smoke tests for model wrappers; gate GPU-heavy or dataset-dependent checks behind skips or small fixtures. Before changing training or pruning logic, run `python -m pytest` and at least one minimal script invocation when feasible.

## Commit & Pull Request Guidelines

Recent history uses concise conventional-style subjects such as `fix: ...`, `feat: ...`, and `refactor(phase3): ...`; keep that pattern and mention the affected pipeline phase when useful. Pull requests should summarize the behavior change, list commands or experiments run, link related issues, and include metrics, plots, or W&B links when quality or speed changes. Do not include large generated outputs unless they are intentional published results.

## Security & Configuration Tips

Do not commit credentials, HuggingFace tokens, local dataset paths, or W&B secrets. Keep machine-specific paths in docs or local shell variables, and pass data locations with `--base-data-folder`.
