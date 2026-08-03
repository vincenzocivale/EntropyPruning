# Repository Guidelines

## Project Structure & Module Organization
Core library code lives under `src/`. The main areas are `src/models/` for model definitions, `src/training/` for training loops, `src/data/` for datasets and WSI feature-store utilities, and `src/evaluation/` for metrics and benchmarking. Operational entry points live in `scripts/`, including WSI import, training, evaluation, and smoke-test helpers. Tests mirror the code layout under `tests/` (`tests/data`, `tests/models`, `tests/scripts`, `tests/training`). Research notes and workflow references are in `docs/`, and exploratory analysis stays in `notebooks/`.

## Build, Test, and Development Commands
Use the Conda environment defined in `environment.yml`.

```bash
conda create -n trident --file environment.yml
conda activate trident
```

Run the full test suite from the repository root with:

```bash
pytest
```

Run a focused subset while iterating, for example:

```bash
pytest tests/models/test_wsi_abmil.py
pytest tests/scripts/test_train_wsi_abmil.py -q
python scripts/train_wsi_abmil.py --help
```

Prefer invoking scripts from the repo root so their `src` imports resolve as expected.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, type hints on public functions, and concise module docstrings where useful. Use `snake_case` for files, functions, variables, and CLI flags; use `PascalCase` for classes and dataclasses. Keep new modules in the existing domain structure instead of adding parallel top-level packages. No formatter or linter config is checked in here, so match the surrounding style closely and keep argument validation explicit, as in `scripts/train_wsi_abmil.py`.

## Testing Guidelines
Tests use `pytest`, with optional dependency gates via `pytest.importorskip(...)` for packages like `h5py`, `torch`, and `matplotlib`. Name new tests `test_<behavior>.py` and keep them beside the closest domain area. Add or update tests for every behavior change, especially for CLI validation paths, feature-store I/O, and WSI batching logic. For script changes, prefer a targeted test in `tests/scripts/` plus a smoke-style path when applicable.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages such as `Restore make_gdc_wsi_manifest script` and `Support singleton multilayer TRIDENT features`. Keep commits focused and descriptive. Pull requests should state the problem, summarize the approach, list validation performed (`pytest ...` commands), and note any data or environment assumptions. Include sample CLI invocations or output paths when changing scripts or manifest formats.

## Data & Configuration Notes
Large datasets and feature stores are external to the repo. Avoid hardcoding machine-specific paths in reusable code; expose them as CLI arguments or environment-specific defaults. When changing manifest, HDF5, or ranking-store behavior, update the relevant docs in `docs/` alongside code and tests.
