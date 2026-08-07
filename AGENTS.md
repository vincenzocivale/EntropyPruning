# Repository Guidelines

## Project structure

Core library code lives under `src/`. Keep reusable functionality there; `scripts/`
contains thin entry points only. Dataset acquisition/manifest code belongs in
`src/data/wsi/`; WSI preprocessing, cache contracts, model adapters and cold-archive
logic belong in `src/wsi_pipeline/`. Tests mirror those domains under `tests/`.

The WSI project uses a single runtime root, `$EAF_WSI_ROOT`, documented in
`docs/data_layout.md`. Do not introduce another top-level data layout.

## Architectural invariants

1. **Offline EAF training.** Frozen teacher models are executed during cache creation,
   not in every training epoch.
2. **Dataset-role separation.** `datasets/pretraining/` and `datasets/downstream/` are
   distinct. Strict pretraining must not contain WSI from the downstream benchmark bank.
3. **One raw copy.** Raw WSI files live under `sources/`; datasets reference them via
   manifests/symlinks.
4. **Preserve existing TCGA/HEST.** Refactors must not move, delete or reprocess the
   current TCGA/HEST assets unless explicitly requested.
5. **Version caches.** Cache metadata records model name/revision, early layer,
   preprocessing resolution, patch size, dtype and attention-reduction policy.
6. **Archive before release.** Raw data may be released only after the cold pixel archive
   and required caches pass integrity checks.

## Operational entry point

Prefer `python scripts/eaf.py ...` for new data/cache/archive operations. Do not add new
one-off smoke/debug scripts when a test or a subcommand can cover the behavior.

## Build and tests

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
pytest tests/data/wsi/test_refactor_layout.py
pytest tests/wsi_pipeline/test_cache_contracts.py -q
python scripts/eaf.py layout --data-root "$EAF_WSI_ROOT"
```

Prefer invoking scripts from the repo root so their `src` imports resolve as expected.
Optional heavy dependencies must be guarded with `pytest.importorskip(...)` or imported
inside the command that needs them.

## Coding Style & Naming Conventions

Follow existing Python conventions: 4-space indentation, type hints on public functions,
and concise module docstrings where useful. Use `snake_case` for files, functions,
variables, and CLI flags; use `PascalCase` for classes and dataclasses. Keep new modules
in the existing domain structure instead of adding parallel top-level packages. Keep
machine-specific paths out of reusable code; expose them as CLI arguments or
environment-specific defaults (`$EAF_WSI_ROOT`). Prefer dataclasses and explicit
validation for cache/data contracts.

## Testing Guidelines

Tests use `pytest`, with optional dependency gates via `pytest.importorskip(...)` for
packages like `h5py`, `torch`, and `matplotlib`. Name new tests `test_<behavior>.py` and
keep them beside the closest domain area. Add or update tests for every behavior change,
especially for CLI validation paths, feature-store I/O, cache I/O, and WSI batching logic.

## Cleanup policy

The post-refactor repository should not retain manual debug/smoke scripts or obsolete WSI
forecaster experiments once their useful behavior is covered by the unified CLI/tests.
Before deleting a legacy script, verify that it is not imported by code/tests/docs and
that any unique behavior has been migrated. See `docs/refactor_migration.md` for the
current removal candidates.

## Commit & Pull Request Guidelines

Recent history favors short, imperative commit messages such as `Restore make_gdc_wsi_manifest script` and `Support singleton multilayer TRIDENT features`. Keep commits focused and descriptive. Pull requests should state the problem, summarize the approach, list validation performed (`pytest ...` commands), and note any data or environment assumptions.

## Commits and local data

Large datasets, caches, archives, checkpoints, logs and results are local-only. Never add
them to Git. Before reconciling a dirty worktree, save `git status`, `git diff`, untracked
file inventory and dataset manifests; do not use destructive reset/clean commands.
