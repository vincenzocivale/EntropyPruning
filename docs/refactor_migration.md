# EAF WSI refactor and local-data migration

This migration is intentionally additive. It must not destroy the already downloaded and
processed TCGA/HEST data or any uncommitted acquisition work.

## Target

The refactor establishes:

- first-class HISTAI and GTEx corpus adapters,
- strict separation between EAF pretraining and labelled downstream benchmarks,
- offline Tile-EAF caches (early layer, final attention, tile embedding),
- offline WSI-EAF caches (tile embeddings, tile scores, WSI embedding),
- compressed cold tissue-pixel archives for future tile encoders,
- canonical checkpoints/results/logs directories,
- one operational CLI (`scripts/eaf.py`) for new data/cache/archive operations.

## Dirty-worktree rule

Before applying cleanup or moving code, capture:

```bash
git status --short > /tmp/eaf_status_before_refactor.txt
git diff > /tmp/eaf_uncommitted_before_refactor.patch
git ls-files --others --exclude-standard > /tmp/eaf_untracked_before_refactor.txt
```

Do not run `git reset --hard`, `git clean`, or blanket `rm -rf` commands. Untracked scripts
created during HISTAI/GTEx work must be inspected and migrated before deletion.

## Local data that must be preserved

- existing `tcga_eaf_multicohort_v1` raw WSI, coordinates and feature stores,
- existing HEST raw WSI/coordinates/features,
- HISTAI `plan.csv` (currently the candidate one-H&E-per-case plan), downloaded HISTAI
  subsets and Hugging Face cache state,
- GTEx IDC smoke-test DICOM series,
- any manifests or symlink views that point to the above.

Data paths may be *referenced* from the new manifests. Avoid copying multi-TB data solely
to satisfy the new naming convention. Prefer symlinks/manifest path updates.

## Local code to migrate into the new core

If present locally, preserve the newest behavior of `scripts/wsi_prepare_strict_pretraining.py`:

- canonical `$EAF_WSI_ROOT` layout,
- one H&E WSI per `(HISTAI subset, case_id)`,
- incremental `--subset` downloads,
- progress reporting and resume,
- TCGA-preservation/leakage guard,
- HEST registration.

Move reusable parts into `src/data/wsi/corpora.py`, `layout.py` and `manifest.py`; keep
`scripts/eaf.py` thin. Do not keep two competing HISTAI planners after migration.

Likewise, migrate unique behavior from `wsi_extract_tile_embeddings.py` and
`wsi_extract_fm_outputs.py` into implementations of `TileTeacherAdapter` and
`WSITeacherAdapter`, writing the cache contracts in `src/wsi_pipeline/cache_io.py`.

**Tile side done (2026-08-07):** `wsi_extract_tile_embeddings.py` and its underlying
`ConchV15MultiLayerEncoder`/`TimmViTMultiLayerEncoder`/`tile_extraction.py` (no attention
support, and never actually validated against the real gated CONCH v1.5 checkpoint) have
been removed. `HookedViTTileTeacherAdapter` (`TileTeacherAdapter`) + `python scripts/eaf.py
cache tile` is now the sole tile-cache entry point, numerically validated end to end
against real HISTAI slides — see `docs/offline_eaf_pipeline.md`. `wsi_extract_fm_outputs.py`
(WSI-level FM outputs, a separate concern) was not touched by this pass.

## Legacy cleanup candidates

After references/tests have been audited, remove manual/debug-only entry points whose
functionality is covered by the core/tests, including the current WSI attention audit and
smoke-script family:

```text
scripts/run_generic_import_smoke.sh
scripts/run_trident_import_smoke.sh
scripts/run_wsi_attention_embedding_audit.sh
scripts/analyze_wsi_attention_embeddings.py
scripts/wsi_analyze_centroid_attention.py
scripts/wsi_analyze_majority_attention.py
scripts/wsi_audit_titan_attention_structure.py
scripts/wsi_complete_titan_attention_audit.py
```

Also review `train_classifier.py` and old WSI forecaster/import entry points. Delete
them only if no active downstream benchmark depends on them. Keep real `pytest` tests;
remove manual smoke/debug scripts, not automated coverage.

`train_forecaster.py`, `train_multi_thunder_forecaster.py`, `finetune_pruned.py`,
`finetune_multi_thunder_pruned.py`, and the older cache-based tile-EAF pair
(`build_wsi_tile_eaf_cache.py`/`train_forecaster_from_cache.py`,
`build_thunder_online_forecaster_cache.py`/`train_thunder_online_forecaster.py`,
`compare_thunder_online_runs.py`) were removed in the tile-EAF minimal-pipeline
refactor (2026-08-24) — all were compatibility shims or superseded by
`train_wsi_tile_eaf_online.py` / `finetune_wsi_tile_encoder_pruned_online.py`, which
already unify the online and cache-backed (`--target-cache-index`) code paths.
`scripts/wsi_prepare_strict_pretraining.py` was removed the same pass; use
`python scripts/eaf.py data ...` directly.

Before each deletion group run:

```bash
grep -R "<script-name>" -n README.md docs tests scripts src 2>/dev/null || true
```

and migrate any still-valid documentation/test reference first.

## Validation before commit

At minimum:

```bash
python -m py_compile scripts/eaf.py \
  src/data/wsi/layout.py src/data/wsi/manifest.py src/data/wsi/corpora.py \
  src/wsi_pipeline/cache_contracts.py src/wsi_pipeline/cache_io.py \
  src/wsi_pipeline/model_adapters.py src/wsi_pipeline/offline_cache.py \
  src/wsi_pipeline/archive.py

pytest tests/data/wsi/test_refactor_layout.py \
       tests/wsi_pipeline/test_cache_contracts.py \
       tests/wsi_pipeline/test_pixel_archive.py -q

python scripts/eaf.py layout --data-root "$EAF_WSI_ROOT"
python scripts/eaf.py data plan-histai --help
python scripts/eaf.py data plan-gtex --help
```

Finally audit that no runtime data are staged:

```bash
git status --short
git diff --cached --stat
git diff --cached --name-only | grep -E '(^|/)(sources|datasets|caches|archives|checkpoints|results|logs)/' && exit 1 || true
```
