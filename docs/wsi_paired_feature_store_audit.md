# Audit — Paired Feature Store Support for WSI Tile Attention Forecasting

Branch: `feat/wsi-tile-attention-forecaster` (current branch; the task brief referenced
`feat/wsi-tile-importance-forecaster` — treated as the same effort under a different name).

Read-only audit. No files were modified while producing this report.

---

## 1. File map

### Core data contracts

| File | Role |
|---|---|
| `src/data/wsi/bag.py` | `WSIBag` dataclass: `slide_id`, `tile_features [N,D]`, `coords [N,2\|4]`, `label`, `attention [N]`, `metadata`. Couples features and attention target in one object. |
| `src/data/wsi/feature_store.py` | `WSIFeatureStore` ABC (`slide_ids`, `exists`, `read`, `write`) + `InMemoryWSIFeatureStore`. |
| `src/data/wsi/h5_feature_store.py` | `H5WSIFeatureStore` — canonical HDF5 backend. One group per slide holding `tile_features`, optional `coords`, optional `attention`, label attrs, `metadata_json`. |
| `src/data/wsi/dataset.py` | `WSIBagDataset` ABC, `InMemoryWSIBagDataset`, `FeatureStoreWSIBagDataset` (lazy read from **one** store by slide id). |
| `src/data/wsi/batch.py` | `PaddedWSIBatch` + `pad_wsi_bags()` — pads a list of `WSIBag` into `[B, max_tiles, D]` tensors; requires all-or-nothing `coords`/`attention` across the batch. |
| `src/data/wsi/collate.py` | `collate_wsi_bags`, `collate_padded_wsi_bags` — DataLoader collate functions wrapping `pad_wsi_bags`. |
| `src/data/wsi/generic_features.py`, `src/data/wsi/trident.py` | Import single-slide feature/coords files (`.pt/.npy/.npz`, TRIDENT HDF5) into a `WSIBag`. Both always set `attention=None` — these are the "import TRIDENT/generic" entry points. |
| `src/data/wsi/__init__.py` | Public surface; re-exports the above. No paired-store symbols exist yet. |

### Model / training

| File | Role |
|---|---|
| `src/models/wsi/tile_attention_forecaster.py` | `WSITileAttentionForecaster` (transformer over tile features → per-tile score) + `wsi_attention_kl_loss`. |
| `src/models/wsi/checkpoint.py` | `WSITileAttentionForecasterConfig`/checkpoint save-load. |
| `src/models/wsi/abmil.py`, `abmil_checkpoint.py` | ABMIL teacher (produces the attention target). |
| `src/training/wsi/attention_forecasting.py` | `run_wsi_attention_forecasting_batch` — **hard-requires `batch.attention is not None`** (line 68). One train/eval epoch loop. |
| `src/training/wsi/abmil.py` | ABMIL teacher training loop. |
| `src/evaluation/wsi_attention_metrics.py` | Spearman, top-k overlap, NDCG@k between predicted scores and target attention. |

### CLI scripts (`scripts/`)

| Script | Role | Store(s) used |
|---|---|---|
| `build_trident_manifest.py`, `import_trident_feature_store.py` | TRIDENT import → single `H5WSIFeatureStore` | 1 (output) |
| `build_generic_feature_manifest.py`, `import_generic_feature_store.py` | generic `.pt/.npy/.npz` import → single store | 1 (output) |
| `create_synthetic_wsi_feature_store.py` | synthetic store with features **and** attention already baked in (softmax of a fixed linear projection of the features) | 1 |
| `validate_wsi_feature_store.py` | single-store schema/NaN/shape validation | 1 |
| `inspect_wsi_feature_store.py` | single-store stats/report JSON | 1 |
| `split_wsi_feature_store.py` | train/val/test split file generator (see §5) | 1 |
| `train_wsi_abmil.py` | ABMIL teacher training | 1 (features+label) |
| `extract_wsi_abmil_attention.py` | reads features from `--input-feature-store`, runs ABMIL, **writes a new store** with `tile_features` + `attention` copied together | 2 args, but output is always a fused single store |
| `train_wsi_attention_forecaster.py` | forecaster training — **one `--feature-store` arg**, `H5WSIFeatureStore(args.feature_store)`, needs both `tile_features` and `attention` in it (line 255 in the script) | 1 |
| `evaluate_wsi_forecaster_pruning.py`, `evaluate_wsi_abmil_pruning_agreement.py` | pruning-quality evaluation | 1 |
| `create_pruned_wsi_feature_store.py` | materializes a pruned store; reads one input store (features+attention), writes one output store | 1 |
| `plot_wsi_pruning_curves.py` | reporting only, no store I/O |
| `run_wsi_synthetic_e2e_smoke.sh`, `run_trident_import_smoke.sh`, `run_generic_import_smoke.sh` | shell smoke scripts driving the above CLIs end to end |

### Tests

`tests/data/test_wsi_{bag,batch,collate,dataset,feature_store,h5_feature_store}.py`,
`tests/models/test_wsi_{abmil,abmil_checkpoint,tile_attention_forecaster,tile_attention_forecaster_checkpoint}.py`,
`tests/training/test_wsi_{abmil_training,attention_forecasting,attention_forecasting_smoke}.py`,
`tests/scripts/test_*.py` (one per script above, mostly subprocess-driven CLI tests using `tmp_path` + `H5WSIFeatureStore`).

---

## 2. How `attention` is read/written today

- **Write**: `H5WSIFeatureStore.write()` (`src/data/wsi/h5_feature_store.py:124-130`) — `attention` is an optional dataset inside the *same* per-slide HDF5 group as `tile_features`. There is no separate group/file concept.
- **Read**: `H5WSIFeatureStore.read()` (lines 67-84) pulls `tile_features` and `attention` from the same `group` in one call.
- **Produced by**: `extract_wsi_abmil_attention.py` — reads a features-only store, runs the ABMIL teacher, and writes `tile_features` (copied verbatim) + freshly computed `attention` into one new fused store.
- **Consumed by**: `train_wsi_attention_forecaster.py` (single `--feature-store`), `evaluate_wsi_forecaster_pruning.py`, `evaluate_wsi_abmil_pruning_agreement.py`, `create_pruned_wsi_feature_store.py` — all single-store.
- **Batched form**: `PaddedWSIBatch.attention` (`src/data/wsi/batch.py:35`), produced by `pad_wsi_bags`, which enforces "all bags have attention or none" (line 196-197) — this rule assumes attention travels with the same bag as `tile_features`, not from a second source.
- **Loss**: `wsi_attention_kl_loss` (`tile_attention_forecaster.py:190`) takes `target_attention` as a plain tensor argument — it is agnostic to where the tensor came from, so **no change needed there**.

## 3. Where the single-store assumption is baked in

1. `WSIBag` (`bag.py`) — one dataclass, one `tile_features`, one `attention`. This is the tightest coupling point; it is the *unit of read/write* for `H5WSIFeatureStore`.
2. `FeatureStoreWSIBagDataset.__getitem__` (`dataset.py:110-113`) calls `self.store.read(slide_id)` on a single `store` — no notion of a second store.
3. `train_wsi_attention_forecaster.py:255` — `store = H5WSIFeatureStore(args.feature_store)`; the same `store` provides both the input to the model and `batch.attention` for the loss.
4. `run_wsi_attention_forecasting_batch` (`attention_forecasting.py:67-68`) — requires `batch.attention is not None`, and `batch.attention` only exists because it rode along inside `WSIBag`/`PaddedWSIBatch`.
5. `pad_wsi_bags` (`batch.py:194-197`) — the all-or-nothing coords/attention rule is fine to keep, but it operates on a list of *already-fused* `WSIBag` objects; a paired loader must fuse two stores into one `WSIBag` (or a new `PairedWSIBag`) **before** this function runs.
6. `create_pruned_wsi_feature_store.py` and `extract_wsi_abmil_attention.py` both read one input store and write one output store that mixes features + attention — this pattern is reasonable to keep for those specific scripts (they are "materialize a fused store" tools by design) but should not be the only way to train the forecaster.

## 4. Where to insert paired-store support

- **New module** `src/data/wsi/paired_feature_store.py` — a `load_paired_wsi_bag(input_store, target_store, slide_id, alignment_mode, require_coords) -> PairedWSIBag` function plus a `PairedWSIBag` dataclass (`slide_id`, `input_features`, `target_importance`, `coords`, `label`, `metadata`, `target_metadata`). This sits next to `bag.py`, does not modify it.
- **New module** `src/data/wsi/coord_alignment.py` — pure-tensor coordinate matching (`align_by_coords(input_coords, target_coords) -> index arrays`), used by `paired_feature_store.py` and reusable by the CLI validator.
- **Dataset**: rather than editing `FeatureStoreWSIBagDataset`, add a small `PairedFeatureStoreWSIBagDataset` (same shape as `FeatureStoreWSIBagDataset` but holds `input_store` + `target_store` and calls `load_paired_wsi_bag`). It can still yield ordinary `WSIBag` objects (with `tile_features=input_features`, `attention=target_importance`) so **`pad_wsi_bags`/`collate_padded_wsi_bags`/`run_wsi_attention_forecasting_batch` need zero changes** — this is the key insight that keeps the blast radius small.
- **CLI**: `train_wsi_attention_forecaster.py` and `create_pruned_wsi_feature_store.py` gain `--input-feature-store`/`--target-feature-store`/`--alignment-mode`/`--require-coords`, with `--feature-store` kept as a legacy alias that sets both input and target to the same path. `evaluate_wsi_forecaster_pruning.py`/`evaluate_wsi_abmil_pruning_agreement.py` are natural follow-ups but are **not** in the minimal scope requested.
- **New standalone CLI** `scripts/validate_wsi_paired_feature_stores.py` — does not need to touch any existing script.

## 5. Existing split utilities

`scripts/split_wsi_feature_store.py` reads **one** `H5WSIFeatureStore`, computes `train/val/test` slide-id lists (optionally stratified by label), and writes `train.txt`/`val.txt`/`test.txt` + `split_summary.json` to `--output-dir`. Training CLIs (`train_wsi_abmil.py`, `train_wsi_attention_forecaster.py`) consume these via `--split-dir` (mutually exclusive with explicit `--train-slide-ids-file`/`--val-slide-ids-file`). Splitting is slide-id-based and store-agnostic — it only needs `slide_ids()` and optionally `label`, so it will work unchanged once paired stores exist, **as long as both stores share the same slide ids**. No changes needed here for the minimal paired-store scope.

## 6. Test framework

Pure `pytest` (`pytest.importorskip("h5py")` gating h5py-dependent tests). CLI scripts are tested via `subprocess.run([sys.executable, "scripts/x.py", ...], cwd=repo_root)` reading JSON stdout and `returncode`. There are also three bash smoke scripts (`run_*_smoke.sh`) that chain multiple CLIs against a synthetic store, wrapped by matching `tests/scripts/test_run_*_smoke.py`. Both pytest and shell-smoke conventions exist; new work should follow the pytest + subprocess-CLI pattern used by `tests/scripts/test_validate_wsi_feature_store.py`.

## 7. Tests to update

- `tests/scripts/test_train_wsi_attention_forecaster.py` — add cases for `--input-feature-store`/`--target-feature-store` alongside the existing `--feature-store` legacy-alias case (no removal of legacy coverage).
- `docs/wsi_attention_forecasting.md` / `docs/wsi_cli_reference.md` — need a new section once CLIs gain the new flags (documentation, not test, but flagged since it's a "Keep" file per `docs/wsi_scripts_inventory.md`).

## 8. Tests to create

- `tests/data/test_wsi_paired_feature_store.py` — `load_paired_wsi_bag` unit tests (see Definition-of-Done list in the implementation task: legacy single-store, index-aligned, coords-aligned, tile-count mismatch, coords mismatch, duplicate coords, NaN target).
- `tests/data/test_wsi_coord_alignment.py` — pure alignment-function unit tests (duplicates, partial coverage, dtype/ordering).
- `tests/scripts/test_validate_wsi_paired_feature_stores.py` — CLI smoke tests, mirroring `test_validate_wsi_feature_store.py`'s subprocess pattern.

## 9. Risks

1. **Silent attention/feature mismatch** is the central risk the brief calls out — any implementation must raise, never coerce/truncate/reindex silently. `WSIBag`/`pad_wsi_bags` already have this philosophy (e.g. `pad_wsi_bags` line 196: "either all bags must have X or none" — fail loud, no silent drop); the new code must match that style.
2. **`index` alignment mode is a trap for real data**: TRIDENT/generic imports do not guarantee stable tile ordering across two independently-produced stores (e.g. one exported by TRIDENT patching, the other re-derived at a different resolution). `index` mode must strictly check both `slide_id` presence and `n_tiles` equality and document that it does **not** check coordinate equality — silently misaligning same-count-but-different-order tiles is the worst-case failure mode and cannot be fully prevented by `index` mode; this should be called out in the CLI `--help` text and docs.
3. **`coords` mode dtype/rounding**: existing coords are stored as `torch.long` (`generic_features.py:197`, `trident.py:169`) after truncation from possibly-float sources — safe for exact-integer coordinate grids but any resampled/rescaled coordinate systems between two stores (e.g. different tiling resolution) will not align even though they "should." Must fail rather than fuzzy-match.
4. **Feature dim conflation**: `H5WSIFeatureStore` has no schema field distinguishing "early" vs "late" features — a `--input-feature-dim` CLI flag is necessary to catch the case where the wrong store is passed as input vs target-adjacent materialize store.
5. **`create_pruned_wsi_feature_store.py`/`extract_wsi_abmil_attention.py` reuse**: these scripts intentionally fuse features+attention into one output store — that is a valid pattern (materializing a self-contained store for downstream single-store consumers) and should not be forced into the paired-store shape.
6. **Backward compatibility**: `H5WSIFeatureStore`'s physical HDF5 layout (schema_version=1) must not change. Paired-store support is purely a *loading-time* concept (join two independent `WSIFeatureStore` instances by slide_id/coords) — nothing here requires a new HDF5 schema.

## 10. Proposed implementation — 6 commits

1. `feat(wsi): add coord_alignment utility` — `src/data/wsi/coord_alignment.py` + unit tests. Pure function, no dependents yet.
2. `feat(wsi): add PairedWSIBag and load_paired_wsi_bag` — `src/data/wsi/paired_feature_store.py`, `PairedFeatureStoreWSIBagDataset`, exported from `src/data/wsi/__init__.py`. Unit tests (legacy/index/coords/mismatch/duplicate/NaN cases).
3. `feat(wsi): support --input-feature-store/--target-feature-store in train_wsi_attention_forecaster.py` — add flags, keep `--feature-store` as legacy alias, wire `PairedFeatureStoreWSIBagDataset` when paired flags are given. Update/extend `tests/scripts/test_train_wsi_attention_forecaster.py`.
4. `feat(wsi): add validate_wsi_paired_feature_stores.py CLI` — standalone script + JSON report + tests.
5. `feat(wsi): support --input-feature-store/--target-feature-store in create_pruned_wsi_feature_store.py` — same alias pattern, matching test updates. *(Optional — only if in scope; the brief's Definition of Done does not strictly require this script.)*
6. `docs(wsi): document paired feature stores` — extend `docs/wsi_attention_forecasting.md` and `docs/wsi_cli_reference.md` with the new flags and the `input/target/materialize` schema.

(Steps 3 and 5 can be dropped to a 4-commit minimal scope if only the validator + data-layer utility are required; the user's Task 2 "Definition of done" only strictly requires the validator script and pairing tests, not CLI wiring in the training script — flagged as a scope decision, see report footer.)

## 11. Recommended API / CLI

```python
# src/data/wsi/paired_feature_store.py
@dataclass(frozen=True)
class PairedWSIBag:
    slide_id: str
    input_features: torch.Tensor        # [N, D_early]
    target_importance: torch.Tensor     # [N]
    coords: torch.Tensor | None
    label: int | float | torch.Tensor | None
    metadata: dict | None
    target_metadata: dict | None

def load_paired_wsi_bag(
    input_store: WSIFeatureStore,
    target_store: WSIFeatureStore,
    slide_id: str,
    *,
    alignment_mode: str = "index",   # "index" | "coords"
    require_coords: bool = False,
) -> PairedWSIBag: ...
```

```bash
python scripts/validate_wsi_paired_feature_stores.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_wsi_importance.h5 \
  --input-feature-dim 384 \
  --alignment-mode coords \
  --require-coords \
  --require-attention \
  --output-json reports/paired_validation.json
```
Exit 0 on success, 1 on any mismatch. JSON report: `n_slides`, `n_tiles_{min,mean,max}`, `alignment_mode`, `feature_dim`, `target_coverage`, `coords_coverage`, `mismatch_examples` (capped list).

## 12. Minimal tests (Definition of Done)

- legacy single-store still valid (`input_store is target_store`, or same underlying file)
- paired stores, index-aligned
- paired stores, coords-aligned
- mismatch tile count → raises
- mismatch coords → raises
- duplicate coords → raises
- NaN/Inf target → raises (already covered at the `WSIBag`/`PaddedWSIBatch` level for the *fused* representation; needs an equivalent pre-fusion check in `load_paired_wsi_bag` since a `PairedWSIBag` should validate before construction reuses `WSIBag`'s own checks)

## 13. Validation commands

```bash
python -m pytest tests/data/test_wsi_paired_feature_store.py tests/data/test_wsi_coord_alignment.py -v
python -m pytest tests/scripts/test_validate_wsi_paired_feature_stores.py -v
python -m pytest tests/data tests/models tests/training tests/scripts -k wsi -v   # full WSI regression
python scripts/create_synthetic_wsi_feature_store.py --output /tmp/eaf_wsi_audit/input.h5 --n-slides 6 --feature-dim 16
python scripts/validate_wsi_paired_feature_stores.py --input-feature-store /tmp/eaf_wsi_audit/input.h5 --target-feature-store /tmp/eaf_wsi_audit/input.h5 --input-feature-dim 16 --alignment-mode index --require-attention
```

## 14. Do NOT modify

- `src/data/wsi/bag.py`, `H5WSIFeatureStore`'s on-disk schema (`_SCHEMA_VERSION`, group layout) — no physical format changes.
- `wsi_attention_kl_loss`, `WSITileAttentionForecaster`, `run_wsi_attention_forecasting_batch` — these are store-agnostic and already work with any `WSIBag`-shaped input; paired-store support should feed them the same way, not fork them.
- `extract_wsi_abmil_attention.py`, `create_pruned_wsi_feature_store.py`'s existing single-store behavior when called with the legacy flags — only additive changes.
- `scripts/split_wsi_feature_store.py` — store-agnostic already, no changes needed.
- Do not rename the physical `attention` HDF5 dataset/field anywhere — it stays `attention` on disk; "tile_importance_target" is a semantic/API-level name only (`PairedWSIBag.target_importance`), not a storage rename.
