# WSI Tile Importance Forecasting

This document explains the generalized tile-importance forecasting pipeline
(`WSITileImportanceForecaster`, paired feature stores) that sits alongside
the original single-store attention-forecasting pipeline described in
`docs/wsi_attention_forecasting.md`. Read that document first for the core
data contract (`WSIBag`, `H5WSIFeatureStore`) and the end-to-end ABMIL
pipeline; this document focuses on what changes when the *target* is not
necessarily ABMIL attention, and when it lives in a store separate from the
input features. For the concrete first real-data runbook, see
`docs/wsi_first_real_experiment.md`.

---

## 1. Attention matrix vs. scalar tile importance

These are two different things that are easy to conflate because both are
sometimes casually called "attention":

- **Tile-encoder attention matrix**: the internal self-attention weights
  inside a transformer-based tile/patch encoder (e.g. a ViT patch encoder,
  or attention between tiles inside a TRIDENT slide encoder). This is an
  `[n_tiles, n_tiles]` (or `[n_heads, n_tiles, n_tiles]`) matrix internal to
  one forward pass of one model. EAF's WSI pipeline does not consume this
  matrix directly — it is usually not exposed by pooling-style slide
  encoders, and even when it is, it describes token-to-token interaction,
  not "how much does this tile matter for the slide-level prediction."
- **Scalar WSI tile importance**: a single non-negative scalar per tile,
  `[n_tiles]`, meant to answer "how much should this tile matter when
  pruning or explaining the slide-level prediction." This is what
  `WSITileImportanceForecaster` predicts and what pruning is driven by. ABMIL
  gated attention is one *source* of this scalar (its per-tile gate output
  is already `[n_tiles]`), but it is not the only one — a WSI foundation
  model tile score, a saliency method, or any other precomputed per-tile
  value is equally valid input.

`WSITileImportanceForecaster` only ever consumes/produces the second kind:
a flat `[n_tiles]` vector. If a target source only exposes an
attention *matrix* (encoder self-attention), it must first be reduced to a
per-tile scalar (e.g. attention received, summed or averaged over queries)
before it fits this pipeline — that reduction is target-source-specific and
out of scope for this pipeline itself.

---

## 2. Why the physical field is still called `attention`

`WSIBag.attention` and the `attention` HDF5 dataset name are **not**
renamed anywhere in this pipeline, even though `WSITileImportanceForecaster`
uses the field to hold a generic tile-importance target rather than
specifically ABMIL attention. This is intentional:

- `H5WSIFeatureStore`'s on-disk schema is unchanged (`schema_version=1`,
  same per-slide group layout as before). Renaming the field would be a
  breaking storage-format change and would fork read/write code for no
  functional benefit.
- `pad_wsi_bags`, `collate_padded_wsi_bags`, and the training loop in
  `src/training/wsi/attention_forecasting.py` all already work against
  `WSIBag.attention`/`PaddedWSIBatch.attention` — reusing the same field
  means paired-store support required zero changes to any of that code.
- The semantic name lives one layer up: `PairedWSIBag.target_importance`
  (`src/data/wsi/paired_feature_store.py`) is the field importers, trainers,
  and evaluators actually reason about. `PairedWSIBag.to_wsi_bag()` maps
  `target_importance -> WSIBag.attention` at the boundary, so downstream
  code never has to know or care that the physical dataset is named
  `attention`.
- Checkpoint metadata written by `save_wsi_tile_importance_forecaster_checkpoint`
  (`src/models/wsi/checkpoint.py`) records `target_type="tile_importance"`
  and `target_source` (e.g. `"abmil"`, `"gigapath_wsi_fm"`,
  `"synthetic_importance_from_late"`) so the *semantic* provenance is always
  explicit even though the *physical* field name is not.

In short: `attention` is a storage-layer name with ABMIL-era history;
`tile_importance_target` is the semantic contract this pipeline actually
implements. Treat them as the same bits, different vocabulary for different
audiences.

---

## 3. Single-store (legacy) mode vs. paired-store mode

Both modes are fully supported and neither is deprecated.

### 3.1 Single-store (legacy) mode

One `H5WSIFeatureStore` provides both `tile_features` (model input) and
`attention` (training target) for every slide — typically produced by
`extract_wsi_abmil_attention.py`, which reads a features-only store, runs a
trained ABMIL teacher, and writes a new store with the original
`tile_features` plus freshly computed `attention` alongside them.

```bash
python scripts/train_wsi_attention_forecaster.py \
  --feature-store data/features_real_abmil_attention.h5 \
  --output-dir checkpoints/exp001_forecaster \
  --feature-dim 1024 --epochs 20 --device cuda --split-dir splits/exp001
```

`scripts/train_wsi_importance_forecaster.py` and
`scripts/evaluate_wsi_importance_pruning.py` also accept this mode via the
`--feature-store` alias, which sets both the input and target store to the
same path — use this when you have no reason to separate input features
from the importance target.

### 3.2 Paired-store mode

Input tile features and the importance target live in two independently
produced `H5WSIFeatureStore` files, joined at load time by
`load_paired_wsi_bag` (`src/data/wsi/paired_feature_store.py`) — nothing
about the physical HDF5 format changes; pairing is a *loading-time* join by
slide id, not a new storage format.

```bash
python scripts/train_wsi_importance_forecaster.py \
  --input-feature-store data/features_layer2.h5 \
  --target-feature-store data/features_wsi_importance.h5 \
  --output-dir checkpoints/exp001_importance_forecaster \
  --input-feature-dim 384 \
  --alignment-mode coords \
  --epochs 20 --device cuda --split-dir splits/exp001
```

Two alignment modes control how tiles in the two stores are matched up:

- `index` (default): requires identical slide id, identical tile count, and
  (if both sides carry coords) identical tile order. This is a real trap for
  independently produced stores — two pipelines rarely guarantee stable tile
  ordering — so treat `index` as safe only when both stores were written by
  the same code path in the same pass.
- `coords`: aligns tiles by exact coordinate match
  (`src/data/wsi/coord_alignment.py`), and requires coords on both sides.
  This is the recommended mode whenever the input and target stores were
  produced by separate pipeline runs.

Mismatches (wrong tile count, missing coords, duplicate coords, incomplete
coordinate coverage, a slide missing from either store) always raise —
nothing is silently dropped, truncated, or reindexed. Validate a pair before
training with `scripts/validate_wsi_paired_feature_stores.py`.

`create_pruned_wsi_feature_store.py` extends the same idea one step further:
tiles are *selected* using one store (e.g. cheap early-layer features) and
the corresponding tiles are *materialized* from a different store (e.g.
expensive late-layer features for a downstream WSI model), using the same
`index`/`coords` alignment machinery.

---

## 4. Where the importance target comes from

`src/models/wsi/importance_providers.py` defines `WSIImportanceProvider`,
the interface for obtaining a tile-importance target from a `WSIBag`:

- `PrecomputedImportanceProvider` — looks up a slide id in a target store,
  typically one produced by `scripts/import_wsi_importance_targets.py` from
  externally computed per-slide artifacts (TRIDENT-style HDF5, `.pt`,
  `.npy`, `.npz`).
- `ABMILImportanceProvider` — wraps a trained `ABMILClassifier` and returns
  its per-tile gated-attention output as the target.
- `TridentSlideEncoderImportanceProvider` — a **documented stub**. Most
  TRIDENT slide encoders pool tiles into a slide embedding without exposing
  a retrievable per-tile attention weight, so this class raises
  `NotImplementedError` rather than guessing at an extraction strategy. Use
  one of the two providers above until a specific TRIDENT slide encoder with
  a documented, extractable tile-level score is wired in.

---

## 5. TRIDENT: recommended backend, not a hard dependency

TRIDENT is the recommended way to go from raw WSI files (`.svs`, `.ndpi`,
`.tiff`, ...) to per-slide tile features in the first place — this pipeline
does not implement raw WSI tiling or patch-encoder inference itself (see
Limitations below). Once TRIDENT (or an equivalent pipeline) has produced
per-slide feature/coordinate HDF5 files, `build_trident_manifest.py` +
`import_trident_feature_store.py` bring them into an `H5WSIFeatureStore`,
and, if TRIDENT (or anything else) also produced a per-tile importance
score, `build_wsi_importance_manifest.py` + `import_wsi_importance_targets.py`
bring that in as a target store.

None of this requires a live TRIDENT installation:

- `src/data/wsi/trident.py` reads TRIDENT-*style* per-slide HDF5 files
  (features/coords) directly with `h5py`; it never imports the `trident`
  package.
- `src/data/wsi/importance_target_file.py` (used by
  `import_wsi_importance_targets.py`) reads target artifacts the same way —
  HDF5, `.pt`/`.pth`, `.npy`, `.npz` — with no TRIDENT import.
- `TridentSlideEncoderImportanceProvider` is the one place that *would*
  import `trident`, and it does so lazily, only inside
  `compute_tile_importance`, and only to raise a clear `NotImplementedError`
  today. Importing `src.models.wsi` or `src.data.wsi` never requires TRIDENT
  to be installed.

So: TRIDENT is the recommended real-data path for feature extraction, but
every CLI and module in this pipeline works against plain HDF5/`.pt`/`.npy`
artifacts and has no import-time or runtime dependency on the `trident`
package.

---

## 6. Full CLI example (synthetic, no dataset required)

```bash
bash scripts/run_wsi_tile_importance_synthetic_smoke.sh
```

This exercises the whole paired-store pipeline on CPU with synthetic data —
useful both as a smoke test and as an executable reference for the full
command sequence. It:

1. Generates three *paired* synthetic stores in one deterministic pass —
   `scripts/create_synthetic_wsi_importance_stores.py` — an early store
   (`features_layer2.h5`), a late store (`features_late.h5`), and an
   importance-target store (`features_synthetic_importance.h5`) correlated
   with the *late* features: `importance_i = softmax(w^T late_feature_i +
   noise_i)`. All three share identical slide ids, tile counts, and
   coordinates by construction — see the script docstring for why three
   independent `create_synthetic_wsi_feature_store.py` calls cannot
   guarantee this.
2. Validates each store individually (`validate_wsi_feature_store.py`) and
   the early/importance pair (`validate_wsi_paired_feature_stores.py`,
   `--alignment-mode coords`).
3. Creates a persistent train/val/test split from the early store
   (`split_wsi_feature_store.py`).
4. Trains `WSITileImportanceForecaster` for one epoch on CPU
   (`train_wsi_importance_forecaster.py`, input = early store, target =
   importance store, `--alignment-mode coords`).
5. Evaluates pruning quality (`evaluate_wsi_importance_pruning.py`).
6. Builds a pruned **late**-feature store by *selecting* tiles from the
   **early** store and *materializing* the corresponding tiles from the
   **late** store (`create_pruned_wsi_feature_store.py
   --selection-feature-store ... --materialize-feature-store ...`) — the
   scenario this whole early/late split exists for: score cheaply, pay the
   expensive feature cost only for the tiles you keep.
7. Validates the pruned store.

Environment variables (`WORKDIR`, `EARLY_FEATURE_DIM`, `LATE_FEATURE_DIM`,
`N_SLIDES`, `FORECASTER_EPOCHS`, `DEVICE`, ...) override defaults; see the
script header. Manual equivalent of steps 1–2:

```bash
python scripts/create_synthetic_wsi_importance_stores.py \
  --early-output /tmp/wsi_imp/features_layer2.h5 \
  --late-output /tmp/wsi_imp/features_late.h5 \
  --importance-output /tmp/wsi_imp/features_synthetic_importance.h5 \
  --n-slides 8 --early-feature-dim 8 --late-feature-dim 16 \
  --min-tiles 4 --max-tiles 8 --noise-std 0.1 --seed 0

python scripts/validate_wsi_paired_feature_stores.py \
  --input-feature-store /tmp/wsi_imp/features_layer2.h5 \
  --target-feature-store /tmp/wsi_imp/features_synthetic_importance.h5 \
  --input-feature-dim 8 --alignment-mode coords \
  --require-coords --require-attention
```

For the pre-existing legacy/ABMIL smoke pipeline, see
`docs/wsi_attention_forecasting.md` §13 and
`scripts/run_wsi_synthetic_e2e_smoke.sh` — it is unmodified and still
exercises the single-store ABMIL-teacher path end to end.

---

## 7. Current limitations

- Feature-level only: no raw WSI tiling, no patch-encoder inference, no
  multi-resolution extraction. Real experiments need TRIDENT (or an
  equivalent) to produce the input feature/coords/importance-target files
  this pipeline imports.
- `TridentSlideEncoderImportanceProvider` is a stub, not a working live
  importance source, for slide encoders that pool tiles without exposing
  tile-level attention.
- `index` alignment mode does not check coordinate equality when only one
  side has coords, and is generally unsafe across independently produced
  stores — prefer `coords` alignment for real (non-single-pipeline-pass)
  data.
- `create_synthetic_wsi_importance_stores.py` and
  `run_wsi_tile_importance_synthetic_smoke.sh` are synthetic sanity checks
  only; the correlated-noise target they generate has no relationship to
  any real biological signal and should not be used to draw conclusions
  about model quality.
- No experiment runner or report generator yet — CSV/JSON outputs from the
  evaluation scripts are the current reporting surface.

See `docs/wsi_attention_forecasting.md` §15 and
`docs/wsi_paired_feature_store_audit.md` for the broader pipeline
limitations and design audit that this pipeline builds on.
