# Canonical WSI data layout

The repository contains **code and documentation only**. Raw slides, TRIDENT
coordinates, offline teacher caches, cold pixel archives, model checkpoints, and
runtime logs live in a single external, versioned data root, `$EAF_WSI_ROOT`.

The current canonical root on the shared server is:

```text
/data2/home/vcivale/projects/imaging/data/WSI
```

Set it once per shell:

```bash
export EAF_WSI_ROOT=/data2/home/vcivale/projects/imaging/data/WSI
```

All commands accept `--data-root` (defaulting to `$EAF_WSI_ROOT`), so another
machine can use the same layout under a different absolute path. Do not
introduce a second top-level data layout.

The central invariant is that **pretraining corpora and downstream evaluation
corpora are different namespaces**. A dataset reserved for downstream
evaluation must never be silently pulled into an EAF pretraining manifest.

## Canonical store

```text
$EAF_WSI_ROOT/
├── catalog/
│   ├── slides.csv
│   ├── artifacts.csv
│   ├── datasets.csv
│   └── encoders.csv
├── sources/                              # one physical copy of recoverable raw data
│   ├── gdc/tcga/<cohort>/<diagnostic|tissue>/*.svs
│   ├── huggingface/hest/...              # existing HEST acquisition
│   ├── histai/<subset>/...               # new: HISTAI H&E WSIs
│   └── gtex/...                          # new: IDC DICOM series (keep siblings together)
├── datasets/
│   ├── pretraining/                      # task-agnostic EAF training corpora
│   │   ├── tcga_eaf_multicohort_v1/      # existing; online tile-EAF training
│   │   │   ├── dataset.yaml
│   │   │   ├── manifests/
│   │   │   │   ├── slides.csv
│   │   │   │   └── trident_pending_*.csv
│   │   │   ├── views/
│   │   │   │   └── raw_flat/*.svs -> ../../../../../sources/...
│   │   │   └── artifacts/
│   │   │       ├── trident/
│   │   │       │   ├── contours/
│   │   │       │   ├── contours_geojson/
│   │   │       │   ├── thumbnails/
│   │   │       │   ├── wsi_states/
│   │   │       │   └── 20x_512px_0px_overlap/
│   │   │       │       ├── patches/*_patches.h5
│   │   │       │       └── visualization/
│   │   │       └── tile_fm/
│   │   │           └── conch_v15/        # retained legacy features
│   │   ├── hest_eaf_thunder_clean_v1/    # existing curated HEST corpus (untouched)
│   │   ├── histai_eaf_wsi_v1/            # new: one H&E per (subset, case_id)
│   │   ├── gtex_eaf_wsi_v1/              # new: one WSI per SeriesInstanceUID
│   │   ├── hest_eaf_wsi_v1/              # new: lightweight HEST registration (optional)
│   │   └── eaf_wsi_pretrain_strict_v1/   # logical union (HISTAI ∪ HEST), no duplicated pixels, TCGA excluded
│   └── downstream/                       # labelled benchmark bank, unseen by strict pretraining
│       ├── tile_level/<benchmark>/
│       └── wsi_level/<benchmark>/
├── caches/                               # NEW: frozen teacher outputs for offline EAF training
│   ├── tile_eaf/<dataset>/<tile_encoder>/<cache_id>/
│   └── wsi_eaf/<dataset>/<tile_encoder>__<wsi_encoder>/<cache_id>/
├── archives/                             # NEW: cold tissue-pixel archives after segmentation
│   ├── histai/
│   └── gtex/
├── checkpoints/
├── results/
├── logs/
├── cache/                                # legacy scratch/thumbnail cache (unrelated to caches/)
└── quarantine/
```

Raw WSI files have exactly one physical copy under `sources/`. Dataset views
and compatibility paths are symlinks; scripts must not copy slides into the
repository or duplicate multi-GB pixel data to satisfy a naming convention.

## Dataset roles

`pretraining` means labels are never required by the EAF objective. **The EAF
training corpus policy is HISTAI + GTEx + HEST only; TCGA is explicitly
excluded from EAF pretraining** (see "WSI EAF Training Data Policy" in
`CLAUDE.md`). Existing TCGA data are **not deleted or moved**; the strict
protocol (`eaf_wsi_pretrain_strict_v1`, default `sources=("histai", "gtex",
"hest")` in `build_strict_corpus`) keeps TCGA outside by construction, while a
separate `strict+TCGA` ablation may reference it explicitly if ever needed.

`downstream` contains datasets used to measure task performance (PANDA,
CAMELYON, TCGA/CPTAC cohorts selected for evaluation, THUNDER splits, and
other labelled benchmarks). The entire benchmark dataset — not only its
labels — is kept unseen by the strict EAF pretraining corpus; see
`assert_dataset_disjoint` in `src/data/wsi/manifest.py`.

## Raw-source rules

- Raw WSI pixels have one physical copy under `sources/`.
- Manifests and symlinks may reference a source; they must not duplicate it.
- A GTEx WSI is one `SeriesInstanceUID`. The `raw_path` stored in a manifest is
  a representative `.dcm`; all DICOM instances in that series remain beside it
  under `sources/gtex/` so OpenSlide can reconstruct the WSI. Never split a
  series into a `raw_flat`-style view that separates a `.dcm` from its
  siblings.
- HISTAI uses `(subset, case_id)` as the case namespace — plain `case_id` is
  not globally unique. `slide_id` is namespaced as `<subset>__<case_id>__<stem>`.
- No cleanup command may delete TCGA, HEST, HISTAI or GTEx raw data without an
  explicit release step and a validated cold archive (see below).

## Canonical slide manifest

All pretraining/downstream datasets share the `SlideRecord` schema
(`src/data/wsi/manifest.py`). Online tile training on the existing TCGA corpus
reads:

```text
$EAF_WSI_ROOT/datasets/pretraining/tcga_eaf_multicohort_v1/manifests/slides.csv
```

Required columns are:

| Column | Meaning |
|---|---|
| `slide_id` | stable slide identifier |
| `case_id` | patient/case identifier used for case-disjoint splitting |
| `cohort` | cohort used by the balanced sampler |
| `raw_path` | WSI path, absolute or relative to `--data-root` |
| `coords_path` | TRIDENT HDF5 coordinate path |

The loader also understands `slide_group`, `include_in_pretraining`,
`coords_available`, `preprocessing_status`, and an optional `split` column.
Rows without an explicit split are assigned deterministically by `case_id`; no
runtime split file is written.

New HISTAI/GTEx/HEST manifests written by `src/data/wsi/corpora.py` use the
same `SlideRecord` fields plus `subset`, `patient_id`, `study_uid`,
`series_uid`, `downloaded`, and `metadata_json`.

## TRIDENT coordinate contract

The preprocessing profile is:

```text
magnification: 20x
patch size:    512 px
stride:        512 px
overlap:       0 px
```

Coordinates are stored as one HDF5 file per slide:

```text
artifacts/trident/20x_512px_0px_overlap/patches/<slide_id>_patches.h5
```

The online loader searches for a two-dimensional coordinate dataset named
`coords`, `coordinates`, or `patches/coords`, then falls back to the first
`[N, >=2]` HDF5 dataset. Coordinates are interpreted as level-0 OpenSlide
locations. `patch_size_level0` takes precedence because a 512 px patch at 20x
corresponds to a larger level-0 crop on a 40x-native slide. Older coordinate
files fall back to `patch_level`, `patch_size`, and magnification attributes,
then to the CLI-configurable default. This is the same coordinate contract
used before HISTAI/GTEx are segmented, so Tile/WSI-EAF cache creation reuses
it unchanged.

When the encoder expects a smaller physical field of view than the canonical
512 px@20x coordinate window, pass `--tile-size-at-target-mag`. Training uses a
random sub-crop inside the coordinate window; validation uses its centered
sub-crop. This preserves the requested magnification/FOV instead of merely
resizing a 512 px field to a 224/256 px model input.

## Two EAF training regimes

The repository intentionally supports two complementary regimes; they are not
in conflict, they cover different corpus scales:

- **Online (existing, `tcga_eaf_multicohort_v1`).** Tile-level EAF training and
  pruning-aware tile-encoder adaptation do **not** materialize per-tile source
  embeddings, per-tile teacher attention maps, per-dataset forecaster HDF5
  caches, or full duplicated fine-tuned backbones. Tiles are read from WSI
  files on demand; source tokens, teacher attention, and full-teacher
  embeddings exist only in GPU memory for the current batch. The only
  persistent training outputs are small, reproducible run artifacts:

  ```text
  checkpoints/wsi_tile_eaf_online/
    best_<run>.pt
    summary_<run>.json

  checkpoints/wsi_tile_pruned_online/
    best_<run>_adapter.pt       # trainable LoRA tensors only
    summary_<run>.json
  ```

  Metrics, sampling coverage, throughput, early-stopping state, and peak CUDA
  memory are logged to Weights & Biases (`--wandb-mode offline`/`disabled`
  when appropriate). See
  [`wsi_tile_online_training.md`](wsi_tile_online_training.md).

- **Offline (new, HISTAI/GTEx/large corpora).** A frozen teacher is executed
  once during cache creation; every subsequent EAF training run reads
  versioned HDF5 caches under `caches/` and performs no teacher forward pass
  per epoch. This is required once corpus size makes re-running a frozen
  foundation model every epoch infeasible, and it guarantees that two EAF
  variants trained from the same cache see identical teacher targets. See
  [`docs/offline_eaf_pipeline.md`](offline_eaf_pipeline.md) for the Tile-EAF /
  WSI-EAF cache contracts and adapter interfaces.

## Cold pixel archive

After segmentation, and only after required teacher caches pass validation,
tissue pixels may be archived in a model-agnostic compressed store
(`archives/`, JPEG quality 95 inside per-slide tar containers with a SHA-256
sidecar — see `src/wsi_pipeline/archive.py`). The archive is insurance against
adding a future tile encoder without redownloading the source WSI; it is never
an implicit trigger for raw deletion. See
[`docs/offline_eaf_pipeline.md`](offline_eaf_pipeline.md#cold-archive) for the
full `RAW → SEGMENTED/COORDS → TEACHER CACHES → PIXEL_ARCHIVED → VERIFIED →
RAW_RELEASABLE` lifecycle and release preconditions.

## Legacy artifacts

Existing CONCH feature stores are retained under `artifacts/tile_fm/conch_v15`
for audit and comparison. They are not inputs to the new online training path
and are not extended to newly preprocessed slides. THUNDER supplies and
initializes each supported tile encoder at training time.

See [`docs/offline_eaf_pipeline.md`](offline_eaf_pipeline.md) and
[`docs/refactor_migration.md`](refactor_migration.md) for the offline
EAF/HISTAI/GTEx refactor, and
[`wsi_tile_online_training.md`](wsi_tile_online_training.md) for the online
sampling policy, training commands, and storage guarantees.
