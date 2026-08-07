# Unlabeled WSI Acquisition for Hierarchical EAF

This workflow creates reproducible, size-bounded cohorts for two label-free objectives:

1. **Tile-EAF:** sample tissue tiles from many WSIs and distill frozen tile-encoder attention.
2. **WSI-EAF:** retain every tile/coordinate belonging to a slide and distill the frozen WSI model.

The acquisition step does not require diagnostic labels. It still preserves source, cohort, case, and slide identifiers so that pretraining and downstream/OOD cohorts can be made case- and source-disjoint.

## Why a plan is generated before downloading

Raw WSI collections are measured in TiB. `scripts/manage_unlabeled_wsi.py` therefore separates:

- live discovery and exact size estimation;
- deterministic budgeted selection;
- manifest review;
- explicit download.

Every discovery produces:

```text
data/unlabeled/<cohort>/manifests/
  acquisition_plan.json
  slides.csv
  gdc_manifest.tsv       # GDC only
  idc_series.csv         # IDC only
```

The JSON plan is the immutable acquisition record. Commit only scripts/configuration; keep generated manifests local when they contain a very large inventory.

## Installation

```bash
pip install -r requirements-wsi-pipeline.txt
pip install -r requirements-wsi-acquisition.txt
```

For GDC downloads, install the official `gdc-client`. Discovery itself only needs `requests`. IDC discovery/download uses the official `idc-index` package and public cloud buckets.

## 1. Inspect the source catalog

```bash
python scripts/manage_unlabeled_wsi.py catalog
```

TCGA/GDC is the recommended first source because it is multi-organ, open access, provides exact byte sizes/checksums, and supports resumable downloads. CPTAC through IDC is the preferred external source. GTEx is listed for scientific completeness but is intentionally not scraped: its public histology service exposes metadata/viewing rather than a supported bulk-image API.

## 2. Create a balanced TCGA pilot

The following command queries every TCGA project, keeps diagnostic slides, samples at most 40 slides per project, and refuses to place more than 1,000 slides or 2,500 GiB in the plan:

```bash
python scripts/manage_unlabeled_wsi.py discover-gdc \
  --all-tcga \
  --per-cohort 40 \
  --max-items 1000 \
  --max-gib 2500 \
  --seed 17 \
  --output-dir data/unlabeled/tcga_eaf_pilot/manifests
```

To use explicit projects:

```bash
python scripts/manage_unlabeled_wsi.py discover-gdc \
  --projects TCGA-BRCA TCGA-COAD TCGA-LUAD TCGA-PRAD \
  --max-items 500 \
  --max-gib 1200 \
  --seed 17 \
  --output-dir data/unlabeled/tcga_selected/manifests
```

`--include-tissue-slides` adds non-diagnostic slide images. Keep the diagnostic-only default for the first study; frozen/tissue slides can introduce a distinct preparation domain.

## 3. Find and select IDC/CPTAC collections

Obtain a live inventory of every IDC collection containing DICOM Slide Microscopy series:

```bash
python scripts/manage_unlabeled_wsi.py list-idc
```

Then build a bounded CPTAC plan:

```bash
python scripts/manage_unlabeled_wsi.py discover-idc \
  --collections cptac_luad cptac_lscc cptac_ccrcc cptac_ucec \
  --per-cohort 250 \
  --max-items 1000 \
  --max-gib 2500 \
  --seed 17 \
  --output-dir data/unlabeled/cptac_eaf/manifests
```

Collection identifiers must come from `list-idc`, because IDC releases and identifiers can change.

## 4. Review and download

```bash
python scripts/manage_unlabeled_wsi.py estimate \
  --plan data/unlabeled/tcga_eaf_pilot/manifests/acquisition_plan.json
```

Download starts only with explicit confirmation:

```bash
python scripts/manage_unlabeled_wsi.py download \
  --plan data/unlabeled/tcga_eaf_pilot/manifests/acquisition_plan.json \
  --output-dir data/unlabeled/tcga_eaf_pilot/raw_wsi \
  --require-free-gib 3000 \
  --processes 8 \
  --yes
```

For GDC, the command invokes `gdc-client`, which supports parallel and resumed transfers. If it is unavailable, the generated `gdc_manifest.tsv` remains compatible with the repository's existing `scripts/download_gdc_manifest_simple.py`.

IDC plans are downloaded by exact DICOM `SeriesInstanceUID` through `idc-index`.

## 5. Preprocess with the existing TRIDENT wrapper

```bash
python scripts/wsi_preprocess.py \
  --trident-repo /path/to/TRIDENT \
  --wsi-dir data/unlabeled/tcga_eaf_pilot/raw_wsi \
  --job-dir data/unlabeled/tcga_eaf_pilot/trident \
  --stages seg coords \
  --gpus 0 1 \
  --mag 20 \
  --patch-size 512
```

DICOM WSI support depends on the reader used by the preprocessing stack. When the selected TRIDENT/OpenSlide environment cannot read an IDC series directly, convert the DICOM WSI to a supported TIFF/OME-TIFF representation or use a DICOM-aware reader while preserving the original `SeriesInstanceUID` in `slides.csv`.

## Recommended scale

Do not start from the full public corpus. A useful progression is:

| Stage | Slides | Indicative raw storage | Purpose |
|---|---:|---:|---|
| Smoke | 20-50 | 30-150 GiB | acquisition, reader, and coordinate checks |
| Pilot | 500-1,000 | roughly 1-3 TiB | Tile-EAF scaling and first WSI-EAF signal audit |
| Main | 3,000-8,000 | roughly 5-15 TiB | multi-organ task-agnostic training |
| Full public | 10,000+ | often 15-30+ TiB | only after scaling laws justify it |

These storage ranges are planning estimates, not download manifests. Scanner, magnification, tissue area, compression, and inclusion of tissue/frozen slides produce large variation. The live plan is the source of truth.

## Leakage and split policy

Before training:

1. build a global table keyed by provider, collection/project, case ID, and slide ID;
2. remove all downstream validation/test cases from unlabeled pretraining;
3. reserve entire organs or cohorts for OOD evaluation, not random slides;
4. keep all slides from the same patient in one split;
5. report overlap with tile benchmarks such as THUNDER separately from overlap with WSI benchmarks;
6. record the tile and WSI foundation models whose original pretraining data may already include TCGA/CPTAC.

The last point cannot be solved by data splitting, but it must be disclosed when interpreting OOD results.
