# EAF-WSI strict pretraining dataset update

This update is intentionally **additive**. Existing TCGA raw slides, TRIDENT
coordinates/features, manifests, and other derived products are never moved or
deleted. TCGA is registered as a preserved source and excluded from the
`eaf_wsi_pretrain_strict_v1` dataset.

Everything below lives under the single canonical store from
[`data_layout.md`](data_layout.md) — `$EAF_WSI_ROOT`. There is no second,
parallel data root: new datasets sit next to `tcga_eaf_multicohort_v1` under
`datasets/pretraining/`, and new raw slides get their one physical copy under
`sources/`, exactly like every other source in that layout.

## Proposed V1

- HISTAI: 8,000 H&E WSIs selected with case-diverse round-robin sampling.
- Existing HEST: inventory the current local H&E WSI tree without copying it.
- GTEx: validate DICOM support on 1-5 slides first; scale only after the smoke
  test succeeds.
- TCGA: preserved exactly where it is, but absent from strict-v1 manifests.

The source data remain physically separated from the derived dataset:

```text
$EAF_WSI_ROOT/
  sources/
    gdc/tcga/...                        # existing; untouched
    histai/                             # new: physical HISTAI copy, one per file
      <subset>/<case>/<slide>.tiff
  datasets/
    pretraining/
      tcga_eaf_multicohort_v1/          # existing; untouched
      hest.../                          # existing; untouched
      histai_eaf_wsi_v1/                # new dataset
        dataset.yaml
        manifests/
          plan.csv                        # frozen HISTAI acquisition plan
          slides.csv                       # downloaded HISTAI slides
        views/
          raw_flat/*.tiff -> ../../../../sources/histai/...
      gtex_eaf_wsi_v1/                  # later, after smoke test
      eaf_wsi_pretrain_strict_v1/       # combined pretraining view
        dataset.yaml
        registry.json
        manifests/
          hest_existing.csv
          slides.csv                      # union of histai + hest, with a `split` column
```

Splits are a `split` column on `manifests/slides.csv`, the same convention the
online tile-training loader already understands (see "Canonical slide
manifest" in `data_layout.md`) — there is no separate `splits/` directory.

## Dependencies

```bash
pip install -U huggingface_hub hf_xet idc-index
hf auth login
```

HISTAI is gated. Accept the access terms for `histai/HISTAI-metadata` and the
specialized HISTAI repositories before launching the WSI download.

## 1. Initialize without touching TCGA

```bash
python scripts/wsi_prepare_strict_pretraining.py init \
  --data-root "$EAF_WSI_ROOT" \
  --hest-root "$HEST_EXISTING_ROOT"
```

`--data-root` defaults to `$EAF_WSI_ROOT` if that env var is set. `--tcga-root`
defaults to `$EAF_WSI_ROOT/sources/gdc/tcga` (the canonical location) and only
needs to be passed if TCGA's raw slides live somewhere else.

`registry.json` records TCGA as `preserved_not_in_strict_v1`. No `rm`, `mv`, or
copy operation is performed on the existing datasets.

## 2. Freeze the HISTAI acquisition plan

```bash
python scripts/wsi_prepare_strict_pretraining.py plan-histai \
  --data-root "$EAF_WSI_ROOT"
```

The default plan contains exactly 8,000 H&E WSIs:

```text
HISTAI-mixed             3200
HISTAI-skin-b2           1400
HISTAI-skin-b1            700
HISTAI-colorectal-b1      800
HISTAI-breast             750
HISTAI-thorax             656
HISTAI-gastrointestinal   200
HISTAI-hematologic        200
HISTAI-colorectal-b2       94
```

Selection is case-diverse: the script takes at most one slide per case per
round before taking a second slide from any case.

## 3. Download HISTAI resumably

```bash
export HF_XET_HIGH_PERFORMANCE=1
nohup python scripts/wsi_prepare_strict_pretraining.py download-histai \
  --data-root "$EAF_WSI_ROOT" \
  --workers 8 \
  > logs/histai_strict_v1_download.log 2>&1 &
```

Hugging Face `snapshot_download(..., local_dir=...)` preserves its metadata and
will avoid re-downloading unchanged files on subsequent runs. Each downloaded
file gets exactly one physical copy under `sources/histai/`; a
`views/raw_flat/` symlink is created per slide under the `histai_eaf_wsi_v1`
dataset, matching the `tcga_eaf_multicohort_v1/views/raw_flat` convention.

## 4. Register current HEST without copying it

```bash
python scripts/wsi_prepare_strict_pretraining.py scan-hest \
  --data-root "$EAF_WSI_ROOT" \
  --hest-root "$HEST_EXISTING_ROOT"
```

The scan prefers files under `wsis/` or `raw_wsi/` and excludes thumbnails,
spatial plots, tissue masks, patches, TRIDENT outputs, and features.

Before the final paper split, replace slide-level grouping with true HEST
patient/study grouping when available in the HEST metadata. This initial view
is intended for acquisition/preprocessing, not the final leakage audit.

## 5. Build strict acquisition manifest and internal splits

```bash
python scripts/wsi_prepare_strict_pretraining.py build \
  --data-root "$EAF_WSI_ROOT"
```

The builder refuses any path resolving below the registered TCGA root. It
writes a single `manifests/slides.csv` with a `split` column carrying 90/5/5
group-level train/validation/holdout partitions for unsupervised EAF
development.

## 6. GTEx: smoke test before bulk acquisition

GTEx in IDC contains 25,503 H&E slide microscopy images and is primarily
provided as DICOM. Test the exact TRIDENT/OpenSlide environment first:

```bash
python scripts/wsi_gtex_smoke.py \
  --output-dir "$EAF_WSI_ROOT/datasets/pretraining/gtex_eaf_wsi_v1/smoke" \
  --n 5 --download
```

Then run TRIDENT on those DICOM files. Current TRIDENT versions can use DICOM
through OpenSlide when the installed OpenSlide build includes DICOM support.
Do not start the 1,500-slide GTEx tranche until this smoke test passes.

## Preservation rule

Do **not** rename the current TCGA directory merely to reflect its changed role.
Existing symlinks, manifests, preprocessing outputs, and scripts may reference
that location. Role changes belong in manifests/registry metadata, not in a
filesystem migration.


## 7. Preprocess the new HISTAI tranche without touching TCGA

Use a new TRIDENT job directory dedicated to HISTAI. Do not point `--job-dir`
at the existing TCGA job directory. For the current repository wrapper:

```bash
python scripts/wsi_preprocess.py \
  --trident-repo "$TRIDENT_REPO" \
  --wsi-dir "$EAF_WSI_ROOT/sources/histai" \
  --job-dir "$EAF_WSI_ROOT/datasets/pretraining/histai_eaf_wsi_v1/artifacts/trident" \
  --stages seg coords \
  --gpus 0 \
  --segmenter hest \
  --mag 20 \
  --patch-size 512
```

The wrapper searches nested WSI folders by default, so the HISTAI
`subset/case/slide.tiff` hierarchy can remain intact. `--job-dir` mirrors the
`artifacts/trident/...` sub-layout `tcga_eaf_multicohort_v1` already uses, so
downstream tooling that expects that shape keeps working unmodified.
