# HEST-1k integration for task-agnostic EAF pretraining

## Scope

The integration uses only H&E pyramidal WSI files from HEST-1k. Spatial
expression matrices, spot patches, masks, cell segmentations, transcripts and
labels are not downloaded or consumed by EAF.

Canonical flow:

```text
HEST metadata
→ human/source/benchmark firewall
→ selected H&E WSI download
→ OpenSlide audit
→ TRIDENT segmentation and 20x/512 coordinates
→ canonical HEST slides.csv
→ TCGA-clean + HEST-clean multisource manifest
→ online Tile-EAF training
```

## Leakage policy

The default inventory excludes:

- non-human samples;
- every sample ID present in the official `MahmoodLab/hest-bench` dataset;
- lung samples, preserving lung/NSCLC as an OOD domain;
- source metadata matching `catalog/hest_overlap/blocked_source_terms.txt`;
- IDs listed in `catalog/hest_overlap/blocked_hest_ids.txt`.

Source-term matching is conservative and must be manually reviewed in
`manifests/excluded.csv`. The resulting corpus should be described as clean
with respect to the exact HEST release, benchmark snapshot and blocklists
stored with the experiment.

## License

HEST-1k is distributed under CC BY-NC-SA 4.0. Keep the HEST dataset and
artifacts outside Git, retain attribution, and do not redistribute the WSI
files through the EAF repository.

## Installation and access

Accept the gated dataset conditions on Hugging Face, then authenticate:

```bash
python -m pip install -U huggingface_hub datasets pandas openslide-python pillow numpy
huggingface-cli login
```

## Build inventory

```bash
python scripts/wsi_data/build_hest_eaf_inventory.py
```

Inspect:

```text
$EAF_WSI_ROOT/datasets/pretraining/hest_eaf_thunder_clean_v1/manifests/
├── inventory_all.csv
├── eligible.csv
├── excluded.csv
├── quarantine.csv
├── download_ids.txt
└── hest_benchmark_ids.txt
```

Do not download until the exclusions and source-cohort distribution have been
reviewed.

## Pilot download

```bash
python scripts/wsi_data/download_hest_eaf_wsis.py \
  --batch-size 24 \
  --start-batch 0 \
  --end-batch 1
```

Audit the pilot and write thumbnails:

```bash
python scripts/wsi_data/audit_materialize_hest_eaf.py \
  --write-thumbnails
```

Review `artifacts/qc/thumbnails` and `manifests/wsi_audit.csv` before the full
download.

## Full WSI-only download

```bash
nohup python -u scripts/wsi_data/download_hest_eaf_wsis.py \
  --batch-size 64 \
  --max-workers 8 \
  > /data2/home/vcivale/projects/imaging/data/WSI/logs/hest_eaf_download/nohup_master.log 2>&1 &
```

Re-running the command is resumable: already materialized WSI files are
skipped.

## Materialize and preprocess

```bash
python scripts/wsi_data/audit_materialize_hest_eaf.py --write-thumbnails
```

Run TRIDENT using:

```text
datasets/pretraining/hest_eaf_thunder_clean_v1/views/raw_flat
manifests/trident_pending.csv
```

with the same canonical EAF settings:

```text
20x magnification
512 px tile size
0 px overlap
```

After TRIDENT finishes, rerun `audit_materialize_hest_eaf.py`; it refreshes
`coords_available` in the canonical manifest.

## Build multisource EAF manifest

```bash
python scripts/wsi_data/build_eaf_multisource_manifest.py \
  --require-coords \
  --hest-share 0.30 \
  --selection-seed 17
```

The output is:

```text
datasets/pretraining/eaf_multisource_clean_v1/manifests/slides.csv
```

The builder retains complete HEST donor/patient groups and targets a 30% HEST slide share by default. Use the resulting manifest with the existing online Tile-EAF trainer. The split is
computed by `source_family + sampling_group`, keeping all serial sections from
the same patient/donor group in one split when HEST metadata exposes that
identifier.

## Recommended first experiment

Keep the teacher, source layer, optimization and total tile budget fixed:

1. TCGA-clean only;
2. HEST-clean only;
3. TCGA-clean + HEST-clean.

Report THUNDER full and per-dataset metrics, patch top-recall, full-vs-pruned
embedding agreement and OOD stability. Do not claim a benefit from spatial
transcriptomics because the EAF pretraining path uses only H&E morphology.
