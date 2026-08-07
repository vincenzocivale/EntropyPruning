#!/usr/bin/env bash
set -euo pipefail

ATTENTION_MANIFEST="${ATTENTION_MANIFEST:-artifacts/wsi_attention/titan_real_maps/manifest.csv}"
ARTIFACT_REGISTRY="${ARTIFACT_REGISTRY:-data/wsi/tcga1000_antibench/registry/artifacts.csv}"
METADATA="${METADATA:-data/wsi/tcga1000_antibench/labels/tcga_project.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/wsi_attention_signal_discovery/titan_final_heads}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/wsi_run_attention_signal_discovery.py \
  --attention-manifest "${ATTENTION_MANIFEST}" \
  --attention-key global_to_tiles_mass_share \
  --target-layer -1 \
  --metadata "${METADATA}" \
  --feature-view conch_final "${ARTIFACT_REGISTRY}" conch_v15_final_d768 final \
  --feature-view conch_layer2 "${ARTIFACT_REGISTRY}" conch_v15_layer2_d1024 final \
  --output-dir "${OUTPUT_DIR}" \
  --head-groups 6 \
  --positive-fraction 0.10 \
  --negative-fraction 0.50 \
  --retention 0.30 0.40 0.50 0.60 \
  --max-train-tiles 100000 \
  --max-tiles-per-slide 2048 \
  --projection-dim 256 \
  --knn-k 32 \
  --knn-backend auto \
  --prototype-count 16 \
  --slide-prototype-count 16 \
  --bootstrap-replicates 500 \
  --seed 17 \
  "$@"
