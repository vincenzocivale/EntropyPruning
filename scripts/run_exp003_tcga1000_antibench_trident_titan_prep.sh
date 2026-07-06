#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data2/home/vcivale/miniconda3/envs/eaf-wsi/bin/python}"
TRIDENT_REPO="${TRIDENT_REPO:-/data2/home/vcivale/repos/TRIDENT}"
# DATA_ROOT holds reusable, cohort-level data (raw WSI, trident features, EAF
# targets) shared across every experiment that trains on this cohort.
# EXPERIMENT_ROOT holds only this run's outputs (checkpoints, rankings, logs).
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/unlabeled/tcga/tcga1000_antibench}"
LABELED_ROOT="${LABELED_ROOT:-${REPO_ROOT}/data/labeled/wsi_level/tcga1000_antibench_project_label}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/experiments/exp003_tcga1000_antibench}"
WSI_DIR="${WSI_DIR:-${DATA_ROOT}/raw_wsi}"
MAG="${MAG:-20}"
PATCH_SIZE="${PATCH_SIZE:-512}"
OVERLAP="${OVERLAP:-0}"
PATCH_ENCODER="${PATCH_ENCODER:-conch_v15}"
SEGMENTER="${SEGMENTER:-otsu}"
GPU_ID="${GPU_ID:-0}"
MAX_WORKERS="${MAX_WORKERS:-4}"
LAYER_INDICES="${LAYER_INDICES:-2}"
DEVICE="${DEVICE:-cuda:0}"
IMG_SIZE="${IMG_SIZE:-448}"
BATCH_LIMIT="${BATCH_LIMIT:-256}"
LAYER_MAX_WORKERS="${LAYER_MAX_WORKERS:-2}"
WSI_CACHE="${WSI_CACHE:-/tmp/trident-wsi-cache-exp003}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
COORDS_SPEC="${MAG}x_${PATCH_SIZE}px_${OVERLAP}px_overlap"
JOB_SPEC="${COORDS_SPEC}_${PATCH_ENCODER}"
TRIDENT_JOB_DIR="${TRIDENT_JOB_DIR:-${DATA_ROOT}/trident/${JOB_SPEC}}"
LOG_DIR="${LOG_DIR:-${EXPERIMENT_ROOT}/reports/nohup_logs}"

RAW_MANIFEST="${RAW_MANIFEST:-${DATA_ROOT}/manifests/slides_tcga1000_antibench.csv}"
LABELS_MANIFEST="${LABELS_MANIFEST:-${LABELED_ROOT}/manifests/slides_tcga1000_antibench_project_label.csv}"
CUSTOM_LIST="${CUSTOM_LIST:-${DATA_ROOT}/manifests/trident_custom_wsi_list_downloaded.csv}"
CONCH_MANIFEST="${CONCH_MANIFEST:-${DATA_ROOT}/manifests/trident_features_conch_v15.csv}"
EARLY_LAYER_MANIFEST="${EARLY_LAYER_MANIFEST:-${DATA_ROOT}/manifests/trident_features_conch_v15_layer2.csv}"
TITAN_TARGET_DIR="${TITAN_TARGET_DIR:-${DATA_ROOT}/eaf_targets/wsi_fm_attention/titan}"
TITAN_TARGET_MANIFEST="${TITAN_TARGET_MANIFEST:-${DATA_ROOT}/manifests/titan_attention_targets_manifest.csv}"
TITAN_IMPORTANCE_MANIFEST="${TITAN_IMPORTANCE_MANIFEST:-${DATA_ROOT}/manifests/titan_attention_importance_manifest.csv}"
# Set TITAN_OVERWRITE=1 to force TITAN attention recompute for every slide;
# by default already-extracted slides are skipped (see extract_titan_tile_attention.py).
TITAN_OVERWRITE="${TITAN_OVERWRITE:-0}"

mkdir -p "${LOG_DIR}" "${DATA_ROOT}/manifests" "${DATA_ROOT}/eaf_targets" \
  "${EXPERIMENT_ROOT}/eaf/checkpoints" "${EXPERIMENT_ROOT}/eaf/rankings"

echo "[exp003-prep] repo_root=${REPO_ROOT}"
echo "[exp003-prep] data_root=${DATA_ROOT}"
echo "[exp003-prep] experiment_root=${EXPERIMENT_ROOT}"
echo "[exp003-prep] wsi_dir=${WSI_DIR}"
echo "[exp003-prep] trident_job_dir=${TRIDENT_JOB_DIR}"
echo "[exp003-prep] custom_list=${CUSTOM_LIST}"
echo "[exp003-prep] device=${DEVICE}"
echo "[exp003-prep] gpu_id=${GPU_ID}"
echo "[exp003-prep] wsi_cache=${WSI_CACHE}"
echo "[exp003-prep] cache_batch_size=${CACHE_BATCH_SIZE}"

"${PYTHON_BIN}" scripts/build_downloaded_wsi_list.py \
  --wsi-dir "${WSI_DIR}" \
  --manifest "${RAW_MANIFEST}" \
  --output-csv "${CUSTOM_LIST}" \
  --overwrite

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-eaf-wsi-exp003}"
TORCH_HOME="${TORCH_HOME:-/tmp/torch-eaf-wsi-exp003}"
HF_HOME="${HF_HOME:-/data2/home/vcivale/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}" "${HF_HUB_CACHE}" "${TRIDENT_JOB_DIR}"
mkdir -p "${WSI_CACHE}"
export MPLCONFIGDIR TORCH_HOME HF_HOME HF_HUB_CACHE

(
  cd "${TRIDENT_REPO}"
  "${PYTHON_BIN}" run_batch_of_slides.py \
    --task all \
    --wsi_dir "${WSI_DIR}" \
    --job_dir "${TRIDENT_JOB_DIR}" \
    --patch_encoder "${PATCH_ENCODER}" \
    --mag "${MAG}" \
    --patch_size "${PATCH_SIZE}" \
    --overlap "${OVERLAP}" \
    --gpus "${GPU_ID}" \
    --max_workers "${MAX_WORKERS}" \
    --segmenter "${SEGMENTER}" \
    --clear_dead_locks \
    --wsi_cache "${WSI_CACHE}" \
    --cache_batch_size "${CACHE_BATCH_SIZE}" \
    --custom_list_of_wsis "${CUSTOM_LIST}"
)

"${PYTHON_BIN}" scripts/build_trident_manifest.py \
  --features-dir "${TRIDENT_JOB_DIR}/${COORDS_SPEC}/features_${PATCH_ENCODER}" \
  --coords-dir "${TRIDENT_JOB_DIR}/${COORDS_SPEC}/patches" \
  --labels-csv "${LABELS_MANIFEST}" \
  --output-manifest "${CONCH_MANIFEST}" \
  --require-coords \
  --require-labels \
  --overwrite

"${PYTHON_BIN}" scripts/extract_conch_v15_layer_features.py \
  --job-dir "${TRIDENT_JOB_DIR}" \
  --wsi-dir "${WSI_DIR}" \
  --coords-dir "${COORDS_SPEC}" \
  --custom-list-of-wsis "${CUSTOM_LIST}" \
  --layer-indices ${LAYER_INDICES} \
  --img-size "${IMG_SIZE}" \
  --device "${DEVICE}" \
  --batch-limit "${BATCH_LIMIT}" \
  --max-workers "${LAYER_MAX_WORKERS}"

"${PYTHON_BIN}" scripts/build_trident_manifest.py \
  --features-dir "${TRIDENT_JOB_DIR}/${COORDS_SPEC}/features_conch_v15_multilayer${LAYER_INDICES}" \
  --coords-dir "${TRIDENT_JOB_DIR}/${COORDS_SPEC}/patches" \
  --labels-csv "${LABELS_MANIFEST}" \
  --output-manifest "${EARLY_LAYER_MANIFEST}" \
  --require-coords \
  --require-labels \
  --overwrite

titan_overwrite_args=()
if [ "${TITAN_OVERWRITE}" = "1" ]; then
  titan_overwrite_args+=(--overwrite)
fi

"${PYTHON_BIN}" scripts/extract_titan_tile_attention.py \
  --input-feature-manifest "${CONCH_MANIFEST}" \
  --output-target-dir "${TITAN_TARGET_DIR}" \
  --output-manifest "${TITAN_TARGET_MANIFEST}" \
  --device "${DEVICE}" \
  "${titan_overwrite_args[@]}"

"${PYTHON_BIN}" scripts/build_wsi_importance_manifest.py \
  --targets-dir "${TITAN_TARGET_DIR}" \
  --labels-csv "${LABELS_MANIFEST}" \
  --output-manifest "${TITAN_IMPORTANCE_MANIFEST}" \
  --target-glob "*.npz" \
  --target-source titan_wsi_fm \
  --require-labels \
  --overwrite

echo "[exp003-prep] done"
echo "[exp003-prep] conch_manifest=${CONCH_MANIFEST}"
echo "[exp003-prep] early_layer_manifest=${EARLY_LAYER_MANIFEST}"
echo "[exp003-prep] titan_target_manifest=${TITAN_TARGET_MANIFEST}"
echo "[exp003-prep] titan_importance_manifest=${TITAN_IMPORTANCE_MANIFEST}"
