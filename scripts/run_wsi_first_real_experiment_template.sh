#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

EARLY_FEATURE_STORE="${EARLY_FEATURE_STORE:-}"
LATE_FEATURE_STORE="${LATE_FEATURE_STORE:-}"
TARGET_FEATURE_STORE="${TARGET_FEATURE_STORE:-}"
SPLIT_DIR="${SPLIT_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

INPUT_FEATURE_DIM="${INPUT_FEATURE_DIM:-}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-2}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cpu}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-coords}"
KEEP_RATIOS="${KEEP_RATIOS:-0.05 0.10 0.25 0.50 1.0}"
PRUNED_KEEP_RATIO="${PRUNED_KEEP_RATIO:-0.10}"
LOSS="${LOSS:-kl}"
TOP_K="${TOP_K:-10}"
DROPOUT="${DROPOUT:-0.1}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[exp001-template] missing required env var: ${name}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[exp001-template] missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[exp001-template] missing required directory: ${path}" >&2
    exit 1
  fi
}

require_var EARLY_FEATURE_STORE
require_var LATE_FEATURE_STORE
require_var TARGET_FEATURE_STORE
require_var SPLIT_DIR
require_var OUTPUT_ROOT
require_var INPUT_FEATURE_DIM

require_file "${EARLY_FEATURE_STORE}"
require_file "${LATE_FEATURE_STORE}"
require_file "${TARGET_FEATURE_STORE}"
require_dir "${SPLIT_DIR}"
require_file "${SPLIT_DIR}/test.txt"

CHECKPOINT_DIR="${OUTPUT_ROOT}/checkpoints/exp001_importance_forecaster"
RESULTS_DIR="${OUTPUT_ROOT}/results"
REPORTS_DIR="${OUTPUT_ROOT}/reports"
PRUNED_STORE="${OUTPUT_ROOT}/features_late_pruned_keep_${PRUNED_KEEP_RATIO}.h5"
PRUNING_CSV="${RESULTS_DIR}/exp001_importance_pruning.csv"

mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${REPORTS_DIR}"

echo "[exp001-template] repo_root=${REPO_ROOT}"
echo "[exp001-template] output_root=${OUTPUT_ROOT}"

echo "[exp001-template] validate paired stores"
echo "${PYTHON_BIN} scripts/validate_wsi_paired_feature_stores.py --input-feature-store ${EARLY_FEATURE_STORE} --target-feature-store ${TARGET_FEATURE_STORE} --input-feature-dim ${INPUT_FEATURE_DIM} --alignment-mode ${ALIGNMENT_MODE} --require-coords --require-attention"
"${PYTHON_BIN}" scripts/validate_wsi_paired_feature_stores.py \
  --input-feature-store "${EARLY_FEATURE_STORE}" \
  --target-feature-store "${TARGET_FEATURE_STORE}" \
  --input-feature-dim "${INPUT_FEATURE_DIM}" \
  --alignment-mode "${ALIGNMENT_MODE}" \
  --require-coords \
  --require-attention

echo "[exp001-template] train tile-importance forecaster"
echo "${PYTHON_BIN} scripts/train_wsi_importance_forecaster.py --input-feature-store ${EARLY_FEATURE_STORE} --target-feature-store ${TARGET_FEATURE_STORE} --output-dir ${CHECKPOINT_DIR} --input-feature-dim ${INPUT_FEATURE_DIM} --hidden-dim ${HIDDEN_DIM} --n-heads ${N_HEADS} --n-layers ${N_LAYERS} --dropout ${DROPOUT} --loss ${LOSS} --top-k ${TOP_K} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --split-dir ${SPLIT_DIR} --alignment-mode ${ALIGNMENT_MODE} --device ${DEVICE}"
"${PYTHON_BIN}" scripts/train_wsi_importance_forecaster.py \
  --input-feature-store "${EARLY_FEATURE_STORE}" \
  --target-feature-store "${TARGET_FEATURE_STORE}" \
  --output-dir "${CHECKPOINT_DIR}" \
  --input-feature-dim "${INPUT_FEATURE_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --n-heads "${N_HEADS}" \
  --n-layers "${N_LAYERS}" \
  --dropout "${DROPOUT}" \
  --loss "${LOSS}" \
  --top-k "${TOP_K}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --split-dir "${SPLIT_DIR}" \
  --alignment-mode "${ALIGNMENT_MODE}" \
  --device "${DEVICE}"

BEST_CKPT="${CHECKPOINT_DIR}/best_wsi_tile_importance_forecaster.pt"
require_file "${BEST_CKPT}"

echo "[exp001-template] evaluate pruning"
echo "${PYTHON_BIN} scripts/evaluate_wsi_importance_pruning.py --input-feature-store ${EARLY_FEATURE_STORE} --target-feature-store ${TARGET_FEATURE_STORE} --forecaster-checkpoint ${BEST_CKPT} --slide-ids-file ${SPLIT_DIR}/test.txt --keep-ratios ${KEEP_RATIOS} --alignment-mode ${ALIGNMENT_MODE} --output-csv ${PRUNING_CSV} --device ${DEVICE}"
"${PYTHON_BIN}" scripts/evaluate_wsi_importance_pruning.py \
  --input-feature-store "${EARLY_FEATURE_STORE}" \
  --target-feature-store "${TARGET_FEATURE_STORE}" \
  --forecaster-checkpoint "${BEST_CKPT}" \
  --slide-ids-file "${SPLIT_DIR}/test.txt" \
  --keep-ratios ${KEEP_RATIOS} \
  --alignment-mode "${ALIGNMENT_MODE}" \
  --output-csv "${PRUNING_CSV}" \
  --device "${DEVICE}"

echo "[exp001-template] materialize pruned late store"
echo "${PYTHON_BIN} scripts/create_pruned_wsi_feature_store.py --selection-feature-store ${EARLY_FEATURE_STORE} --materialize-feature-store ${LATE_FEATURE_STORE} --output-feature-store ${PRUNED_STORE} --forecaster-checkpoint ${BEST_CKPT} --keep-ratio ${PRUNED_KEEP_RATIO} --alignment-mode ${ALIGNMENT_MODE} --device ${DEVICE}"
"${PYTHON_BIN}" scripts/create_pruned_wsi_feature_store.py \
  --selection-feature-store "${EARLY_FEATURE_STORE}" \
  --materialize-feature-store "${LATE_FEATURE_STORE}" \
  --output-feature-store "${PRUNED_STORE}" \
  --forecaster-checkpoint "${BEST_CKPT}" \
  --keep-ratio "${PRUNED_KEEP_RATIO}" \
  --alignment-mode "${ALIGNMENT_MODE}" \
  --device "${DEVICE}"

cat <<EOF
[exp001-template] done
checkpoint: ${BEST_CKPT}
pruning_csv: ${PRUNING_CSV}
pruned_store: ${PRUNED_STORE}
EOF
