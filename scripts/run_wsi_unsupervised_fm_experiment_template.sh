#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

EARLY_FEATURES_DIR="${EARLY_FEATURES_DIR:-}"
LATE_FEATURES_DIR="${LATE_FEATURES_DIR:-}"
TARGETS_DIR="${TARGETS_DIR:-}"
COORDS_DIR="${COORDS_DIR:-}"
TARGET_COORDS_DIR="${TARGET_COORDS_DIR:-${COORDS_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

INPUT_FEATURE_DIM="${INPUT_FEATURE_DIM:-}"
LATE_FEATURE_DIM="${LATE_FEATURE_DIM:-}"

EARLY_FEATURE_GLOB="${EARLY_FEATURE_GLOB:-*}"
LATE_FEATURE_GLOB="${LATE_FEATURE_GLOB:-*}"
TARGET_GLOB="${TARGET_GLOB:-*}"

EARLY_FEATURE_KEY="${EARLY_FEATURE_KEY:-}"
LATE_FEATURE_KEY="${LATE_FEATURE_KEY:-}"
EARLY_COORDS_KEY="${EARLY_COORDS_KEY:-}"
LATE_COORDS_KEY="${LATE_COORDS_KEY:-}"
TARGET_KEY="${TARGET_KEY:-}"
TARGET_COORDS_KEY="${TARGET_COORDS_KEY:-}"

TARGET_SOURCE="${TARGET_SOURCE:-wsi_fm}"
TARGET_TYPE="${TARGET_TYPE:-tile_importance}"
TARGET_NORMALIZE="${TARGET_NORMALIZE:-none}"
CREATED_BY="${CREATED_BY:-run_wsi_unsupervised_fm_experiment_template}"

TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"
TEST_RATIO="${TEST_RATIO:-0.15}"
SPLIT_SEED="${SPLIT_SEED:-0}"

HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
LOSS="${LOSS:-kl}"
TOP_K="${TOP_K:-10}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cpu}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-coords}"
KEEP_RATIOS="${KEEP_RATIOS:-0.05 0.10 0.25 0.50 1.0}"
PRUNED_KEEP_RATIO="${PRUNED_KEEP_RATIO:-0.10}"
OVERWRITE="${OVERWRITE:-0}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[wsi-unsupervised-template] missing required env var: ${name}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[wsi-unsupervised-template] missing required directory: ${path}" >&2
    exit 1
  fi
}

append_if_set() {
  local var_name="$1"
  local flag_name="$2"
  local -n array_ref="$3"
  local value="${!var_name:-}"
  if [[ -n "${value}" ]]; then
    array_ref+=("${flag_name}" "${value}")
  fi
}

require_var EARLY_FEATURES_DIR
require_var LATE_FEATURES_DIR
require_var TARGETS_DIR
require_var COORDS_DIR
require_var OUTPUT_ROOT
require_var INPUT_FEATURE_DIM
require_var LATE_FEATURE_DIM

require_dir "${EARLY_FEATURES_DIR}"
require_dir "${LATE_FEATURES_DIR}"
require_dir "${TARGETS_DIR}"
require_dir "${COORDS_DIR}"
require_dir "${TARGET_COORDS_DIR}"

MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
STORE_DIR="${OUTPUT_ROOT}/stores"
SPLIT_DIR="${OUTPUT_ROOT}/splits"

EARLY_MANIFEST="${MANIFEST_DIR}/early_features.csv"
LATE_MANIFEST="${MANIFEST_DIR}/late_features.csv"
TARGET_MANIFEST="${MANIFEST_DIR}/tile_importance_targets.csv"

EARLY_STORE="${STORE_DIR}/features_layer2.h5"
LATE_STORE="${STORE_DIR}/features_late.h5"
TARGET_STORE="${STORE_DIR}/features_wsi_importance.h5"

mkdir -p "${MANIFEST_DIR}" "${STORE_DIR}" "${SPLIT_DIR}"

BUILD_EARLY_ARGS=(
  --features-dir "${EARLY_FEATURES_DIR}"
  --coords-dir "${COORDS_DIR}"
  --feature-glob "${EARLY_FEATURE_GLOB}"
  --output-manifest "${EARLY_MANIFEST}"
  --require-coords
)
BUILD_LATE_ARGS=(
  --features-dir "${LATE_FEATURES_DIR}"
  --coords-dir "${COORDS_DIR}"
  --feature-glob "${LATE_FEATURE_GLOB}"
  --output-manifest "${LATE_MANIFEST}"
  --require-coords
)
BUILD_TARGET_ARGS=(
  --targets-dir "${TARGETS_DIR}"
  --coords-dir "${TARGET_COORDS_DIR}"
  --target-glob "${TARGET_GLOB}"
  --output-manifest "${TARGET_MANIFEST}"
  --target-source "${TARGET_SOURCE}"
  --target-type "${TARGET_TYPE}"
  --require-coords
)
IMPORT_EARLY_ARGS=(
  --manifest "${EARLY_MANIFEST}"
  --output-feature-store "${EARLY_STORE}"
  --feature-dim "${INPUT_FEATURE_DIM}"
)
IMPORT_LATE_ARGS=(
  --manifest "${LATE_MANIFEST}"
  --output-feature-store "${LATE_STORE}"
  --feature-dim "${LATE_FEATURE_DIM}"
)
IMPORT_TARGET_ARGS=(
  --manifest "${TARGET_MANIFEST}"
  --output-feature-store "${TARGET_STORE}"
  --normalize "${TARGET_NORMALIZE}"
  --require-coords
  --created-by "${CREATED_BY}"
)
SPLIT_ARGS=(
  --feature-store "${EARLY_STORE}"
  --output-dir "${SPLIT_DIR}"
  --train-ratio "${TRAIN_RATIO}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --seed "${SPLIT_SEED}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  BUILD_EARLY_ARGS+=(--overwrite)
  BUILD_LATE_ARGS+=(--overwrite)
  BUILD_TARGET_ARGS+=(--overwrite)
  IMPORT_EARLY_ARGS+=(--overwrite)
  IMPORT_LATE_ARGS+=(--overwrite)
  IMPORT_TARGET_ARGS+=(--overwrite)
fi

append_if_set EARLY_FEATURE_KEY --feature-key IMPORT_EARLY_ARGS
append_if_set LATE_FEATURE_KEY --feature-key IMPORT_LATE_ARGS
append_if_set EARLY_COORDS_KEY --coords-key IMPORT_EARLY_ARGS
append_if_set LATE_COORDS_KEY --coords-key IMPORT_LATE_ARGS
append_if_set TARGET_KEY --target-key IMPORT_TARGET_ARGS
append_if_set TARGET_COORDS_KEY --coords-key IMPORT_TARGET_ARGS

echo "[wsi-unsupervised-template] repo_root=${REPO_ROOT}"
echo "[wsi-unsupervised-template] output_root=${OUTPUT_ROOT}"

echo "[wsi-unsupervised-template] build early-feature manifest"
"${PYTHON_BIN}" scripts/build_generic_feature_manifest.py "${BUILD_EARLY_ARGS[@]}"

echo "[wsi-unsupervised-template] build late-feature manifest"
"${PYTHON_BIN}" scripts/build_generic_feature_manifest.py "${BUILD_LATE_ARGS[@]}"

echo "[wsi-unsupervised-template] build target manifest"
"${PYTHON_BIN}" scripts/build_wsi_importance_manifest.py "${BUILD_TARGET_ARGS[@]}"

echo "[wsi-unsupervised-template] import early feature store"
"${PYTHON_BIN}" scripts/import_generic_feature_store.py "${IMPORT_EARLY_ARGS[@]}"

echo "[wsi-unsupervised-template] import late feature store"
"${PYTHON_BIN}" scripts/import_generic_feature_store.py "${IMPORT_LATE_ARGS[@]}"

echo "[wsi-unsupervised-template] import tile-importance target store"
"${PYTHON_BIN}" scripts/import_wsi_importance_targets.py "${IMPORT_TARGET_ARGS[@]}"

echo "[wsi-unsupervised-template] validate individual stores"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${EARLY_STORE}" \
  --feature-dim "${INPUT_FEATURE_DIM}" \
  --require-coords
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${LATE_STORE}" \
  --feature-dim "${LATE_FEATURE_DIM}" \
  --require-coords
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${TARGET_STORE}" \
  --feature-dim 1 \
  --require-coords \
  --require-attention

echo "[wsi-unsupervised-template] create unsupervised train/val/test split"
if [[ "${OVERWRITE}" == "1" ]]; then
  SPLIT_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" scripts/split_wsi_feature_store.py "${SPLIT_ARGS[@]}"

echo "[wsi-unsupervised-template] run EAF importance-forecasting experiment"
EARLY_FEATURE_STORE="${EARLY_STORE}" \
LATE_FEATURE_STORE="${LATE_STORE}" \
TARGET_FEATURE_STORE="${TARGET_STORE}" \
SPLIT_DIR="${SPLIT_DIR}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
INPUT_FEATURE_DIM="${INPUT_FEATURE_DIM}" \
HIDDEN_DIM="${HIDDEN_DIM}" \
N_HEADS="${N_HEADS}" \
N_LAYERS="${N_LAYERS}" \
DROPOUT="${DROPOUT}" \
LOSS="${LOSS}" \
TOP_K="${TOP_K}" \
EPOCHS="${EPOCHS}" \
BATCH_SIZE="${BATCH_SIZE}" \
DEVICE="${DEVICE}" \
ALIGNMENT_MODE="${ALIGNMENT_MODE}" \
KEEP_RATIOS="${KEEP_RATIOS}" \
PRUNED_KEEP_RATIO="${PRUNED_KEEP_RATIO}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_wsi_first_real_experiment_template.sh

cat <<EOF
[wsi-unsupervised-template] done
early_store: ${EARLY_STORE}
late_store: ${LATE_STORE}
target_store: ${TARGET_STORE}
split_dir: ${SPLIT_DIR}
EOF
