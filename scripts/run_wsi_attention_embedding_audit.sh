#!/usr/bin/env bash

# Label-free WSI attention/embedding audit.
#
# Required:
#   FEATURE_STORE=/path/to/features.h5
#   OUTPUT_DIR=/path/to/output
#
# Attention source (choose at most one; omit both for embedded HDF5 attention):
#   ATTENTION_MANIFEST=/path/to/attention.csv
#   ATTENTION_STORE=/path/to/legacy_attention.h5
#
# Optional feature import:
#   TRIDENT_MANIFEST=/path/to/trident_features.csv
# If FEATURE_STORE does not exist and TRIDENT_MANIFEST is set, the script imports
# the existing TRIDENT features without recomputing WSI preprocessing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_STORE="${FEATURE_STORE:?set FEATURE_STORE to the schema-v1 HDF5 feature store}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for audit artifacts}"
TRIDENT_MANIFEST="${TRIDENT_MANIFEST:-}"
ATTENTION_MANIFEST="${ATTENTION_MANIFEST:-}"
ATTENTION_STORE="${ATTENTION_STORE:-}"
ALIGNMENT="${ALIGNMENT:-auto}"
ATTENTION_NORMALIZATION="${ATTENTION_NORMALIZATION:-auto}"
KNN_K="${KNN_K:-0}"
KNN_REFERENCE_SIZE="${KNN_REFERENCE_SIZE:-4096}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"

if [[ -n "${ATTENTION_MANIFEST}" && -n "${ATTENTION_STORE}" ]]; then
  echo "Set only one of ATTENTION_MANIFEST or ATTENTION_STORE." >&2
  exit 2
fi

if [[ ! -f "${FEATURE_STORE}" ]]; then
  if [[ -z "${TRIDENT_MANIFEST}" ]]; then
    echo "FEATURE_STORE does not exist and TRIDENT_MANIFEST is not set: ${FEATURE_STORE}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${FEATURE_STORE}")"
  "${PYTHON_BIN}" scripts/import_trident_feature_store.py \
    --manifest "${TRIDENT_MANIFEST}" \
    --output-feature-store "${FEATURE_STORE}"
fi

args=(
  --feature-store "${FEATURE_STORE}"
  --output-dir "${OUTPUT_DIR}"
  --alignment "${ALIGNMENT}"
  --attention-normalization "${ATTENTION_NORMALIZATION}"
  --knn-k "${KNN_K}"
  --knn-reference-size "${KNN_REFERENCE_SIZE}"
  --knn-chunk-size "${KNN_CHUNK_SIZE}"
)

if [[ -n "${ATTENTION_MANIFEST}" ]]; then
  args+=(--attention-manifest "${ATTENTION_MANIFEST}")
elif [[ -n "${ATTENTION_STORE}" ]]; then
  args+=(--attention-store "${ATTENTION_STORE}")
fi

# Model-specific native tensor semantics remain explicit. Example:
#   EXTRA_AUDIT_ARGS='--tile-axis 3 --tile-slice-start 1 --attention-select 0=-1 --attention-select 2=0'
# shellcheck disable=SC2206
extra_args=(${EXTRA_AUDIT_ARGS:-})
args+=("${extra_args[@]}")

"${PYTHON_BIN}" scripts/analyze_wsi_attention_embeddings.py "${args[@]}"
