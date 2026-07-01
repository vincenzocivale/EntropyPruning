#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PARENT_EXE="$(readlink -f "/proc/${PPID}/exe" 2>/dev/null || true)"
  if [[ -n "${PARENT_EXE}" && -x "${PARENT_EXE}" && "${PARENT_EXE}" == *python* ]]; then
    PYTHON_BIN="${PARENT_EXE}"
  else
    PYTHON_BIN="python"
  fi
fi

WORKDIR="${WORKDIR:-/tmp/eaf_wsi_importance_synthetic_smoke}"
EARLY_FEATURE_DIM="${EARLY_FEATURE_DIM:-8}"
LATE_FEATURE_DIM="${LATE_FEATURE_DIM:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-16}"
N_SLIDES="${N_SLIDES:-8}"
MIN_TILES="${MIN_TILES:-4}"
MAX_TILES="${MAX_TILES:-8}"
N_CLASSES="${N_CLASSES:-2}"
NOISE_STD="${NOISE_STD:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
FORECASTER_EPOCHS="${FORECASTER_EPOCHS:-1}"
KEEP_RATIOS="${KEEP_RATIOS:-0.10 0.5 1.0}"
PRUNED_KEEP_RATIO="${PRUNED_KEEP_RATIO:-0.10}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-0}"

EARLY_STORE="${WORKDIR}/features_layer2.h5"
LATE_STORE="${WORKDIR}/features_late.h5"
IMPORTANCE_STORE="${WORKDIR}/features_synthetic_importance.h5"
PRUNED_STORE="${WORKDIR}/features_late_pruned_keep_${PRUNED_KEEP_RATIO}.h5"

SPLIT_DIR="${WORKDIR}/splits"
CHECKPOINT_DIR="${WORKDIR}/checkpoints/importance_forecaster"
RESULTS_DIR="${WORKDIR}/results"
REPORT_DIR="${WORKDIR}/reports"

IMPORTANCE_PRUNING_CSV="${RESULTS_DIR}/wsi_importance_pruning.csv"

mkdir -p "${WORKDIR}" "${SPLIT_DIR}" "${RESULTS_DIR}" "${REPORT_DIR}"

echo "[wsi-importance-smoke] workdir=${WORKDIR}"

# Early, late, and importance-target stores are generated together in one
# deterministic pass (not three independent create_synthetic_wsi_feature_store.py
# calls) so slide ids, tile counts, and coords stay aligned by construction.
# The importance target is correlated with the *late* features:
#   importance_i = softmax(w^T late_feature_i + noise_i)
echo "[wsi-importance-smoke] creating synthetic early store (${EARLY_STORE})"
echo "[wsi-importance-smoke] creating synthetic late store (${LATE_STORE})"
echo "[wsi-importance-smoke] creating synthetic importance target store correlated with late features (${IMPORTANCE_STORE})"
"${PYTHON_BIN}" scripts/create_synthetic_wsi_importance_stores.py \
  --early-output "${EARLY_STORE}" \
  --late-output "${LATE_STORE}" \
  --importance-output "${IMPORTANCE_STORE}" \
  --n-slides "${N_SLIDES}" \
  --early-feature-dim "${EARLY_FEATURE_DIM}" \
  --late-feature-dim "${LATE_FEATURE_DIM}" \
  --min-tiles "${MIN_TILES}" \
  --max-tiles "${MAX_TILES}" \
  --n-classes "${N_CLASSES}" \
  --noise-std "${NOISE_STD}" \
  --seed "${SEED}" \
  --overwrite

echo "[wsi-importance-smoke] validating early feature store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${EARLY_STORE}" \
  --feature-dim "${EARLY_FEATURE_DIM}" \
  --require-coords

echo "[wsi-importance-smoke] validating late feature store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${LATE_STORE}" \
  --feature-dim "${LATE_FEATURE_DIM}" \
  --require-coords

echo "[wsi-importance-smoke] validating importance target store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${IMPORTANCE_STORE}" \
  --feature-dim 1 \
  --require-coords \
  --require-attention

echo "[wsi-importance-smoke] validating paired early/importance stores"
"${PYTHON_BIN}" scripts/validate_wsi_paired_feature_stores.py \
  --input-feature-store "${EARLY_STORE}" \
  --target-feature-store "${IMPORTANCE_STORE}" \
  --input-feature-dim "${EARLY_FEATURE_DIM}" \
  --alignment-mode coords \
  --require-coords \
  --require-attention \
  --output-json "${REPORT_DIR}/paired_validation.json"

echo "[wsi-importance-smoke] creating persistent train/val/test split"
"${PYTHON_BIN}" scripts/split_wsi_feature_store.py \
  --feature-store "${EARLY_STORE}" \
  --output-dir "${SPLIT_DIR}" \
  --train-ratio 0.5 \
  --val-ratio 0.25 \
  --test-ratio 0.25 \
  --stratify-label \
  --seed "${SEED}" \
  --overwrite

echo "[wsi-importance-smoke] training WSI tile importance forecaster (paired stores)"
"${PYTHON_BIN}" scripts/train_wsi_importance_forecaster.py \
  --input-feature-store "${EARLY_STORE}" \
  --target-feature-store "${IMPORTANCE_STORE}" \
  --output-dir "${CHECKPOINT_DIR}" \
  --input-feature-dim "${EARLY_FEATURE_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --n-heads 4 \
  --n-layers 1 \
  --dropout 0.0 \
  --loss kl \
  --top-k 2 \
  --epochs "${FORECASTER_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --alignment-mode coords \
  --require-coords \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --split-dir "${SPLIT_DIR}"

echo "[wsi-importance-smoke] evaluating importance pruning"
"${PYTHON_BIN}" scripts/evaluate_wsi_importance_pruning.py \
  --input-feature-store "${EARLY_STORE}" \
  --target-feature-store "${IMPORTANCE_STORE}" \
  --forecaster-checkpoint "${CHECKPOINT_DIR}/best_wsi_tile_importance_forecaster.pt" \
  --slide-ids-file "${SPLIT_DIR}/test.txt" \
  --keep-ratios ${KEEP_RATIOS} \
  --alignment-mode coords \
  --output-csv "${IMPORTANCE_PRUNING_CSV}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --overwrite

echo "[wsi-importance-smoke] creating pruned late feature store using early-store selection"
"${PYTHON_BIN}" scripts/create_pruned_wsi_feature_store.py \
  --selection-feature-store "${EARLY_STORE}" \
  --materialize-feature-store "${LATE_STORE}" \
  --output-feature-store "${PRUNED_STORE}" \
  --forecaster-checkpoint "${CHECKPOINT_DIR}/best_wsi_tile_importance_forecaster.pt" \
  --keep-ratio "${PRUNED_KEEP_RATIO}" \
  --alignment-mode coords \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --overwrite

echo "[wsi-importance-smoke] validating pruned late feature store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${PRUNED_STORE}" \
  --feature-dim "${LATE_FEATURE_DIM}" \
  --require-coords \
  | tee "${REPORT_DIR}/pruned_store_validation.json"

cat <<EOF
[wsi-importance-smoke] done
workdir: ${WORKDIR}
early_store: ${EARLY_STORE}
late_store: ${LATE_STORE}
importance_store: ${IMPORTANCE_STORE}
pruned_store: ${PRUNED_STORE}
split_dir: ${SPLIT_DIR}
forecaster_checkpoint: ${CHECKPOINT_DIR}/best_wsi_tile_importance_forecaster.pt
importance_pruning_csv: ${IMPORTANCE_PRUNING_CSV}
report_dir: ${REPORT_DIR}
EOF
