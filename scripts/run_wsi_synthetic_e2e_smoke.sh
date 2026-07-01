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

WORKDIR="${WORKDIR:-/tmp/eaf_wsi_synthetic_e2e}"
FEATURE_DIM="${FEATURE_DIM:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-16}"
N_SLIDES="${N_SLIDES:-8}"
MIN_TILES="${MIN_TILES:-4}"
MAX_TILES="${MAX_TILES:-8}"
N_CLASSES="${N_CLASSES:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ABMIL_EPOCHS="${ABMIL_EPOCHS:-1}"
FORECASTER_EPOCHS="${FORECASTER_EPOCHS:-1}"
KEEP_RATIOS="${KEEP_RATIOS:-0.5 1.0}"
PRUNED_KEEP_RATIO="${PRUNED_KEEP_RATIO:-0.5}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-0}"

RAW_STORE="${WORKDIR}/features_raw.h5"
ATTENTION_STORE="${WORKDIR}/features_abmil_attention.h5"
PRUNED_STORE="${WORKDIR}/features_pruned_keep_${PRUNED_KEEP_RATIO}.h5"

ABMIL_DIR="${WORKDIR}/checkpoints/abmil"
FORECASTER_DIR="${WORKDIR}/checkpoints/forecaster"
SPLIT_DIR="${WORKDIR}/splits"
RESULTS_DIR="${WORKDIR}/results"
REPORT_DIR="${WORKDIR}/reports/pruning"

FORECASTER_PRUNING_CSV="${RESULTS_DIR}/wsi_forecaster_pruning.csv"
ABMIL_AGREEMENT_CSV="${RESULTS_DIR}/wsi_abmil_pruning_agreement.csv"

mkdir -p "${WORKDIR}" "${SPLIT_DIR}" "${RESULTS_DIR}" "${REPORT_DIR}"

echo "[wsi-e2e] workdir=${WORKDIR}"
echo "[wsi-e2e] creating synthetic feature store"
"${PYTHON_BIN}" scripts/create_synthetic_wsi_feature_store.py \
  --output "${RAW_STORE}" \
  --n-slides "${N_SLIDES}" \
  --feature-dim "${FEATURE_DIM}" \
  --min-tiles "${MIN_TILES}" \
  --max-tiles "${MAX_TILES}" \
  --n-classes "${N_CLASSES}" \
  --seed "${SEED}" \
  --overwrite

echo "[wsi-e2e] validating raw feature store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${RAW_STORE}" \
  --feature-dim "${FEATURE_DIM}" \
  --require-coords

echo "[wsi-e2e] creating persistent train/val/test split"
"${PYTHON_BIN}" scripts/split_wsi_feature_store.py \
  --feature-store "${RAW_STORE}" \
  --output-dir "${SPLIT_DIR}" \
  --train-ratio 0.5 \
  --val-ratio 0.25 \
  --test-ratio 0.25 \
  --stratify-label \
  --seed "${SEED}" \
  --overwrite

echo "[wsi-e2e] training ABMIL teacher"
"${PYTHON_BIN}" scripts/train_wsi_abmil.py \
  --feature-store "${RAW_STORE}" \
  --output-dir "${ABMIL_DIR}" \
  --feature-dim "${FEATURE_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --n-classes "${N_CLASSES}" \
  --dropout 0.0 \
  --epochs "${ABMIL_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --split-dir "${SPLIT_DIR}"

echo "[wsi-e2e] extracting ABMIL attention"
"${PYTHON_BIN}" scripts/extract_wsi_abmil_attention.py \
  --input-feature-store "${RAW_STORE}" \
  --output-feature-store "${ATTENTION_STORE}" \
  --abmil-checkpoint "${ABMIL_DIR}/best_abmil_classifier.pt" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --overwrite

echo "[wsi-e2e] validating attention feature store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${ATTENTION_STORE}" \
  --feature-dim "${FEATURE_DIM}" \
  --require-attention \
  --require-coords

echo "[wsi-e2e] training WSI tile attention forecaster"
"${PYTHON_BIN}" scripts/train_wsi_attention_forecaster.py \
  --feature-store "${ATTENTION_STORE}" \
  --output-dir "${FORECASTER_DIR}" \
  --feature-dim "${FEATURE_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --n-heads 4 \
  --n-layers 1 \
  --dropout 0.0 \
  --epochs "${FORECASTER_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --top-k 2 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --split-dir "${SPLIT_DIR}"

echo "[wsi-e2e] evaluating forecaster pruning"
"${PYTHON_BIN}" scripts/evaluate_wsi_forecaster_pruning.py \
  --feature-store "${ATTENTION_STORE}" \
  --forecaster-checkpoint "${FORECASTER_DIR}/best_wsi_tile_attention_forecaster.pt" \
  --slide-ids-file "${SPLIT_DIR}/test.txt" \
  --keep-ratios ${KEEP_RATIOS} \
  --output-csv "${FORECASTER_PRUNING_CSV}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --overwrite

echo "[wsi-e2e] evaluating ABMIL full-vs-pruned agreement"
"${PYTHON_BIN}" scripts/evaluate_wsi_abmil_pruning_agreement.py \
  --feature-store "${ATTENTION_STORE}" \
  --abmil-checkpoint "${ABMIL_DIR}/best_abmil_classifier.pt" \
  --forecaster-checkpoint "${FORECASTER_DIR}/best_wsi_tile_attention_forecaster.pt" \
  --slide-ids-file "${SPLIT_DIR}/test.txt" \
  --keep-ratios ${KEEP_RATIOS} \
  --output-csv "${ABMIL_AGREEMENT_CSV}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --overwrite

echo "[wsi-e2e] plotting pruning curves"
"${PYTHON_BIN}" scripts/plot_wsi_pruning_curves.py \
  --forecaster-pruning-csv "${FORECASTER_PRUNING_CSV}" \
  --abmil-agreement-csv "${ABMIL_AGREEMENT_CSV}" \
  --output-dir "${REPORT_DIR}" \
  --overwrite

echo "[wsi-e2e] creating pruned feature store"
"${PYTHON_BIN}" scripts/create_pruned_wsi_feature_store.py \
  --input-feature-store "${ATTENTION_STORE}" \
  --output-feature-store "${PRUNED_STORE}" \
  --forecaster-checkpoint "${FORECASTER_DIR}/best_wsi_tile_attention_forecaster.pt" \
  --keep-ratio "${PRUNED_KEEP_RATIO}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --overwrite

echo "[wsi-e2e] validating pruned feature store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${PRUNED_STORE}" \
  --feature-dim "${FEATURE_DIM}" \
  --require-attention \
  --require-coords

cat <<EOF
[wsi-e2e] done
workdir: ${WORKDIR}
raw_store: ${RAW_STORE}
attention_store: ${ATTENTION_STORE}
pruned_store: ${PRUNED_STORE}
split_dir: ${SPLIT_DIR}
abmil_checkpoint: ${ABMIL_DIR}/best_abmil_classifier.pt
forecaster_checkpoint: ${FORECASTER_DIR}/best_wsi_tile_attention_forecaster.pt
forecaster_pruning_csv: ${FORECASTER_PRUNING_CSV}
abmil_agreement_csv: ${ABMIL_AGREEMENT_CSV}
report_dir: ${REPORT_DIR}
EOF
