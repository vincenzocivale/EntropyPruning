#!/usr/bin/env bash
set -euo pipefail

: "${THUNDER_BASE_DATA_FOLDER:?THUNDER_BASE_DATA_FOLDER non definita}"

THUNDER_DATASETS="$THUNDER_BASE_DATA_FOLDER/datasets"
WSI_ROOT=/data2/home/vcivale/projects/imaging/data/WSI
OUT_DIR="$WSI_ROOT/catalog/thunder_overlap"

mkdir -p "$OUT_DIR"

extract_cases_from_names() {
  local source_root="$1"
  local output="$2"

  if [[ ! -d "$source_root" ]]; then
    echo "WARN: dataset assente: $source_root" >&2
    : > "$output"
    return
  fi

  find "$source_root" \
    \( -type f -o -type d \) \
    -printf '%f\n' \
    | grep -oE 'TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}' \
    | tr '[:lower:]' '[:upper:]' \
    | sort -u \
    > "$output" || true
}

extract_cases_from_names \
  "$THUNDER_DATASETS/tcga_uniform" \
  "$OUT_DIR/tcga_uniform_cases.txt"

extract_cases_from_names \
  "$THUNDER_DATASETS/tcga_crc_msi" \
  "$OUT_DIR/tcga_crc_msi_cases.txt"

extract_cases_from_names \
  "$THUNDER_DATASETS/tcga_tils" \
  "$OUT_DIR/tcga_tils_cases.txt"

extract_cases_from_names \
  "$THUNDER_DATASETS/ccrcc" \
  "$OUT_DIR/ccrcc_tcga_cases.txt"

cat \
  "$OUT_DIR/tcga_uniform_cases.txt" \
  "$OUT_DIR/tcga_crc_msi_cases.txt" \
  "$OUT_DIR/tcga_tils_cases.txt" \
  "$OUT_DIR/ccrcc_tcga_cases.txt" \
  | sort -u \
  > "$OUT_DIR/all_thunder_tcga_cases.txt"

echo "=== THUNDER TCGA CASE AUDIT ==="

for file in \
  tcga_uniform_cases.txt \
  tcga_crc_msi_cases.txt \
  tcga_tils_cases.txt \
  ccrcc_tcga_cases.txt \
  all_thunder_tcga_cases.txt
do
  printf '%-32s %8d\n' \
    "$file" \
    "$(wc -l < "$OUT_DIR/$file")"
done

echo
echo "Output: $OUT_DIR"
