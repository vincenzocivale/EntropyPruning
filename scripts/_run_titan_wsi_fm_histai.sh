#!/usr/bin/env bash
set -uo pipefail
ROOT=/data2/home/vcivale/projects/imaging/data/WSI
REPO=/data2/home/vcivale/projects/imaging/EAF
TC_BASE="$ROOT/caches/tile_eaf/histai_eaf_wsi_v1/conch_v15/15055866f4a7"
OUTBASE="$ROOT/caches/wsi_eaf/histai_eaf_wsi_v1/conch_v15__titan"
LOGD="$ROOT/logs/wsi_eaf_fm_pipeline"
PY=/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python3

SUBSETS=(HISTAI-breast HISTAI-colorectal-b1 HISTAI-colorectal-b2 HISTAI-gastrointestinal HISTAI-hematologic HISTAI-skin-b1 HISTAI-skin-b2 HISTAI-thorax)

for s in "${SUBSETS[@]}"; do
  log="$LOGD/${s}.titan_wsi_fm.log"
  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $s"
    echo "COMMAND: $PY $REPO/scripts/wsi_eaf_infer_wsi_fm.py --tile-cache-dir $TC_BASE/$s --output-dir $OUTBASE/$s --model titan --device cuda"
  } > "$log"
  cd "$REPO" && "$PY" scripts/wsi_eaf_infer_wsi_fm.py \
    --tile-cache-dir "$TC_BASE/$s" \
    --output-dir "$OUTBASE/$s" \
    --model titan \
    --device cuda \
    >> "$log" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $s (exit=$?)" >> "$log"
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL SUBSETS DONE" >> "$LOGD/_all_subsets.log"
