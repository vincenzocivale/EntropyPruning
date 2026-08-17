#!/usr/bin/env bash
set -uo pipefail
ROOT=/data2/home/vcivale/projects/imaging/data/WSI
REPO=/data2/home/vcivale/projects/imaging/EAF
TC_BASE="$ROOT/caches/tile_eaf/histai_eaf_wsi_v1/conch_v15/15055866f4a7"
OUTBASE="$ROOT/caches/wsi_eaf/histai_eaf_wsi_v1/conch_v15__titan"
LOGD="$ROOT/logs/wsi_eaf_fm_pipeline"
PY=/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python3

# Only the subsets that had CUDA-OOM errors in the first pass (74 slides total,
# see complete=/errors= summary lines in <subset>.titan_wsi_fm.log). The bridge
# script already skips slides whose output is `complete`, so rerunning the same
# command retries only what's missing -- no need to enumerate the failed slide_ids.
SUBSETS=(HISTAI-breast HISTAI-hematologic HISTAI-skin-b1 HISTAI-skin-b2 HISTAI-thorax)

for s in "${SUBSETS[@]}"; do
  log="$LOGD/${s}.titan_wsi_fm.retry1.log"
  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RETRY START $s"
  } > "$log"
  cd "$REPO" && "$PY" scripts/wsi_eaf_infer_wsi_fm.py \
    --tile-cache-dir "$TC_BASE/$s" \
    --output-dir "$OUTBASE/$s" \
    --model titan \
    --device cuda \
    >> "$log" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] RETRY DONE $s (exit=$?)" >> "$log"
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] RETRY PASS 1 DONE" >> "$LOGD/_all_subsets.log"
