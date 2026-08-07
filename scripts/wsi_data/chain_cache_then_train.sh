#!/usr/bin/env bash
# Wait for the WSI tile-EAF cache build to finish (both split .done markers
# present), then launch full-epoch EAF training from that cache. Polls the
# completion markers written by build_wsi_tile_eaf_cache.py rather than a PID,
# so it is safe against the extraction process crashing (it just waits
# forever instead of training on a partial/incomplete cache).
set -uo pipefail

RUN_DIR="${RUN_DIR:?set RUN_DIR to the cache run directory}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[chain] waiting for cache completion markers in $RUN_DIR"
while [[ ! -f "$RUN_DIR/train.done" || ! -f "$RUN_DIR/val.done" ]]; do
    sleep 60
done
echo "[chain] cache complete ($(date -u +%FT%TZ)); launching EAF training"

cd "$REPO_ROOT"
source ~/.bashrc 2>/dev/null || true
conda activate eaf_env

python scripts/train_forecaster_from_cache.py \
    --cache-dir "$RUN_DIR" \
    --output-dir results/wsi_tile_eaf_cached \
    --experiment-name titan_wsi_cached_h1024_e50 \
    --epochs 50 \
    --hidden 1024 \
    --batch-size 128 \
    --num-workers 8 \
    --wandb-project eaf-wsi-cached

echo "[chain] training finished ($(date -u +%FT%TZ))"
