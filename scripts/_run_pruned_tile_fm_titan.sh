#!/usr/bin/env bash
set -uo pipefail
ROOT=/data2/home/vcivale/projects/imaging/data/WSI
REPO=/data2/home/vcivale/projects/imaging/EAF
PY=/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python3
LOGD="$ROOT/logs/pruned_tile_fm"

MANIFEST="$ROOT/datasets/pretraining/histai_eaf_wsi_v1/manifests/conch_v15_complete_v1/slides.csv"
TARGET_CACHE_INDEX="$ROOT/datasets/pretraining/histai_eaf_wsi_v1/manifests/conch_v15_complete_v1/tile_cache_index.csv"
FORECASTER="$REPO/checkpoints/tile_eaf/titan_tpw500/best_conch_v15_histai_complete7_tpw500_lr2e4_scratch_ep20.pt"

# keep_ratio = fraction of tokens KEPT after layer 2; "pruning al 80/90%" ->
# keep_ratio 0.20/0.10. The 70% run (keep_ratio=0.30) was stopped by the user
# after 3 epochs (val loss 0.02243 -> 0.02118 -> 0.02109, judged converged
# enough); its checkpoint is preserved at
# checkpoints/pruned_finetuned/titan/best_titan_prune2_pruned70pct_distill_ep20_adapter.pt.
# --epochs 5 this time (was 20) per explicit user instruction. Sequential, not
# parallel: 3x this heavy a job at once on one A100 already caused OOM/hangs.
#
# --target-cache-index reads the frozen unpruned model's final per-tile embedding
# straight from the existing Tile-EAF cache (`eaf.py cache tile`) instead of
# recomputing it online via a second full unpruned forward every step.
#
# batch-size=64 (not 128): peak memory depends on keep_ratio -- more surviving
# tokens from layer 2 onward means quadratically more per-tile self-attention
# cost, not just linearly more. batch-size=128 measured ~26GB at keep_ratio=0.10
# but OOM'd (>32GB, crashed) at keep_ratio=0.30. batch-size=64 measured ~24GB at
# keep_ratio=0.30 (the worst case of the three), validated on a real-data probe;
# grad-accum=2 keeps the effective batch at 128.
declare -A KEEP_RATIOS=( [80]=0.20 [90]=0.10 )

for prune_pct in 80 90; do
  keep=${KEEP_RATIOS[$prune_pct]}
  run_name="titan_prune2_pruned${prune_pct}pct_distill_ep5"
  log="$LOGD/${run_name}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $run_name (keep_ratio=$keep)" > "$log"
  cd "$REPO" && "$PY" scripts/finetune_wsi_tile_encoder_pruned_online.py \
    --model-name titan \
    --manifest "$MANIFEST" \
    --data-root "$ROOT" \
    --forecaster-ckpt "$FORECASTER" \
    --target-cache-index "$TARGET_CACHE_INDEX" \
    --prune-layer 2 \
    --keep-ratio "$keep" \
    --hidden 256 --n-heads 4 --n-layers 2 \
    --tiles-per-wsi 500 \
    --batch-size 64 --slides-per-batch 16 --grad-accum 2 \
    --train-wsi-fraction 0.50 --val-wsis 128 --val-tiles-per-wsi 16 \
    --num-workers 16 --slide-cache-size 20 --openslide-cache-mib 256 \
    --amp-dtype bf16 \
    --lr 2e-4 \
    --epochs 5 --early-stopping-patience 6 \
    --wandb-project Tile-FM-Pruned \
    --wandb-mode online \
    --run-name "$run_name" \
    >> "$log" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $run_name (exit=$?)" >> "$log"
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL PRUNE LEVELS DONE" >> "$LOGD/_all_runs.log"
