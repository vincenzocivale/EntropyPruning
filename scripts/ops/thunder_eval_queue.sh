#!/bin/bash
# Runs queued jobs with at most MAX_PARALLEL concurrent processes.
# Each job is a full shell command (already includes its own log redirection).
# Existing jobs already running when this supervisor starts can be tracked via
# TRACK_PIDS so the cap accounts for them too.
set -u

MAX_PARALLEL=2
TRACK_PIDS=()   # PIDs started outside this script, counted toward the cap
QUEUE=()        # commands to launch when a slot frees up

# --- usage: thunder_eval_queue.sh <comma-separated-queue-job-names> <pid> [pid...] ---
QUEUE_NAMES_ARG="${1:-}"
shift || true
for arg in "$@"; do
    TRACK_PIDS+=("$arg")
done

# --- known job command templates, selected by name via QUEUE_NAMES below ---
declare -A JOB_CMD
JOB_CMD[src02_keep20]="cd /data2/home/vcivale/projects/imaging/EAF && \
EAF_WSI_ROOT=/data2/home/vcivale/data/WSI \
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python -u scripts/evaluation/evaluate_tile_thunder.py \
  --experiment-id tile_eval_thunder_conch15 --variant-id src02_keep20 \
  --pruned-checkpoint /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/distillation/tile_joint_source_keep_conch15/src02_keep20/seed_42/best.pt \
  --forecaster-checkpoint /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/conch_v15/conch_v15_src02/best_conch_v15_src02.pt \
  --base-data-folder /data2/home/vcivale/projects/imaging/data/thunder-tiles/datasets \
  --model-name titan --batch-size 16 --num-workers 2 --seed 42 \
  > logs/tile_eval_thunder_conch15_src02_keep20_queued_\$(date +%Y%m%d_%H%M%S).log 2>&1"

JOB_CMD[src02_keep10_src01_keep20]="cd /data2/home/vcivale/projects/imaging/EAF && \
EAF_WSI_ROOT=/data2/home/vcivale/data/WSI \
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python -u scripts/evaluation/evaluate_tile_thunder.py \
  --experiment-id tile_eval_thunder_conch15 --variant-id src02_keep10 \
  --pruned-checkpoint /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/distillation/tile_joint_source_keep_conch15/src02_keep10/seed_42/best.pt \
  --forecaster-checkpoint /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/conch_v15/conch_v15_src02/best_conch_v15_src02.pt \
  --also-variant-id src01_keep20 \
  --also-pruned-checkpoint /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/distillation/tile_joint_source_keep_conch15/src01_keep20/seed_42/best.pt \
  --also-forecaster-checkpoint /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/forecaster/tile_source_ablation_conch15/src01/seed_42/best.pt \
  --base-data-folder /data2/home/vcivale/projects/imaging/data/thunder-tiles/datasets \
  --model-name titan --batch-size 16 --num-workers 2 --seed 42 \
  > logs/tile_eval_thunder_conch15_src02_keep10_src01_keep20_queued_\$(date +%Y%m%d_%H%M%S).log 2>&1"

JOB_CMD[src01_keep15]="cd /data2/home/vcivale/projects/imaging/EAF && \
EAF_WSI_ROOT=/data2/home/vcivale/data/WSI \
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python -u scripts/training/distill_tile_encoder.py \
  --experiment-id tile_joint_source_keep_conch15 --variant-id src01_keep15 --model-name titan \
  --manifest /data2/home/vcivale/data/WSI/datasets/pretraining/histai_eaf_wsi_v1/manifests/conch_v15_complete_v1/slides.csv \
  --data-root /data2/home/vcivale/data/WSI \
  --target-cache-index /data2/home/vcivale/data/WSI/datasets/pretraining/histai_eaf_wsi_v1/manifests/conch_v15_complete_v1/tile_cache_index.csv \
  --forecaster-ckpt /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/forecaster/tile_source_ablation_conch15/src01/seed_42/best.pt \
  --prune-layer 1 --keep-ratio 0.15 \
  --hidden 256 --n-heads 4 --n-layers 2 --dropout 0.1 \
  --lora-r 8 --lora-alpha 32 --lora-dropout 0.05 \
  --split-column split --split-seed 42 --slide-group diagnostic \
  --exclude-cohort HISTAI-mixed HISTAI-skin-b2 \
  --batch-size 32 --slides-per-batch 16 --train-wsi-fraction 0.5 --tiles-per-wsi 100 \
  --val-wsis 128 --val-tiles-per-wsi 8 --cohort-balance-power 0.5 \
  --num-workers 4 --prefetch-factor 2 --slide-cache-size 20 --openslide-cache-mib 256 \
  --epochs 20 --lr 2e-4 --weight-decay 0.01 \
  --cosine-weight 1.0 --mse-weight 1.0 --pairwise-weight 0.25 \
  --grad-accum 4 --amp-dtype bf16 \
  --early-stopping-patience 3 --early-stopping-min-delta 0.001 \
  --seed 42 --wandb-mode online --log-every 25 \
  > logs/distill_src01_keep15_queued_\$(date +%Y%m%d_%H%M%S).log 2>&1"

JOB_CMD[final]="cd /data2/home/vcivale/projects/imaging/EAF && \
EAF_WSI_ROOT=/data2/home/vcivale/data/WSI \
/data2/home/vcivale/miniconda3/envs/eaf_env/bin/python -u scripts/evaluation/evaluate_tile_thunder.py \
  --experiment-id tile_eval_thunder_conch15 --variant-id final \
  --pruned-checkpoint /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/distillation/tile_joint_source_keep_conch15/src02_keep15/seed_42/best.pt \
  --forecaster-checkpoint /data2/home/vcivale/data/WSI/checkpoints/tile_eaf/conch_v15/conch_v15_src02/best_conch_v15_src02.pt \
  --base-data-folder /data2/home/vcivale/projects/imaging/data/thunder-tiles/datasets \
  --model-name titan --batch-size 16 --num-workers 2 --seed 42 \
  > logs/tile_eval_thunder_conch15_final_queued_\$(date +%Y%m%d_%H%M%S).log 2>&1"

# --- second script argument: comma-separated job names to enqueue (only jobs
# NOT already covered by the tracked PIDs in the first argument belong here,
# otherwise a finished tracked job would be relaunched as a duplicate) ---
IFS=',' read -ra QUEUE_NAMES <<< "${QUEUE_NAMES_ARG:-}"
for name in "${QUEUE_NAMES[@]:-}"; do
    [ -n "$name" ] && QUEUE+=("${JOB_CMD[$name]}")
done

RUNNING_PIDS=()
LOGFILE="logs/thunder_eval_queue_supervisor.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGFILE"; }

log "supervisor started; tracking pre-existing pids: ${TRACK_PIDS[*]}; queue length: ${#QUEUE[@]}"

alive() { kill -0 "$1" 2>/dev/null; }

while true; do
    # prune dead tracked pids (avoid bash's empty-array + ${:-} quirk by
    # guarding on length before iterating)
    new_track=()
    if [ "${#TRACK_PIDS[@]}" -gt 0 ]; then
        for pid in "${TRACK_PIDS[@]}"; do
            alive "$pid" && new_track+=("$pid")
        done
    fi
    TRACK_PIDS=("${new_track[@]}")

    # prune dead running (queue-launched) pids
    new_running=()
    if [ "${#RUNNING_PIDS[@]}" -gt 0 ]; then
        for pid in "${RUNNING_PIDS[@]}"; do
            alive "$pid" && new_running+=("$pid")
        done
    fi
    if [ "${#new_running[@]}" -lt "${#RUNNING_PIDS[@]}" ]; then
        log "a queued job finished (pid set shrank ${#RUNNING_PIDS[@]} -> ${#new_running[@]})"
    fi
    RUNNING_PIDS=("${new_running[@]}")

    active=$(( ${#TRACK_PIDS[@]} + ${#RUNNING_PIDS[@]} ))

    while [ "$active" -lt "$MAX_PARALLEL" ] && [ "${#QUEUE[@]}" -gt 0 ]; do
        cmd="${QUEUE[0]}"
        QUEUE=("${QUEUE[@]:1}")
        log "launching queued job (slot free, active=$active): ${cmd:0:120}..."
        bash -c "$cmd" &
        newpid=$!
        RUNNING_PIDS+=("$newpid")
        active=$(( active + 1 ))
        log "queued job started as pid $newpid"
    done

    if [ "${#TRACK_PIDS[@]}" -eq 0 ] && [ "${#RUNNING_PIDS[@]}" -eq 0 ] && [ "${#QUEUE[@]}" -eq 0 ]; then
        log "all tracked and queued jobs finished; supervisor exiting"
        break
    fi

    sleep 60
done
