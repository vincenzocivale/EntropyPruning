#!/bin/bash
# Pipeline: Phase1 (LoRA) → Phase2 (Forecaster) → Phase3 (Pruned) x3 keep-ratios
# Datasets: mhist, wilds, spider_colorectal
# Pruning: 10% (keep=0.9), 20% (keep=0.8), 30% (keep=0.7)
set -euo pipefail

PYTHON=/home/oem/EAF/EntropyPruning/.venv/bin/python
THUNDER_CLI=/home/oem/EAF/EntropyPruning/.venv/bin/thunder
SCRIPTS=/home/oem/EAF/EntropyPruning/scripts
BASE_DATA=/data/EAF_data/thunder/datasets
CKPT_BASE=/data/EAF_data/thunder/checkpoints
LOG_DIR=/data/EAF_data/thunder/logs/lora_experiments
MODEL=uni
DATASETS=(mhist wilds spider_colorectal)
KEEP_RATIOS=(0.9 0.8 0.7)
PRUNE_LAYER=2
PRUNE_LAYER_FMT=$(printf "%02d" $PRUNE_LAYER)

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── Download datasets ──────────────────────────────────────────────────────────

download_dataset() {
    local ds=$1
    local ds_dir="$BASE_DATA/$ds"
    local split_file="$BASE_DATA/data_splits/${ds}.json"

    if [ -d "$ds_dir" ] && [ -f "$split_file" ]; then
        log "[SKIP] Download $ds: already present"
        return 0
    fi
    log "[RUN] Downloading $ds ..."
    THUNDER_BASE_DATA_FOLDER=/data/EAF_data/thunder \
        "$THUNDER_CLI" download-datasets "$ds" --make-splits \
        2>&1 | tee "$LOG_DIR/download_${ds}.log"
    log "[DONE] Download $ds"
}

# ── Phase 1: LoRA classifier ───────────────────────────────────────────────────

phase1() {
    local ds=$1
    local ckpt="$CKPT_BASE/$ds/${MODEL}_lora/best_model.pt"

    if [ -f "$ckpt" ]; then
        log "[SKIP] Phase1 $ds: $ckpt exists"
        return 0
    fi
    log "[RUN] Phase1 LoRA $ds ..."
    "$PYTHON" "$SCRIPTS/train_classifier.py" \
        --model-name "$MODEL" \
        --dataset-name "$ds" \
        --base-data-folder "$BASE_DATA" \
        --adaptation lora \
        --output-dir "$CKPT_BASE/$ds/${MODEL}_lora" \
        --wandb-project eaf \
        2>&1 | tee "$LOG_DIR/phase1_${MODEL}_${ds}.log"
    [ -f "$ckpt" ] || die "Phase1 $ds did not produce $ckpt"
    log "[DONE] Phase1 $ds"
}

# ── Phase 2: Attention Forecaster ─────────────────────────────────────────────

phase2() {
    local ds=$1
    local forecaster_dir="$CKPT_BASE/$ds/${MODEL}_forecaster_lora"
    # run_name inside train_forecaster: {model}_{ds}_phase2_src{02d}_tgt{02d}
    local forecaster_ckpt="$forecaster_dir/forecaster_${MODEL}_${ds}_phase2_src${PRUNE_LAYER_FMT}_tgt23.pt"

    if [ -f "$forecaster_ckpt" ]; then
        log "[SKIP] Phase2 $ds: $forecaster_ckpt exists"
        return 0
    fi
    log "[RUN] Phase2 Forecaster $ds ..."
    "$PYTHON" "$SCRIPTS/train_forecaster.py" \
        --model-name "$MODEL" \
        --dataset-name "$ds" \
        --base-data-folder "$BASE_DATA" \
        --adaptation lora \
        --classifier-ckpt "$CKPT_BASE/$ds/${MODEL}_lora/best_model.pt" \
        --forecaster-dir "$forecaster_dir" \
        --cache-dir "$CKPT_BASE/$ds/lora_cache" \
        --wandb-project eaf \
        2>&1 | tee "$LOG_DIR/phase2_${MODEL}_${ds}.log"
    [ -f "$forecaster_ckpt" ] || die "Phase2 $ds did not produce $forecaster_ckpt"
    log "[DONE] Phase2 $ds"
}

# ── Phase 3: Fine-tune with pruning ───────────────────────────────────────────

phase3() {
    local ds=$1
    local keep_ratio=$2
    local keep_pct=$(python3 -c "print(int($keep_ratio*100))")
    local prune_pct=$((100 - keep_pct))
    local run_name="${MODEL}_${ds}_prune${PRUNE_LAYER}_keep${keep_pct}"
    local result_file="$CKPT_BASE/$ds/${MODEL}_pruned_lora/results_${run_name}.json"
    local forecaster_dir="$CKPT_BASE/$ds/${MODEL}_forecaster_lora"
    local forecaster_ckpt="$forecaster_dir/forecaster_${MODEL}_${ds}_phase2_src${PRUNE_LAYER_FMT}_tgt23.pt"

    if [ -f "$result_file" ]; then
        log "[SKIP] Phase3 $ds prune${prune_pct}%: results exist"
        return 0
    fi
    log "[RUN] Phase3 $ds prune=${prune_pct}% (keep-ratio=$keep_ratio) ..."
    "$PYTHON" "$SCRIPTS/finetune_pruned.py" \
        --model-name "$MODEL" \
        --dataset-name "$ds" \
        --base-data-folder "$BASE_DATA" \
        --adaptation lora \
        --classifier-ckpt "$CKPT_BASE/$ds/${MODEL}_lora/best_model.pt" \
        --forecaster-ckpt "$forecaster_ckpt" \
        --output-dir "$CKPT_BASE/$ds/${MODEL}_pruned_lora" \
        --prune-layer "$PRUNE_LAYER" \
        --keep-ratio "$keep_ratio" \
        --eval-baseline \
        --wandb-project eaf \
        2>&1 | tee "$LOG_DIR/phase3_${MODEL}_${ds}_keep${keep_pct}.log"
    [ -f "$result_file" ] || die "Phase3 $ds keep${keep_pct} did not produce $result_file"
    log "[DONE] Phase3 $ds prune=${prune_pct}%"
}

# ── Main ───────────────────────────────────────────────────────────────────────

log "=== Starting LoRA experiments: UNI on mhist, wilds, spider_colorectal ==="
log "Pruning rates: 10%, 20%, 30% (keep-ratios: 0.9, 0.8, 0.7)"

# Step 0: download missing datasets
for ds in "${DATASETS[@]}"; do
    download_dataset "$ds"
done

# Steps 1-3: pipeline per dataset
for ds in "${DATASETS[@]}"; do
    log "--- Dataset: $ds ---"
    phase1 "$ds"
    phase2 "$ds"
    for kr in "${KEEP_RATIOS[@]}"; do
        phase3 "$ds" "$kr"
    done
done

log "=== All experiments complete ==="

# ── Summary ────────────────────────────────────────────────────────────────────

log "Results summary:"
for ds in "${DATASETS[@]}"; do
    result_dir="$CKPT_BASE/$ds/${MODEL}_pruned_lora"
    if [ -d "$result_dir" ]; then
        echo "  $ds:"
        for f in "$result_dir"/results_*.json; do
            [ -f "$f" ] || continue
            python3 -c "
import json, pathlib
d = json.loads(pathlib.Path('$f').read_text())
print(f\"    prune={100-int(d['keep_ratio']*100)}% | test_acc={d.get('test_acc','?'):.4f} | test_f1={d.get('test_f1_macro','?'):.4f}\")
" 2>/dev/null || echo "    $f (parse error)"
        done
    fi
done
