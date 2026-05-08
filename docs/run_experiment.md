# Avviare un esperimento: encoder + dataset

Guida rapida per lanciare le tre fasi del pipeline su questa macchina (Nanopore-PC).
Per i dettagli su ogni flag, vedere [training_pipeline.md](training_pipeline.md).

---

## 0. Variabili

```bash
MODEL=uni                                    # encoder: uni, hoptimus0, virchow, dinov2base, ...
DATASET=mhist                                # dataset: mhist, wilds, spider_colorectal, crc, ...

PYTHON=/home/oem/EAF/EntropyPruning/.venv/bin/python
SCRIPTS=/home/oem/EAF/EntropyPruning/scripts
BASE_DATA=/data/EAF_data/thunder/datasets
CKPT=$( echo /data/EAF_data/thunder/checkpoints/$DATASET )
```

Se il dataset non è ancora scaricato:

```bash
THUNDER_BASE_DATA_FOLDER=/data/EAF_data/thunder \
    /home/oem/EAF/EntropyPruning/.venv/bin/thunder download-datasets $DATASET --make-splits
```

---

## Fase 1 — Classificatore LoRA

Addestra la testa di classificazione con adattamento LoRA.
Output: `$CKPT/${MODEL}_lora/best_model.pt`

```bash
$PYTHON $SCRIPTS/train_classifier.py \
    --model-name  $MODEL   \
    --dataset-name $DATASET \
    --base-data-folder $BASE_DATA \
    --adaptation lora \
    --output-dir $CKPT/${MODEL}_lora \
    --wandb-project eaf
```

Parametri modificabili:

| Flag | Default | Note |
|------|---------|------|
| `--epochs` | 20 | |
| `--batch-size` | 8 | abbassare in caso di OOM |
| `--lora-r` | 8 | rank LoRA |
| `--lora-alpha` | 32 | |
| `--lr-backbone` | 1e-5 | |
| `--lr-head` | 1e-3 | |
| `--early-stopping-metric` | f1_macro | oppure `acc` |

---

## Fase 2 — AttentionForecaster

Estrae le feature con il modello di Fase 1 (cache HDF5) e addestra il forecaster.
Output: `$CKPT/${MODEL}_forecaster/forecaster_${MODEL}_${DATASET}_phase2_src02_tgt23.pt`

```bash
PRUNE_LAYER=2   # layer sorgente (default 2 per ViT-L/24 blocchi)

$PYTHON $SCRIPTS/train_forecaster.py \
    --model-name  $MODEL   \
    --dataset-name $DATASET \
    --base-data-folder $BASE_DATA \
    --adaptation lora \
    --classifier-ckpt $CKPT/${MODEL}_lora/best_model.pt \
    --forecaster-dir  $CKPT/${MODEL}_forecaster \
    --cache-dir       $CKPT \
    --layers-source   $PRUNE_LAYER \
    --wandb-project eaf
```

La cache HDF5 (`$CKPT/${DATASET}_${MODEL}_features.h5`) viene generata al primo run
e riutilizzata nei successivi. Per riestrarla, eliminarla prima di rilanciare.

Parametri modificabili:

| Flag | Default | Note |
|------|---------|------|
| `--epochs` | 30 | |
| `--layers-source` | 2 | layer sorgente; più valori → un forecaster per layer |
| `--layer-target` | ultimo blocco | default `n_blocks - 1` |
| `--hidden` | 256 | dimensione nascosta del forecaster |
| `--n-heads` | 4 | |
| `--n-layers` | 2 | |

Metrica di qualità: `val/rho` (Spearman) ≥ 0.5 indica un forecaster utile.

---

## Fase 3 — Fine-tuning con potatura

Carica i pesi di Fase 1, applica la potatura guidata dal forecaster, ri-addestra.
Output: `$CKPT/${MODEL}_pruned/results_${MODEL}_${DATASET}_prune${PRUNE_LAYER}_keep${K}.json`

```bash
PRUNE_LAYER=2
KEEP_RATIO=0.9   # 0.9 = 10% potato, 0.8 = 20%, 0.7 = 30%

FORECASTER=$CKPT/${MODEL}_forecaster/forecaster_${MODEL}_${DATASET}_phase2_src$(printf "%02d" $PRUNE_LAYER)_tgt23.pt

$PYTHON $SCRIPTS/finetune_pruned.py \
    --model-name  $MODEL   \
    --dataset-name $DATASET \
    --base-data-folder $BASE_DATA \
    --adaptation lora \
    --classifier-ckpt  $CKPT/${MODEL}_lora/best_model.pt \
    --forecaster-ckpt  $FORECASTER \
    --output-dir       $CKPT/${MODEL}_pruned \
    --prune-layer $PRUNE_LAYER \
    --keep-ratio  $KEEP_RATIO \
    --eval-baseline \
    --wandb-project eaf
```

Sweep sui keep-ratio:

```bash
for KEEP_RATIO in 0.9 0.8 0.7; do
    $PYTHON $SCRIPTS/finetune_pruned.py \
        --model-name $MODEL --dataset-name $DATASET \
        --base-data-folder $BASE_DATA \
        --adaptation lora \
        --classifier-ckpt $CKPT/${MODEL}_lora/best_model.pt \
        --forecaster-ckpt $FORECASTER \
        --output-dir      $CKPT/${MODEL}_pruned \
        --prune-layer $PRUNE_LAYER \
        --keep-ratio  $KEEP_RATIO \
        --eval-baseline \
        --wandb-project eaf
done
```

Parametri modificabili:

| Flag | Default | Note |
|------|---------|------|
| `--keep-ratio` | 0.1 | frazione di token mantenuti |
| `--prune-layer` | 2 | deve coincidere con `--layers-source` di Fase 2 |
| `--epochs` | 20 | |
| `--batch-size` | 16 | |
| `--eval-baseline` | off | aggiunge valutazione del modello non potato |

---

## Script di orchestrazione

Per lanciare le tre fasi in sequenza su più dataset con skip automatico
dei checkpoint già esistenti:

```bash
bash /home/oem/EAF/EntropyPruning/scripts/run_lora_experiments.sh
```

Log in: `/data/EAF_data/thunder/logs/lora_experiments/`

---

## Struttura checkpoint

```
/data/EAF_data/thunder/checkpoints/{dataset}/
├── {model}_lora/
│   └── best_model.pt                                        ← Fase 1
├── {model}_forecaster/
│   └── forecaster_{model}_{dataset}_phase2_src02_tgt23.pt  ← Fase 2
├── {dataset}_{model}_features.h5                           ← cache Fase 2
└── {model}_pruned/
    ├── best_{model}_{dataset}_prune2_keep90.pt              ← Fase 3
    └── results_{model}_{dataset}_prune2_keep90.json
```
