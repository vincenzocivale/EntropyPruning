# Training Pipeline

Complete walkthrough of the three-phase EAF training process.

---

## Prerequisites

```bash
source .venv/bin/activate

# Thunder data must be downloaded:
thunder download crc --base-data-folder /path/to/thunder/data
```

Set a convenience variable:
```bash
DATA=/path/to/thunder/data
MODEL=uni          # or hoptimus0, virchow, dinov2base, etc.
DATASET=crc        # or break_his, mhist, patch_camelyon, etc.
```

---

## Phase 1 — Base Classifier

Fine-tune a classification head on top of the backbone using the chosen adaptation strategy.

```bash
python scripts/train_classifier.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --adaptation lora \
    --lora-r 8 \
    --lora-alpha 32 \
    --epochs 20 \
    --batch-size 8 \
    --lr-backbone 1e-5 \
    --lr-head 1e-3 \
    --weight-decay 0.01 \
    --warmup-steps 100 \
    --label-smoothing 0.1 \
    --early-stopping-patience 3
```

**Early stopping:** Ferma il training se la metrica scelta non migliora per N epoche consecutive (default: 3).
Imposta `--early-stopping-patience 0` per disabilitarlo.

**Output:** `checkpoints/$DATASET/${MODEL}_lora/best_model.pt`

The script prints a classification report on the test split at the end.

### Choosing `--prune-layer` early

The source layer used for feature extraction in Phase 2 should be decided now.
For a ViT-L (24 blocks), layer 2 is a good starting point.
For H-optimus (40 blocks), layer 5–8.

---

## Phase 2 — AttentionForecaster

### Step 2a — Feature extraction

The first time you run Phase 2, `collect_and_save_dataset` is called automatically.
It runs the Phase 1 model in eval mode and caches, for each image:
- Patch embeddings at the source layer(s): shape `(P, D)`, float16
- CLS attention weights at the target layer: shape `(P,)`, float16

```bash
python scripts/train_forecaster.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --adaptation lora \            # must match Phase 1
    --layers-source 2 \            # which layers to extract embeddings from
    --layer-target 23 \            # which layer to predict attention for (default: last)
    --epochs 30 \
    --hidden 256 \
    --n-heads 4 \
    --n-layers 2 \
    --lr 1e-4 \
    --wandb-project eaf-forecaster
```

**Feature cache:** `checkpoints/$DATASET/${DATASET}_${MODEL}_features.h5`
**Output:** `checkpoints/$DATASET/${MODEL}_forecaster/forecaster_src02_tgt23.pt`

If the cache already exists (from a previous run), feature extraction is skipped.

### Step 2b — Multiple source layers

You can extract from multiple layers and train one forecaster per layer in a single run:

```bash
python scripts/train_forecaster.py \
    ... \
    --layers-source 2 4 8 12
```

This trains four forecasters: `forecaster_src02_tgt23.pt`, `forecaster_src04_tgt23.pt`, etc.

### Monitoring forecaster quality

Watch `val/rho` in W&B (Spearman rank correlation against ground-truth attention).
A good forecaster typically reaches `rho > 0.5` on the validation set.
`test/delta_vs_norm` shows the improvement over a simple token-norm baseline.

---

## Phase 3 — Pruned Fine-tuning

Load Phase 1 weights into `GenericLoRAWithForecasterPruning`, freeze the forecaster, and fine-tune.

```bash
python scripts/finetune_pruned.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --prune-layer 2 \
    --keep-ratio 0.1 \            # keep 10% of patches
    --epochs 20 \
    --batch-size 16 \
    --lr-backbone 1e-4 \
    --lr-head 1e-3 \
    --wandb-project eaf-pruning \
    --eval-baseline \             # also evaluate unpruned model for comparison
    --early-stopping-patience 3
```

**Early stopping:** Ferma il training se la metrica non migliora per N epoche consecutive (default: 3).

**Output:** `checkpoints/$DATASET/${MODEL}_pruned/best_${MODEL}_prune2_keep10.pt`

### `--keep-ratio` sweep

```bash
for RATIO in 0.05 0.1 0.2 0.3 0.5; do
    python scripts/finetune_pruned.py \
        --model-name $MODEL --dataset-name $DATASET \
        --base-data-folder $DATA \
        --prune-layer 2 --keep-ratio $RATIO \
        --epochs 20
done
```

---

## Evaluation

Reload any pruned checkpoint and evaluate on the test set:

```bash
python scripts/evaluate_pruned_checkpoints.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --prune-layers 2 \
    --keep-ratios 0.05 0.1 0.2 0.3 0.5 \
    --output-csv results/eval_${MODEL}_${DATASET}.csv
```

The CSV contains accuracy, F1-macro, AUROC, TAR@FAR, ms/img, and GFLOPs for each configuration.

---

## Checkpoint naming conventions

```
checkpoints/
└── {dataset}/
    ├── {model}_{adaptation}/
    │   └── best_model.pt                           ← Phase 1
    ├── {model}_forecaster/
    │   └── forecaster_src{L:02d}_tgt{T:02d}.pt    ← Phase 2
    ├── {dataset}_{model}_features.h5               ← Phase 2 cache
    └── {model}_pruned/
        └── best_{model}_prune{L}_keep{k}.pt        ← Phase 3
```

---

## Unsupervised EAF (nessun Phase 1 richiesto)

Alternativa alla pipeline supervisionata: addestra `AttentionForecaster` usando come target
l'attenzione CLS→patch dell'**ultimo blocco** del modello fondazionale **frozen** (nessun
fine-tuning in Phase 1). Disponibile in due varianti:

- **Universale** (`train_forecaster_unsupervised.py`): un unico modello addestrato sul corpus
  concatenato di tutti i dataset disponibili.
- **Per-dataset** (`train_per_dataset_eaf.py`): un modello separato per ciascun dataset,
  addestrato indipendentemente (utile come baseline di confronto con il modello universale).

### Corpus supportato

16 dataset THUNDER di classificazione (esclusi i 4 di sola segmentazione):

```
bach bracs break_his ccrcc crc esca mhist patch_camelyon
spider_breast spider_colorectal spider_skin spider_thorax
tcga_crc_msi tcga_tils tcga_uniform wilds
```

Dataset mancanti (no `data_splits/{name}.json`) vengono saltati automaticamente con un avviso.
Per scaricarli:

```bash
THUNDER_BASE_DATA_FOLDER=/path/to/thunder-tiles \
    thunder download <dataset> --make-splits --base-data-folder /path/to/thunder-tiles/datasets
```

> **Variabile d'ambiente**: `THUNDER_BASE_DATA_FOLDER` deve puntare alla **root** Thunder
> (quella che contiene `pretrained_ckpts/`), mentre `--base-data-folder` punta alla
> sottocartella `datasets/` (che contiene `data_splits/`). Sono percorsi distinti.

### Step 1 — Build delle cache HDF5 per-dataset

```bash
THUNDER_BASE_DATA_FOLDER=/path/to/thunder-tiles \
python scripts/build_unsupervised_cache.py \
    --model-name $MODEL \
    --base-data-folder /path/to/thunder-tiles/datasets \
    --layers-source 2
```

Per ogni dataset estrae `emb_layer{L}` e `attn_layer{last}` dal backbone frozen e scrive
`checkpoints/unsupervised/{dataset}_{model}_attn_features.h5`.
Il sweep è **resumabile**: dataset con cache già valida vengono saltati.

Argomenti principali:

| Flag | Default | Note |
|---|---|---|
| `--layers-source` | `2` | uno o più layer sorgente (es. `2 4 8`); gli embedding vengono concatenati |
| `--layer-target` | ultimo blocco | layer di cui predire l'attenzione CLS |
| `--batch-size` | 64 | |
| `--cache-dir` | `checkpoints/unsupervised` | |

### Step 2a — Modello universale (un modello su tutti i dataset)

```bash
THUNDER_BASE_DATA_FOLDER=/path/to/thunder-tiles \
python scripts/train_forecaster_unsupervised.py \
    --model-name $MODEL \
    --base-data-folder /path/to/thunder-tiles/datasets \
    --layers-source 2 \
    --epochs 30 \
    --hidden 512 \
    --forecaster-dir checkpoints/unsupervised/${MODEL}_forecaster_h512 \
    --wandb-project eaf-forecaster
```

Riutilizza le cache di Step 1, concatena i dataset via `MultiH5ForecastDataset`, addestra
un unico `AttentionForecaster`. La valutazione avviene **solo a fine epoca**. Al termine
produce un report per-dataset di Spearman `rho`.

**Output:**
`checkpoints/unsupervised/{model}_forecaster/forecaster_{model}_{src_tag}_attn{T:02d}_universal.pt`

dove `src_tag = "src" + "+".join(f"{ls:02d}" for ls in layers_source)`, es. `src02` oppure `src01+02+03+04+05`.

Argomenti principali:

| Flag | Default | Note |
|---|---|---|
| `--hidden` | 256 | dimensione nascosta; 512 aumenta la capacità del modello |
| `--n-layers` | 2 | numero di layer transformer |
| `--epochs` | 30 | |
| `--forecaster-dir` | `checkpoints/unsupervised/{model}_forecaster` | separare h256 e h512 con percorsi distinti |
| `--wandb-project` | `attention-forecaster` | |

### Step 2b — Modelli per-dataset (un modello per dataset, baseline di confronto)

```bash
THUNDER_BASE_DATA_FOLDER=/path/to/thunder-tiles \
python scripts/train_per_dataset_eaf.py \
    --cache-dir checkpoints/unsupervised \
    --checkpoint-dir checkpoints/unsupervised/per_dataset \
    --epochs 30 \
    --out-csv logs/per_dataset_eaf_rho.csv
```

Addestra un modello per ciascun dataset in sequenza, salva il miglior checkpoint e aggiunge
una riga al CSV non appena ogni dataset completa. Checkpoint organizzati in:
`checkpoints/unsupervised/per_dataset/{dataset}/forecaster_src{L:02d}_attn{T:02d}.pt`

### Step 3 — Linear probe con EAF Pruning (esperimento)

Valuta quanto è utile il pruning EAF anche senza fine-tuning del backbone: backbone e forecaster
sono completamente frozen, si addestra solo una testa lineare (`nn.Linear`).

```bash
THUNDER_BASE_DATA_FOLDER=/path/to/thunder-tiles \
python scripts/linear_probe_pruned_eaf.py \
    --model-name $MODEL \
    --base-data-folder /path/to/thunder-tiles/datasets \
    --cache-dir checkpoints/unsupervised \
    --eaf-types per_dataset universal \
    --layers-source 2 \
    --keep-ratios 0.1 0.25 0.5 0.75 \
    --epochs 20 \
    --results-dir results/linear_probe_pruned
```

Lo script:
1. Scopre automaticamente i dataset dai file `.h5` in `--cache-dir` (o usa `--datasets`).
2. Per `eaf_type=universal` carica il forecaster una sola volta per tutti i dataset.
3. Per `eaf_type=per_dataset` carica il forecaster specifico per ogni dataset.
4. È **resumabile**: righe già presenti nel CSV vengono saltate.

**Output:** `results/linear_probe_pruned/{model_name}.csv`

Colonne del CSV:

| Colonna | Descrizione |
|---|---|
| `model_name` | Encoder usato |
| `dataset` | Nome del dataset |
| `eaf_type` | `per_dataset` o `universal` |
| `layers_source` | Blocchi sorgente (es. `2` o `1+2+3+4+5`) |
| `prune_layer` | Blocco dove viene applicato il pruning |
| `keep_ratio` | Frazione di patch mantenute |
| `n_classes`, `n_train`, `n_val`, `n_test` | Statistiche del dataset |
| `best_epoch`, `best_val_acc`, `best_val_f1` | Risultato migliore su validation |
| `test_acc` | Accuracy sul test |
| `test_f1_macro` | F1 macro sul test |
| `test_auroc` | AUROC (OvR macro) sul test |
| `test_tar_at_far` | TAR@FAR (default FAR=1e-4) |

Argomenti principali:

| Flag | Default | Note |
|---|---|---|
| `--eaf-types` | `per_dataset universal` | uno o entrambi |
| `--layers-source` | `2` | deve coincidere con le cache disponibili |
| `--keep-ratios` | `0.25 0.5 0.75` | sweep multipli in un solo run |
| `--per-dataset-dir` | `{cache-dir}/per_dataset` | dove cercare i forecaster per-dataset |
| `--forecaster-n-heads` | `4` | deve coincidere con il training del forecaster |
| `--epochs` | `20` | epoche per la testa lineare |

### Step 4 — Generare lo spider plot per-dataset

```bash
python scripts/eval_spider_plot.py \
    --checkpoint checkpoints/unsupervised/${MODEL}_forecaster_h512/forecaster_src02_attn23_universal.pt \
    --cache-dir checkpoints/unsupervised \
    --hidden 512 \
    --out logs/spider_rho_h512.png
```

Carica il checkpoint, valuta sul test set, genera un grafico radar con il confronto
forecaster vs. baseline token-norm per ciascun dataset.

### Step 5 — Hookup con Phase 3

```bash
python scripts/finetune_pruned.py \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder /path/to/thunder-tiles/datasets \
    --prune-layer 2 \
    --keep-ratio 0.1 \
    --forecaster-ckpt checkpoints/unsupervised/${MODEL}_forecaster_h512/forecaster_src02_attn23_universal.pt
```

### Struttura checkpoint non-supervisionato

```
checkpoints/unsupervised/
├── {dataset}_{model}_attn_features.h5                          ← cache HDF5 (Step 1)
├── {model}_forecaster/
│   ├── forecaster_{model}_{src_tag}_attn{T:02d}_universal.pt   ← universale (Step 2a)
│   └── results_forecaster_{model}_{src_tag}_attn{T:02d}_universal.json
└── per_dataset/
    └── {dataset}/
        └── forecaster_src{L:02d}_attn{T:02d}.pt                ← per-dataset (Step 2b)

results/linear_probe_pruned/
└── {model_name}.csv                                            ← Step 3
```

### Monitoraggio training (nohup + wandb)

Per esecuzioni SSH-detached, avviare con nohup e disown:

```bash
nohup bash -c 'THUNDER_BASE_DATA_FOLDER=... conda run --no-capture-output -n eaf_env \
    python scripts/train_forecaster_unsupervised.py ... \
    > logs/train_forecaster_unsupervised_uni_h512.log 2>&1' > /dev/null 2>&1 & disown
```

Monitoraggio live:
```bash
# Metriche di validazione (aggiornate solo a fine epoca — stdout bufferizzato)
tail -f wandb/run-*/files/output.log | grep -E "new best|Ep [0-9]"

# Log completo
tail -f logs/train_forecaster_unsupervised_uni_h512.log
```

---

## Tips

**Out of memory in Phase 1** — reduce `--batch-size` or switch to `--adaptation linear_probing` (no backbone gradients).

**Phase 2 cache is stale** — delete the `.h5` file; it will be regenerated on the next run.

**Phase 3 not improving over baseline** — try a lower `--keep-ratio` (less aggressive pruning), a different `--prune-layer`, or more `--epochs`.

**Forecaster `val/rho` stays near 0** — the source layer may be too early (features not yet discriminative). Try a later source layer.
