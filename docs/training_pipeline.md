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

### Cropr baseline

`finetune_pruned.py` can also run a Cropr-style pruning baseline instead of EAF:

```bash
python scripts/finetune_pruned.py \
    --pruning-method cropr \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --cropr-pruning-rate 8
```

Cropr does not load an `AttentionForecaster`; it trains auxiliary pruning heads
jointly with the classifier. Cropr does not use EAF's `--prune-layer`: it prunes
progressively after many transformer blocks. See [Alternative Pruning Methods](alternative_pruning_methods.md).

### EViT baseline

`finetune_pruned.py` can also run EViT token reorganization from Liang et al.:

```bash
python scripts/finetune_pruned.py \
    --pruning-method evit \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --evit-drop-loc 3,6,9 \
    --evit-base-keep-rate 0.7 \
    --evit-fuse-token
```

EViT does not load an `AttentionForecaster` and does not use `--prune-layer`.
It ranks patches with the current block's CLS attention, keeps the most
attentive patch tokens, and optionally fuses inattentive tokens before the MLP.
See [Alternative Pruning Methods](alternative_pruning_methods.md).

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

Reload EVIT checkpoints with:

```bash
python scripts/evaluate_pruned_checkpoints.py \
    --pruning-method evit \
    --model-name $MODEL \
    --dataset-name $DATASET \
    --base-data-folder $DATA \
    --evit-drop-loc 3,6,9 \
    --evit-base-keep-rates 0.7 0.6 0.5 \
    --evit-fuse-token \
    --output-csv results/eval_evit_${MODEL}_${DATASET}.csv
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
| `--backbone-ckpt` | nessuno | se impostato, carica questo `state_dict` sul backbone prima del probing (es. output di Step 3b) |
| `--backbone-tag` | `pretrained` | etichetta scritta nella colonna `backbone_variant` del CSV, per distinguere righe `pretrained` da righe `distilled` nello stesso file |

### Step 3b — Distillazione feature token-level (Approccio 3, dataset-agnostic)

Terza alternativa di Phase 3, oltre a `finetune_pruned.py` (Approccio 2 — LoRA + cross-entropy su un solo
dataset) e a Step 3 (Approccio 1 — backbone interamente frozen). Qui i blocchi **dopo** `--prune-layer`
ricevono adapter LoRA addestrati con una loss di **distillazione**: devono riprodurre gli hidden state finali
che lo stesso backbone, frozen e senza pruning, avrebbe prodotto (`DistilledPrunedBackbone` in
`src/models/pruned_classifier.py`). Il termine principale confronta, con cosine distance, i token patch
sopravvissuti allo student con i token teacher corrispondenti agli stessi indici originali. Due termini
ausiliari confrontano il CLS con cosine distance e la magnitudine del CLS con SmoothL1. Nessuna label,
nessuna testa di classificazione: il corpus è l'unione di più dataset THUNDER, esattamente come nello
Step 1/2a. Prima del training lo script costruisce (o riusa) una cache HDF5 per dataset con le immagini —
vedi la nota costo più sotto.

```bash
THUNDER_BASE_DATA_FOLDER=/path/to/thunder-tiles \
python scripts/distill_pruned.py \
    --model-name $MODEL \
    --base-data-folder /path/to/thunder-tiles/datasets \
    --cache-dir checkpoints/unsupervised \
    --prune-layer 2 \
    --keep-ratio 0.1 \
    --epochs 10 \
    --wandb-project eaf-distill
```

Per default usa il forecaster **universale** già addestrato allo Step 2a (stesso `--prune-layer` come
source layer). Al termine produce un singolo backbone distillato, riutilizzabile su qualunque dataset.

**Output:** `checkpoints/unsupervised/{model}_distilled/distilled_{model}_prune{L}_keep{k}.pt`

Argomenti principali:

| Flag | Default | Note |
|---|---|---|
| `--datasets` | i 16 dataset THUNDER | corpus di immagini per costruire la cache di distillazione (nessuna label usata) |
| `--forecaster-ckpt` | forecaster universale in `--cache-dir` | deve corrispondere a `--prune-layer` come source layer |
| `--token-weight`, `--cls-weight`, `--mag-weight` | `1.0`, `0.5`, `0.1` | pesi di token cosine, CLS cosine e SmoothL1 sulle norme CLS |
| `--keep-ratio-min` | unset | se impostato, campiona a ogni step un keep ratio uniforme in `[keep-ratio-min, keep-ratio]`; la validazione resta a `--keep-ratio` |
| `--retrieval-k` | `5` | k per la metrica di retrieval consistency (deve essere < `--batch-size`) |
| `--lora-r`, `--lora-alpha` | `8`, `32` | LoRA solo sui blocchi dopo `--prune-layer` |
| `--epochs` | `10` | |

**Eval e metriche loggate su W&B.** Train/val/test sono gli split standard THUNDER (nessuna eval
"pre-training": si parte direttamente dalla prima epoca). Ogni epoca valuta sul **val set** (selezione
checkpoint + early stopping su `val/loss`); a fine training il checkpoint migliore viene ri-valutato una
sola volta sul **test set**. In entrambi i casi (`val/*` e, a fine corsa, `test/*`) vengono loggate:

- **Cosine similarity CLS** student↔teacher (`cls_cosine_sim`)
- **Cosine similarity token-level**, media sui token patch sopravvissuti al pruning (`token_cosine_sim`)
- **Errore relativo di magnitudine CLS**, `|‖cls_student‖ − ‖cls_teacher‖| / ‖cls_teacher‖` (`cls_mag_rel_error`)
- **CKA lineare** tra le feature CLS student e teacher, accumulata sull'intero split (non per-batch) (`cka_linear`)
- **k-NN retrieval consistency**, recall@k in-batch: per ogni campione, frazione dei top-k vicini (cosine,
  self escluso) calcolati nello spazio teacher che sono anche tra i top-k nello spazio student
  (`retrieval_recall_at_k`). Il pool di retrieval è il batch stesso (serve `--batch-size` > `--retrieval-k`),
  quindi è una proxy economica per-step, non una retrieval valutata sull'intero corpus.

Poi, per il linear probe per-dataset sul backbone distillato (chiudendo il loop dell'Approccio 3):

```bash
python scripts/linear_probe_pruned_eaf.py \
    --model-name $MODEL \
    --base-data-folder /path/to/thunder-tiles/datasets \
    --cache-dir checkpoints/unsupervised \
    --eaf-types universal \
    --keep-ratios 0.1 \
    --backbone-ckpt checkpoints/unsupervised/${MODEL}_distilled/distilled_${MODEL}_prune2_keep10.pt \
    --backbone-tag distilled
```

Le righe finiscono nello stesso `results/linear_probe_pruned/{model}.csv` dello Step 3, con
`backbone_variant=distilled` invece di `pretrained` — permette il confronto diretto Approccio 1 vs
Approccio 3 a parità di forecaster/keep_ratio/dataset.

> **Nota costo:** i blocchi fino a `--prune-layer` sono frozen e identici tra teacher e student (nessun
> LoRA lì), e il teacher è interamente frozen — quindi il loro output è costante per tutta la run. Lo
> script costruisce/riusa automaticamente una cache HDF5 per dataset (`{dataset}_{model}_distill_prune{L}.h5`
> in `--cache-dir`, vedi `src/collection/distill_cache.py` e, per pre-costruirla a mano,
> `scripts/build_distill_cache.py`) contenente l'output grezzo di `blocks[--prune-layer]` più il CLS/patch
> token finale del teacher. Il training legge solo questi tensori: niente immagini, niente forward del
> backbone congelato o del teacher — ogni step esegue solo i blocchi LoRA dopo `--prune-layer`, sulla
> sequenza già potata. `--cache-batch-size`/`--cache-num-workers` controllano solo la passata di estrazione
> una tantum; per backbone molto grandi, riduci `--batch-size` (training) se la memoria è il collo di
> bottiglia.

### Step 4 — Confronto per-dataset vs universale (spider plot + CSV)

Script `compare_eaf_spider.py`: confronta la Spearman ρ dei forecaster per-dataset (re-valutati dai checkpoint) con quella del modello universale (letta dal JSON prodotto in Step 2a). Produce due file in `results/ablations/per_vs_universal/{model_name}/`.

```bash
# Solo universale (da JSON)
python scripts/compare_eaf_spider.py \
    --model-name $MODEL \
    --cache-dir checkpoints/unsupervised \
    --universal-json \
        checkpoints/unsupervised/${MODEL}_forecaster/results_forecaster_${MODEL}_src02_attn23_universal.json

# Confronto completo (re-valuta per-dataset + carica universale)
python scripts/compare_eaf_spider.py \
    --model-name $MODEL \
    --cache-dir checkpoints/unsupervised \
    --per-dataset-dir checkpoints/unsupervised/per_dataset \
    --layers-source 2 \
    --layer-target 23 \
    --universal-json \
        checkpoints/unsupervised/${MODEL}_forecaster/results_forecaster_${MODEL}_src02_attn23_universal.json \
        checkpoints/unsupervised/${MODEL}_forecaster_h512/results_forecaster_${MODEL}_src02_attn23_universal.json
```

**Output** (`results/ablations/per_vs_universal/{model_name}/`):

| File | Descrizione |
|---|---|
| `rho_spider.png` | Grafico radar — tutte le serie + baseline token-norm |
| `rho_table.csv` | Tabella numerica dataset × serie (+ avg) |

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
├── per_dataset/
│   └── {dataset}/
│       └── forecaster_src{L:02d}_attn{T:02d}.pt                ← per-dataset (Step 2b)
└── {model}_distilled/
    ├── distilled_{model}_prune{L}_keep{k}.pt                   ← backbone distillato (Step 3b)
    ├── lora_{model}_prune{L}_keep{k}.pt                        ← checkpoint LoRA intermedio (resume)
    └── results_{model}_prune{L}_keep{k}.json

results/linear_probe_pruned/
└── {model_name}.csv                                  ← Step 3 (pretrained) + Step 3b (distilled)

results/ablations/per_vs_universal/
└── {model_name}/
    ├── rho_spider.png                                          ← Step 4
    └── rho_table.csv                                          ← Step 4
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
