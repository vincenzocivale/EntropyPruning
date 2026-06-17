# Training Performance Bottlenecks

Note operative per eseguire esperimenti EAF piu velocemente senza monopolizzare
CPU e RAM su un server condiviso.

## Bottleneck risolvibili trovati

### 1. `train_forecaster.py` ignorava `--batch-size` e `--num-workers`

La fase di training dell'`AttentionForecaster` da cache HDF5 usava sempre:

```python
batch_size=128, num_workers=4, persistent_workers=True
```

anche quando da CLI venivano passati valori piu conservativi. Questo rendeva
difficile ridurre pressione su CPU, RAM e file descriptor in un server condiviso.

Stato: corretto. Ora `scripts/train_forecaster.py` usa i valori CLI:

```bash
python scripts/train_forecaster.py \
    --model-name uni \
    --dataset-name crc \
    --base-data-folder $DATA \
    --batch-size 64 \
    --num-workers 1
```

`persistent_workers` resta attivo solo se `--num-workers > 0`.

### 2. Pruning con overhead Python nel forward

`GenericLoRAWithForecasterPruning` calcolava `topk` due volte e ricostruiva i
token con una list comprehension su ogni elemento del batch. Questo aggiungeva
overhead nel path di Phase 3, soprattutto con batch piu grandi.

Stato: corretto. Il forward usa un solo `topk` e `torch.gather`, mantenendo la
stessa selezione di token.

## Bottleneck presenti ma configurabili

### DataLoader immagini Thunder

`build_thunder_loaders` usa `image_pre_loading=False`, quindi non carica tutto
il dataset in RAM. Il costo resta pero I/O + transform CPU per batch.

Impostazioni consigliate su server condiviso:

```bash
--num-workers 1 --batch-size 8
```

Se la GPU resta molto scarica e la RAM e libera, salire a:

```bash
--num-workers 2
```

Evitare `--num-workers 4+` durante sweep lunghi o multi-run paralleli: ogni run
mantiene worker persistenti e prefetch di batch.

### Cache HDF5

La cache riduce il costo delle fasi successive, ma la prima estrazione resta il
passo piu pesante: legge immagini, fa forward completo del backbone e scrive
embedding/attenzioni in HDF5.

Per debug o confronto rapido:

```bash
python scripts/build_unsupervised_cache.py \
    --model-name uni \
    --base-data-folder $DATA \
    --datasets crc mhist \
    --max-samples-per-split 512 \
    --batch-size 32 \
    --num-workers 1
```

Per run reali, conviene costruire le cache una volta e poi riusarle. Gli script
controllano gia se la cache e valida prima di ricostruirla.

### Multi-dataset forecaster

`train_forecaster_unsupervised.py` concatena molte cache HDF5. Questo e piu
leggero di rileggere immagini, ma puo aprire piu file HDF5 e usare worker
persistenti.

Impostazione conservativa:

```bash
python scripts/train_forecaster_unsupervised.py \
    --model-name uni \
    --base-data-folder $DATA \
    --datasets crc mhist patch_camelyon \
    --cache-num-workers 1 \
    --num-workers 1 \
    --cache-batch-size 32 \
    --batch-size 64
```

## Priorita pratica

1. Fare smoke run con `--max-samples-per-split` prima di lanciare sweep completi.
2. Usare `--num-workers 1` come default su server condiviso; aumentare solo se la
   GPU e chiaramente sotto-utilizzata.
3. Tenere `batch-size` moderato e aumentare solo quando la memoria GPU lo permette.
4. Costruire cache HDF5 una volta per combinazione dataset/modello/layer e poi
   riusarle.
5. Preferire `linear_probing` o LoRA per esperimenti esplorativi; full fine-tuning
   e il modo piu costoso in memoria e tempo.
