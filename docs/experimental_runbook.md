# Runbook di esecuzione e implementazione degli esperimenti

Aggiornato il 16 settembre 2026 sul codice `aa09c9a`. Questo documento distingue
comandi esistenti da API proposte: **non esiste ancora un comando che esegua
l'intero benchmark del paper**. Non avviare uno sweep copiando i template prima
di risolvere gli input e superare i gate indicati.

Riferimenti: [roadmap/stato](experimental_roadmap.md),
[protocolli](experimental_protocols.md), [layout](data_layout.md).

## 1. Ambiente e prerequisiti

Eseguire dalla radice della repo. `environment.yml` è un file Conda YAML:

```bash
conda env create --name trident --file environment.yml
conda activate trident
export EAF_WSI_ROOT=/percorso/assoluto/del/runtime/WSI
export WANDB_MODE=disabled
```

Creare l'ambiente solo se assente e sostituire il percorso dimostrativo.
`WANDB_MODE=disabled` evita logging remoto nei template WSI, che non espongono
`--wandb-mode`. Registrare l'ambiente effettivo e le versioni; non installare
silenziosamente dipendenze incompatibili per un singolo backbone.

Su questo host, durante l'aggiornamento della documentazione, non era presente
un ambiente chiamato `trident`; l'audit è stato eseguito con l'ambiente esistente
`eaf-wsi`. Ciò verifica l'audit, non la compatibilità di tutti i training/FM.
La posizione host del runtime è documentata in [continuity](continuity.md).

Prima di dichiarare E0 per un run controllare:

- Pesi accessibili, revisione e termini registrati; nessun token nei file di
  configurazione, log o comandi condivisi. Gli accessi non sono stati verificati
  per tutti i modelli nella sola scrittura di questa documentazione.
- GPU, VRAM, dtype supportato, spazio libero e budget; nessuna prenotazione o
  training è implicita nella creazione di questi documenti.
- Manifest versionato, path risolvibili, pazienti/studi disgiunti e label valide.
- Cache complete con coordinate nello stesso ordine e preprocessing identico.
- Checkpoint caricabile con forecaster, layer, pooling e input feature corretti.
- Cartelle output nuove/versionate sotto il runtime root, mai output nella repo.

## 2. Inventario e validazione già disponibili

Questi comandi esistono. L'audit legge metadati e riscrive il catalogo locale;
non valida tutti gli array o i checkpoint. `layout` può creare directory.

```bash
python scripts/eaf.py experiments audit --data-root "$EAF_WSI_ROOT"
jq '.counts' "$EAF_WSI_ROOT/results/experiment_catalog/catalog.json"
jq '.runs[] | {id, name, status, missing, checkpoints, summaries}' \
  "$EAF_WSI_ROOT/results/experiment_catalog/catalog.json"
python scripts/eaf.py cache tile --help
python scripts/eaf.py cache index-tile --help
```

Per validare una singola cache tile, impostare `EAF_CACHE_SLIDE` a una directory
`.npyd` o a un input HDF5 legacy, non alla directory di un intero dataset:

```bash
: "${EAF_CACHE_SLIDE:?Impostare il percorso di una cache slide}"
python scripts/eaf.py cache validate "$EAF_CACHE_SLIDE" --kind tile_eaf
```

Non applicare indiscriminatamente `--kind wsi_eaf` agli output di
`wsi_eaf_infer_wsi_fm.py`: gli output `eaf.wsi.fm_output.v1` e i contratti
cache hanno schemi distinti. Validare lo schema effettivo con il reader
appropriato; la validazione uniforme degli adapter fa parte di I02/I04.

### Manifest pretraining e HEST

`data build-strict` esiste, ma per default comprende HISTAI, GTEx e HEST.
Per il paper che usa HEST come test biologico, usare fonti HISTAI/GTEx dopo
l'audit dei rispettivi pazienti. Il comando seguente **riscrive il manifest
strict di destinazione**: archiviare prima la versione precedente e il relativo
hash con il meccanismo di versionamento dei manifest; non lanciarlo su un
manifest usato da run attivi senza conservarne la versione.

```bash
python scripts/eaf.py data build-strict \
  --data-root "$EAF_WSI_ROOT" --source histai --source gtex --seed 17
```

Il guard TCGA del builder non sostituisce l'audit dell'intera banca downstream.
L'esclusione HEST dal nuovo manifest non decontamina checkpoint già addestrati.
Prima di avviare i template seguenti definire esplicitamente:

| Variabile | Contenuto richiesto |
| --- | --- |
| `EAF_PRETRAIN_MANIFEST` | CSV canonico versionato con split e coordinate, escluso downstream |
| `EAF_TILE_CACHE_ROOT` | Root di UNA variante/versione full tile, con sottocartelle coorte |
| `EAF_TILE_CACHE_INDEX` | Nuovo CSV indice per quello stesso manifest/versione |
| `EAF_RUN_NAME` | Nome univoco; includere modello, layer, retention se applicabile e seed |
| `EAF_FORECASTER_CKPT` | Checkpoint selezionato del forecaster tile; non un checkpoint WSI |
| `EAF_TILE_STUDENT_CKPT` | Adapter della distillazione Tile-EAF |
| `EAF_WSI_CACHE_ROOT` | Root coorti dei target/hidden TITAN full, allineata alla root tile |
| `EAF_WSI_FORECASTER_CKPT` | Forecaster TITAN hidden con layer noto |
| `EAF_WSI_STUDENT_CKPT` | Checkpoint di distillazione TITAN pruned |

Il manifest canonico è definito da `src/data/wsi/manifest.py`: almeno
`slide_id,case_id,source,cohort,raw_path,coords_path,split`, con gli altri campi
canonici dove disponibili. Gli ID slide devono essere globalmente univoci
nell'indice tile; il caso locale può richiedere namespace di coorte.
Congelare il mapping tra namespace e paziente reale, anche tra dataset diversi.

## 3. Template del percorso CONCH Tile-EAF esistente

I comandi sono template verificati rispetto ai parser, non una selezione di
iperparametri già approvata dai risultati. Il caso seguente usa source layer 0,
retention 20% e seed 42; modificare il nome run a ogni fase. Nel registro THUNDER
storico `titan` carica CONCH v1.5 per il tile training: verificare modello e
revisioni effettivi, senza confonderlo con il TITAN slide encoder.

### Cache full e indice

Riutilizzare una cache esistente validata quando compatibile. Creare nuovi output
solo per la variante mancante; non usare `--overwrite` nel normale workflow.

```bash
: "${EAF_PRETRAIN_MANIFEST:?Impostare il manifest pretraining validato}"
: "${EAF_TILE_CACHE_ROOT:?Impostare una root cache versionata}"
: "${EAF_TILE_CACHE_INDEX:?Impostare il nuovo indice cache}"
python scripts/eaf.py cache tile \
  --data-root "$EAF_WSI_ROOT" --manifest "$EAF_PRETRAIN_MANIFEST" \
  --output-dir "$EAF_TILE_CACHE_ROOT" --encoder conch_v15 \
  --dataset eaf_wsi_pretrain_strict_v1 --early-layer 0 \
  --input-mpp 0.5 --input-mag 20 --patch-size 512 --stride 512
python scripts/eaf.py cache index-tile \
  --data-root "$EAF_WSI_ROOT" --slides "$EAF_PRETRAIN_MANIFEST" \
  --cache-root "$EAF_TILE_CACHE_ROOT" --output "$EAF_TILE_CACHE_INDEX"
```

`cache tile` scrive nella directory indicata: per ottenere la gerarchia per
coorte richiesta dai reader WSI, eseguirlo su manifest per coorte con
`--output-dir <root>/<cohort>`, poi costruire l'indice sulla root comune.
Non supporre che la CLI raggruppi automaticamente le coorti.
MPP e dimensioni sopra sono nominali CONCH; controllare la risoluzione effettiva
dopo resize. Non riutilizzarle automaticamente per altri encoder.

### Forecaster e distillazione

Passare sempre `--target-cache-index`: il fallback senza indice presente nel
codice può eseguire il teacher completo online e non è il workflow del paper.
Le early feature sono ricalcolate online; le quantità finali del teacher sono
lette dalla cache. I pixel raw restano necessari durante il training.

```bash
: "${EAF_RUN_NAME:?Impostare un nome univoco per il forecaster}"
python scripts/training/train_wsi_tile_eaf_online.py \
  --model-name titan --manifest "$EAF_PRETRAIN_MANIFEST" \
  --data-root "$EAF_WSI_ROOT" --target-cache-index "$EAF_TILE_CACHE_INDEX" \
  --source-layer 0 --seed 42 --split-seed 17 \
  --exclude-cohort --run-name "$EAF_RUN_NAME" --wandb-mode disabled \
  --output-dir "$EAF_WSI_ROOT/checkpoints/tile_eaf/conch_v15/$EAF_RUN_NAME"
```

`--exclude-cohort` senza valori disabilita le esclusioni implicite HISTAI-mixed
e HISTAI-skin-b2: il manifest congelato determina così il corpus. Per riprodurre
run storici usare invece le esclusioni registrate. Per il profiling aggiungere
`--profile-json` con percorso sotto `logs/` o `results/`, dopo aver predisposto
la directory; usare lo stesso campione/config per confrontare ottimizzazioni.

Selezionare il checkpoint dal summary, poi impostare un **nuovo** nome run:

```bash
: "${EAF_FORECASTER_CKPT:?Impostare il checkpoint tile del layer 0}"
: "${EAF_RUN_NAME:?Impostare un nome univoco per la distillazione}"
python scripts/training/finetune_wsi_tile_encoder_pruned_online.py \
  --model-name titan --manifest "$EAF_PRETRAIN_MANIFEST" \
  --data-root "$EAF_WSI_ROOT" --target-cache-index "$EAF_TILE_CACHE_INDEX" \
  --forecaster-ckpt "$EAF_FORECASTER_CKPT" --prune-layer 0 --keep-ratio 0.20 \
  --seed 42 --split-seed 17 --exclude-cohort --wandb-mode disabled \
  --run-name "$EAF_RUN_NAME" \
  --output-dir "$EAF_WSI_ROOT/checkpoints/pruned_finetuned/conch_v15/$EAF_RUN_NAME"
```

Non alterare il FOV con `--tile-size-at-target-mag` rispetto alla cache. I target
devono corrispondere agli stessi crop, nello stesso ordine. Non lanciare i cinque
encoder insieme prima di aver misurato risorse e validato l'adapter di ciascuno.

### Estrazione compressa per il downstream

Il manifest seguente è downstream e **non entra nel training EAF**. Definire
`EAF_DOWNSTREAM_MANIFEST` per una coorte e `EAF_PRUNED_TILE_DIR` come directory
di output nuova `<root_variante>/<cohort>`.

```bash
: "${EAF_TILE_STUDENT_CKPT:?Impostare l'adapter Tile-EAF}"
: "${EAF_DOWNSTREAM_MANIFEST:?Impostare il manifest downstream}"
: "${EAF_PRUNED_TILE_DIR:?Impostare una directory variante compressa}"
python scripts/eaf.py cache tile \
  --data-root "$EAF_WSI_ROOT" --manifest "$EAF_DOWNSTREAM_MANIFEST" \
  --output-dir "$EAF_PRUNED_TILE_DIR" --dataset paper_downstream \
  --pruned-adapter-ckpt "$EAF_TILE_STUDENT_CKPT" \
  --input-mpp 0.5 --input-mag 20 --patch-size 512 --stride 512
```

I parametri forecaster/layer/retention sono risolti dal checkpoint quando
presenti. Non sovrascriverli con default diversi senza registrare la variante.

## 4. Template TITAN WSI-EAF esistente

### Estrazione target full e hidden

`EAF_TILE_COHORT_DIR` è una sola directory di cache full tile;
`EAF_WSI_COHORT_DIR` è il relativo output TITAN versionato. Per training usare
solo coorti di pretraining disgiunte; per valutazione creare cache separate.

```bash
: "${EAF_TILE_COHORT_DIR:?Impostare la cache tile di una coorte}"
: "${EAF_WSI_COHORT_DIR:?Impostare l'output TITAN della stessa coorte}"
python scripts/features/wsi_eaf_infer_wsi_fm.py \
  --tile-cache-dir "$EAF_TILE_COHORT_DIR" --output-dir "$EAF_WSI_COHORT_DIR" \
  --model titan --titan-hidden-layer 0 --device cuda
```

Verificare scale coordinate/patch-size-level0 per la coorte e presenza
`auxiliary/hidden_layer_000` e `attention/global_to_tiles_mass_share`. Non
richiedere matrici complete di attenzione sulle slide grandi senza una stima
di memoria. Non dare per intercambiabili cache con score di attenzione diversi.

### Training sul layer hidden corretto

Le root contengono le stesse sottocartelle coorte. Il default dell'entry point
`train_wsi_landmark_forecaster.py` non è automaticamente la configurazione
hidden/ALiBi richiesta dal pruning TITAN: specificarla.

```bash
: "${EAF_WSI_CACHE_ROOT:?Impostare i target TITAN pretraining}"
: "${EAF_RUN_NAME:?Impostare un nuovo nome run WSI forecaster}"
python scripts/training/train_wsi_landmark_forecaster.py \
  --tile-eaf-root "$EAF_TILE_CACHE_ROOT" --wsi-eaf-root "$EAF_WSI_CACHE_ROOT" \
  --tile-encoder conch_v15 --wsi-encoder titan --tile-input-variant base \
  --input-source titan_hidden --titan-hidden-layer 0 --architecture dense_alibi \
  --attention-key attention/global_to_tiles_mass_share --target-layer -1 \
  --seed 42 --split-seed 17 --exclude-cohort --run-name "$EAF_RUN_NAME" \
  --output-dir "$EAF_WSI_ROOT/checkpoints/wsi_eaf/conch_v15__titan/$EAF_RUN_NAME"
```

Verificare che lo split effettivo del dataset WSI non separi slide dello stesso
paziente: il seed da solo non è una prova. Il runner con manifest di split
esplicito fa parte di I01/I03; non approvare run del paper se il controllo fallisce.

```bash
: "${EAF_WSI_FORECASTER_CKPT:?Impostare il forecaster TITAN hidden layer 0}"
: "${EAF_RUN_NAME:?Impostare un nuovo nome run WSI distillazione}"
python scripts/training/finetune_wsi_titan_pruned.py \
  --tile-eaf-root "$EAF_TILE_CACHE_ROOT" --wsi-eaf-root "$EAF_WSI_CACHE_ROOT" \
  --forecaster-checkpoint "$EAF_WSI_FORECASTER_CKPT" \
  --tile-encoder conch_v15 --wsi-encoder titan --tile-input-variant base \
  --prune-layer 0 --keep-ratio 0.20 --seed 42 --split-seed 17 \
  --exclude-cohort --run-name "$EAF_RUN_NAME" \
  --output-dir "$EAF_WSI_ROOT/checkpoints/wsi_eaf_pruned/conch_v15__titan/$EAF_RUN_NAME"
```

Per il braccio combinato leggere I04 prima di sostituire le root: gli hidden
input e i target full possono provenire da passaggi diversi. Le opzioni di
naming non effettuano questa separazione automaticamente.

### Valutazione attuale: soltanto CV interna TITAN

Definire root **downstream** esplicite `EAF_EVAL_WSI_ROOT`,
`EAF_EVAL_TILE_ROOT` e `EAF_LABELS_ROOT`, con nomi coorte allineati. Il checkpoint
è quello WSI distillato, non l'adapter Tile-EAF.

```bash
: "${EAF_EVAL_WSI_ROOT:?Impostare le cache TITAN baseline downstream}"
: "${EAF_EVAL_TILE_ROOT:?Impostare le feature tile downstream}"
: "${EAF_LABELS_ROOT:?Impostare le label downstream revisionate}"
: "${EAF_WSI_STUDENT_CKPT:?Impostare il checkpoint TITAN distillato}"
python scripts/evaluation/eval_wsi_linear_probing.py \
  --data-root "$EAF_WSI_ROOT" --labels-root "$EAF_LABELS_ROOT" \
  --wsi-eaf-root "$EAF_EVAL_WSI_ROOT" --tile-eaf-root "$EAF_EVAL_TILE_ROOT" \
  --pruned-checkpoint "$EAF_WSI_STUDENT_CKPT" \
  --folds 5 --seed 42 --wandb-mode disabled
```

Questo entry point non implementa il test train-TCGA/test-CPTAC, CoxNet, tutte
le metriche del paper o il pannello generico WSI. Per TCGA ricava il paziente
dal barcode, mentre per gli altri ID usa lo slide ID: serve il mapping esplicito
I01/I03 prima di applicarlo a coorti multi-slide non TCGA. I risultati attuali
sono utili per verifica preliminare, non sostituiscono E01/E02/B04.

## 5. Backlog implementativo con criteri di accettazione

Tutte le interfacce in questa sezione sono **da implementare**, non comandi
già disponibili. Riutilizzare `src/` e i domini esistenti; `scripts/` rimane
thin entry point. Nuove operazioni di dati/cache via `scripts/eaf.py`, nessun
nuovo script manuale smoke/debug. Dipendenze pesanti opzionali e test con gate.

| ID / priorità | Implementazione | Accettazione prima dei run |
| --- | --- | --- |
| I01 / P0 | Manifest task/split e audit label in `src/data/wsi/`; mapping globale pazienti, centro, fonte label, stato profiling | Rifiuta pazienti condivisi tra ruoli; missing label non diventa negativo; namespace e collisioni testati |
| I02 / P0 | Registro adapter tile/WSI in `src/wsi_pipeline/`; native transforms, pooling, revisioni e capability flags | Output full equivalente alla release su input fisso; shape, dtype, register token e coordinate testati |
| I03 / P0 | Runner generico classificazione esterna in `src/evaluation/`; entry point sottile e config congelata | Scaler/tuning solo sul training; mapping paziente esplicito; output per paziente; split comuni e skip motivati |
| I04 / P0 | Quattro bracci TITAN e distinzione input/target cache in `src/wsi_pipeline/` e dataset WSI | Hidden compressi allineati ai target full; mismatch di coordinate/cache/checkpoint rifiutato; nessun full teacher per epoca |
| I05 / P1 | Profiler raw-WSI-to-prediction e stima risorse in `src/wsi_pipeline/` | Tempi per fase e totale, sincronizzazione GPU, cache warm/cold, memoria, OOM e costo iniziale separati |
| I06 / P1 | CoxNet, few-shot, pannello molecolare, retrieval, metriche cliniche in `src/evaluation/` | Fold/seed condivisi; censoring valido; soglie dal validation; query/gallery disgiunte; FDR su tutto il pannello |
| I07 / P1 | CAMELYON, HEST e BRACS loader/valutazione nei domini data/evaluation/analysis | Coordinate ROI/spot verificate, leakage donatore/paziente escluso, bootstrap all'unità corretta |
| I08 / P1 | Tracciamento token, attenzione e artefatti in `src/analysis/` e hook esistenti | Indici risalenti alla WSI originale; prefissi esclusi dalle mappe; annotazioni reali separate da proxy |
| I09 / P2 | Runner multimodale con prompt/versione/input manifest | Stesso support set; risposte raw locali, parsing failures contati, costo API e disponibilità snapshot dichiarati |
| I10 / gate | Fattibilità WSI-EAF GigaPath | Attenzione/target definiti, pooling e coordinate fedeli, pruning LongNet verificato senza cambiare algoritmo; altrimenti solo comparatore |

### Contratti minimi da aggiungere senza duplicare i dati

- **Task specification:** ID stabile, origine EAGLE/nuovo, tipo di endpoint,
  classe positiva o class map, unità, coorti train/val/test, label provenance,
  eligibility, metriche primarie e famiglia di correzione multipla.
- **Split table locale:** task, cohort, patient_id, slide_id, center, fold,
  split; label e survival time/event tramite join documentato. Nessuna derivazione
  implicita del paziente non TCGA dal nome del file.
- **Model specification:** release/commit pesi, preprocess/FOV/MPP, pooling,
  dimensione, coordinate, native tile encoder, early layer, retention, dtype,
  attention reduction, checkpoint forecaster/LoRA e vincoli di compatibilità.
- **Predizioni:** task/run/model/seed/fold/cohort/patient, slide quando utile,
  y_true, score/probabilità per classe o rischio survival, split e validità.
  Conservare predizioni individuali, non soltanto medie o CSV finali.
- **Risultato esperimento:** estendere il riepilogo esistente con artifact path
  del protocollo, hash dei manifest/split, copertura, esclusioni, tempi, metriche
  e IC. Non creare un secondo formato incompatibile con
  `src/wsi_pipeline/experiment_results.py` senza migrazione/versione.

Il publisher corrente scrive `eaf.experiment_result.v1` in
`results/<family>/<stage>/<run_name>/summary.json`. Alcuni campi possono essere
null o incompleti nei run legacy: il fatto che esista il file non certifica
il contratto scientifico. Integrare metadati reali, non ricostruire valori ignoti.

### Organizzazione locale degli output

Usare soltanto sottodirectory della struttura esistente:

```text
$EAF_WSI_ROOT/
  datasets/pretraining/<version>/manifests/
  datasets/downstream/<dataset>/manifests/
  caches/tile_eaf/<dataset>/<encoder>/<cache-id>/
  caches/wsi_eaf/<dataset>/<pair>/<variant>/
  checkpoints/<family>/<model-or-pair>/<run-name>/
  results/paper/<experiment-id>/<run-name>/
  logs/paper/<experiment-id>/<run-name>/
```

`results/paper/...` contiene protocol/config, predizioni, coverage, metriche,
timing e figure. Il summary canonico del publisher può puntare a questi
artefatti; non duplicare array o raw WSI. Nessuno di questi output entra in Git.
Commit consentiti: codice, test, documentazione e piccole configurazioni prive
di dati individuali. I protocolli locali congelati devono restare recuperabili
anche se la configurazione di sviluppo cambia.

## 6. Verifiche e aggiornamento del registro

Per la sola documentazione: link relativi risolvibili, code fence bilanciati,
opzioni CLI confrontate con i parser e `git diff --check`. Non serve eseguire
training, download o l'intera suite per una modifica documentale.

Durante le implementazioni I01–I10 aggiungere test nella zona più vicina:

- Join etichette, pazienti multi-slide, coorti omonime, missing molecular assay,
  duplicati e split incompatibili.
- Equivalenza adapter vs riferimento e full-vs-retention-1, corretta esclusione
  dei register token, pooling e unità spaziali; memoria su bag variabili.
- Ordine coordinate e target cache; failure su teacher/student mismatch e
  verifica che il teacher completo non venga invocato per ogni epoca.
- Test leakage scaler/tuning/threshold, classi mancanti, pairing e bootstrap
  cluster; AUROC multiclass e dati censurati con esempi sintetici verificabili.
- Retrieval senza self-match o pazienti duplicati, metrica ROI/spot con geometria
  nota, pipeline multimodale con parsing e risposte invalide.
- Round-trip delle predizioni/summary, run interrotti non pubblicati come
  completi e errori/OOM registrati senza cambiare silenziosamente il protocollo.

Subset esistenti pertinenti, da usare quando cambia il relativo comportamento:

```bash
pytest tests/data/wsi/test_refactor_layout.py -q
pytest tests/wsi_pipeline/test_cache_contracts.py -q
pytest tests/wsi_pipeline/test_model_adapters.py -q
pytest tests/training/test_online_attention_distillation.py -q
pytest tests/scripts/test_eval_wsi_linear_probing.py -q
pytest tests/wsi_pipeline/test_experiment_results.py -q
```

Alla fine di ogni run: verificare gli artefatti, aggiornare stato/readiness
nella roadmap, collegare il summary e annotare la prossima azione. Un run
fallito o un effetto negativo sono informazioni da conservare. Aggiornare
`continuity.md` per i cambi operativi, mantenendo questa roadmap come registro
del programma del paper.
