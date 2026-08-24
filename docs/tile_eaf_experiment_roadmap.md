# Tile-EAF experiment roadmap

Piano delle fasi per la pipeline tile-level EAF dopo il refactoring minimale
(2026-08-24). Copre inferenza/cache offline per encoder, forecaster (EAF), pruning
distillation, e valutazione linear-probing pruned-vs-baseline. La pipeline WSI-EAF
non è coperta da questo documento.

## Convenzioni

- **Naming run**: `<tile-encoder>_src<NN>` (forecaster), `<tile-encoder>_src<NN>_pruned<keep%>pct`
  (pruning). Progetto W&B: `EAF-Tile-level` (Stage 1), `EAF-Tile-level-Pruned` (Stage 2).
- **Checkpoint**: `$EAF_WSI_ROOT/checkpoints/tile_eaf/<tile-encoder>/<run>/best_<run>.pt`
  + `summary_<run>.json` (idem `pruned_finetuned/`, `multi_thunder/`) — sotto la
  cartella dati canonica, non sotto la repo (`src/utils.py::default_checkpoint_root`,
  fix 2026-08-24; prima il default cadeva erroneamente in `EAF/checkpoints/`
  relativo alla repo). Fallback su `EAF/checkpoints/` con warning se
  `$EAF_WSI_ROOT` non è impostato. Una sotto-directory per run (mai file distinti
  solo dal suffisso del nome). `<tile-encoder>` è il nome reale dell'encoder
  caricato (es. `conch_v15`), non necessariamente il `--model-name` THUNDER usato
  per caricarlo (`titan` → `conch_v15`, vedi `src/utils.py::tile_encoder_dir_name`).
- **Cohort default**: `HISTAI-mixed` e `HISTAI-skin-b2` esclusi di default
  (`--exclude-cohort`) in ogni script tile-EAF — i due sotto-set HISTAI più grandi e
  lenti da scaricare.
- **Cache riutilizzabile tra source layer**: la cache compatta (`eaf.py cache-tile`)
  non dipende dal source layer (solo `final_attention`/`tile_embeddings`); una sola
  cache per encoder serve qualunque `--source-layer` (vedi fix 2026-08-24 in
  `train_wsi_tile_eaf_online.py`).
- **Nota naming nota**: `titan_src01`, `titan_src02_pruned{30,20,10}pct` (run lanciati
  il 2026-08-24 prima del fix successivo) usano ancora il `--model-name` THUNDER grezzo
  (`titan`) invece del tile-encoder risolto (`conch_v15`) nel nome del run/file —
  vivono comunque correttamente sotto `checkpoints/.../conch_v15/`. Corretto per i
  run successivi (`conch_v15_src00`, `conch_v15_src03`, ecc.); non rinominato in corsa
  per non rompere i path di checkpoint di job già in esecuzione.

## Dati utilizzati

### Stage 1 (forecaster) e Stage 2 (pruning distillation)

Stessa cache/manifest per entrambi gli stage — la distillazione Stage 2 legge
`tile_embeddings` dalla stessa cache compatta usata per addestrare il forecaster.

- **Manifest**: `$EAF_WSI_ROOT/datasets/pretraining/histai_eaf_wsi_v1/manifests/conch_v15_complete_v1/slides.csv`
  — **5495 slide, solo HISTAI** (colonna `source`=`histai` al 100%), 7 dei 9
  sotto-set (`HISTAI-mixed` e `HISTAI-skin-b2` assenti — non ancora scaricati al
  momento della costruzione della cache):

  | Sotto-set | Slide |
  |---|---|
  | HISTAI-skin-b1 | 1763 |
  | HISTAI-breast | 1687 |
  | HISTAI-colorectal-b1 | 996 |
  | HISTAI-thorax | 653 |
  | HISTAI-hematologic | 214 |
  | HISTAI-gastrointestinal | 120 |
  | HISTAI-colorectal-b2 | 62 |

  **Nota importante**: questo NON è il corpus "strict" HISTAI+GTEx+HEST
  (`eaf_wsi_pretrain_strict_v1`) descritto come corpus canonico di pretraining EAF
  in CLAUDE.md/`docs/data_layout.md`. `hest_eaf_thunder_clean_v1` esiste su disco
  ma non è ancora stato passato tramite `eaf.py cache-tile` per CONCH v1.5; GTEx non
  è scaricato in questo ambiente. Il tile-EAF (a differenza del WSI-EAF) sta quindi
  girando su un corpus più ristretto di quello canonico — espandere a HEST/GTEx è
  lavoro futuro, non bloccante per lo sweep del source layer.
- **Cache**: `$EAF_WSI_ROOT/datasets/pretraining/histai_eaf_wsi_v1/manifests/conch_v15_complete_v1/tile_cache_index.csv`
  (compatta, `coords`/`final_attention`/`tile_embeddings`, indipendente dal source
  layer — vedi nota sopra).
- **Split**: nessuno split pre-assegnato nel manifest (colonna `split` vuota) —
  calcolato a runtime da `load_wsi_manifest` con `--val-fraction 0.1
  --split-seed 42` (deterministico), `--slide-group diagnostic`.
- **Campionamento**: `--tiles-per-wsi 500`, `--batch-size 64-128`, bilanciamento
  per cohort (`--cohort-balance-power 0.5`).

### Stage 3 (linear probing pruned vs baseline)

- **Base data folder**: `$THUNDER_BASE_DATA_FOLDER` (`/data2/home/vcivale/projects/imaging/data/thunder-tiles/`),
  scoperto automaticamente da `ThunderDatasetRegistry` via `datasets/data_splits/*.json`
  — nessun elenco curato nel codice.
- **Dataset effettivamente presenti in questo ambiente** (15, verificato 2026-08-24):
  `bach`, `bracs`, `break_his`, `ccrcc`, `crc`, `esca`, `patch_camelyon`,
  `spider_breast`, `spider_colorectal`, `spider_skin`, `spider_thorax`,
  `tcga_crc_msi`, `tcga_tils`, `tcga_uniform`, `wilds`. Il registro THUNDER ne
  supporta altri (es. `mhist`, `ocelot`, `pannuke`, `segpath_epithelial`,
  `segpath_lymphocytes`) ma non sono materializzati qui — se scaricati, verrebbero
  inclusi automaticamente senza modifiche al codice.
- **Split**: gestito da `ThunderDatasetRegistry`/`build_multi_thunder_train_loaders`
  a partire da `data_splits/*.json`; `--n-holdout 0` per usare tutti i dataset
  (nessun holdout) nel confronto pruned-vs-baseline.

## Note operative (GPU condivisa)

Lezioni dall'incidente del 2026-08-24 (lanciando 4 job concorrenti tramite MPS,
vedi Fase 1/2 sotto), utili per ogni lancio futuro su questa macchina:

- **Sempre `--num-workers 16 --slide-cache-size 20`** con `--slides-per-batch 16`
  (il default dello script, `--num-workers 8 --slide-cache-size 4`, causa un pattern
  a raffica-poi-stallo: con `slide-cache-size` troppo piccolo rispetto a
  `slides-per-batch`, quasi ogni WSI del gruppo va riaperta da zero ad ogni batch).
- **MPS**: su GPU condivisa con altri processi, lanciare sempre con
  `CUDA_MPS_PIPE_DIRECTORY=/data2/home/vcivale/mps/pipe
  CUDA_MPS_LOG_DIRECTORY=/data2/home/vcivale/mps/log` (demone MPS già attivo sulla
  macchina) — altrimenti si gira in context-switching non-MPS, peggiorando ulteriormente
  il pattern raffica-stallo.
- **Concorrenza GPU reale ≈ 2 job**, non di più, anche con MPS: un job pruning
  (Stage 2, `--keep-ratio` qualunque) può stabilizzarsi ovunque tra ~2GB e ~19GB a
  seconda del traffico/allocator (osservato: un run keep20% consolidato a 19.4GB,
  ben oltre gli altri due). Su 40GB totali, con un altro processo utente da ~4GB
  quasi sempre presente, un terzo job concorrente rischia OOM concreto (successo 2
  volte su 2 tentativi il 2026-08-24). Per lanci multipli, preferire sequenziale con
  un controllo di margine GPU reale (`nvidia-smi --query-gpu=memory.used`), non un
  numero fisso di job.

## Fase 0 — Smoke test del refactor multi-encoder

`eaf.py cache-tile --encoder uni2h` su poche slide già in cache, poi
`train_wsi_tile_eaf_online.py --target-cache-index` sul mini-cache risultante.
Convalida `HookedViTTileTeacherAdapter.from_thunder_model` end-to-end prima di
spendere compute reale su un encoder nuovo.

## Fase 1 — CONCH v1.5: sweep del source layer (IN CORSO)

Determinare se layer 2 è davvero il miglior compromesso efficienza/performance tra i
primissimi layer. Un solo checkpoint Stage-1 tenuto finora:
`checkpoints/tile_eaf/conch_v15/conch_v15_src02/` (20/20 epoche, best_val_kl=0.0889,
**val_rho=0.81** — vedi `PROVENANCE.md` lì dentro per la storia completa, inclusi i
run scartati).

**Priorità rivista (2026-08-24)**: rho=0.81 al layer 2 è già molto alto — probabile
plateau. Il payoff di efficienza di EAF sta tutto nell'andare più in superficie
(meno blocchi transformer calcolati prima del pruning), non più in profondità, quindi
lo sweep non punta più a {1, 2, 3} ma a **quanto presto si può andare senza perdere
qualità**:

1. **`conch_v15_src01`** (priorità alta) — un blocco più in superficie del 2. Se rho
   resta vicino a 0.81, guadagno di efficienza quasi gratis.
2. **`conch_v15_src00`** (priorità media/alta, solo se src01 regge bene) — il
   pavimento assoluto (output del primissimo blocco). Se rho collassa qui, individua
   la vera soglia minima di segnale utile tra layer 0 e 1.
3. **`conch_v15_src03`** (priorità bassa, opzionale) — solo per confermare la forma
   del plateau oltre il layer 2, non per cercare un miglioramento; da lanciare solo
   se avanza budget dopo 0 e 1.

Stessa cache/corpus/iperparametri del layer 2, cambia solo `--source-layer` (la cache
compatta è indipendente dal source layer, riutilizzabile senza rebuild). Confronto
finale su rho/KL **e** costo computazionale del pruning a quel layer — non vince
semplicemente la metrica di fit più alta.

**Stato (2026-08-24)**: `titan_src01` (source layer 1) in corso — vedi nota naming
sopra sul perché si chiama `titan_src01` e non `conch_v15_src01`.
`conch_v15_src00` non ancora lanciato (in attesa dell'esito di src01).

## Fase 2 — Stage 2 (pruning distillation), sul layer vincente

Lanciata in anticipo sul layer 2 (in parallelo allo sweep Fase 1, non atteso l'esito)
per validare da subito lo Stage 3 — vedi Fase 3. Tre keep-ratio (30/20/10%), stesso
budget di epoche coerente e verificato-completo per tutti (i checkpoint precedenti
erano stati eliminati: uno senza summary/epoche non verificabili, gli altri due
fermati a 5/20 epoche pianificate — budget incoerente per un confronto valido).

**Stato (2026-08-24)**: `titan_src02_pruned20pct` in corso, riavviato con
`--tiles-per-wsi 100` (invece di 500 → epoca 3850 batch invece di 19250) dopo aver
osservato su W&B che la loss di distillazione era in plateau da migliaia di step con
l'epoca lunga — early-stopping/validation scattano molto più spesso così, nessun
cambio alla ricetta (lr/LoRA/pesi di loss invariati). `titan_src02_pruned30pct` e
`titan_src02_pruned10pct` ancora da lanciare, stessa `--tiles-per-wsi 100`,
manualmente uno alla volta (lo script di coda automatico è stato abbandonato dopo
una race condition — ha lanciato pruned30pct nello stesso momento del riavvio
manuale di pruned20pct, rischiando un altro OOM a 3 job).

## Fase 3 — Stage 3 (linear probing, tutti i dataset THUNDER)

`scripts/train_multi_thunder_classifier.py --adaptation linear_probing --n-holdout 0`,
una volta senza `--pruned-adapter-ckpt` (baseline) e una per ciascun keep-ratio con
`--pruned-adapter-ckpt` (auto-risolve forecaster/prune-layer/keep-ratio dal
checkpoint Stage 2). Copre l'intero registro dataset THUNDER scoperto
automaticamente, non un sottoinsieme curato — il linear probing non fa backward nel
backbone, quindi è economico farlo su tutti.

## Fase 4 — Nuovi encoder (UNI2-h → Virchow2 → H-Optimus-1 → Prov-GigaPath)

Sequenziale, un encoder alla volta (evita contesa GPU). Per ciascuno: Fase 0 (se non
già fatta) → cache offline sul corpus completo → sweep source layer {1, 2, 3} → Stage
2 sul vincente → Stage 3 linear probing su tutti i dataset THUNDER.
