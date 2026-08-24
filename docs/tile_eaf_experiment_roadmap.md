# Tile-EAF experiment roadmap

Piano delle fasi per la pipeline tile-level EAF dopo il refactoring minimale
(2026-08-24). Copre inferenza/cache offline per encoder, forecaster (EAF), pruning
distillation, e valutazione linear-probing pruned-vs-baseline. La pipeline WSI-EAF
non è coperta da questo documento.

## Convenzioni

- **Naming run**: `<tile-encoder>_src<NN>` (forecaster), `<tile-encoder>_src<NN>_pruned<keep%>pct`
  (pruning). Progetto W&B: `EAF-Tile-level` (Stage 1), `EAF-Tile-level-Pruned` (Stage 2).
- **Checkpoint**: `checkpoints/tile_eaf/<tile-encoder>/<run>/best_<run>.pt` +
  `summary_<run>.json`, una sotto-directory per run (mai file distinti solo dal
  suffisso del nome). `<tile-encoder>` è il nome reale dell'encoder caricato (es.
  `conch_v15`), non necessariamente il `--model-name` THUNDER usato per caricarlo
  (`titan` → `conch_v15`, vedi `src/utils.py::tile_encoder_dir_name`).
- **Cohort default**: `HISTAI-mixed` e `HISTAI-skin-b2` esclusi di default
  (`--exclude-cohort`) in ogni script tile-EAF — i due sotto-set HISTAI più grandi e
  lenti da scaricare.
- **Cache riutilizzabile tra source layer**: la cache compatta (`eaf.py cache-tile`)
  non dipende dal source layer (solo `final_attention`/`tile_embeddings`); una sola
  cache per encoder serve qualunque `--source-layer` (vedi fix 2026-08-24 in
  `train_wsi_tile_eaf_online.py`).

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

## Fase 2 — Stage 2 (pruning distillation), sul layer vincente

Una volta scelto il source layer migliore per CONCH v1.5, ri-addestrare Stage 2 ai
tre keep-ratio (30/20/10%) con un budget di epoche coerente e verificato-completo
(i checkpoint precedenti sono stati eliminati: uno senza summary/epoche non
verificabili, gli altri due fermati a 5/20 epoche pianificate — budget incoerente per
un confronto valido).

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
