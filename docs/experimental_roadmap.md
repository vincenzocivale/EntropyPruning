# Roadmap sperimentale del paper EAF

Ultima verifica documentale e degli artefatti: **16 settembre 2026**.
Codice ispezionato prima di questo aggiornamento: `aa09c9a`.

Questo è il registro di pianificazione del paper: obiettivi, modelli, stato,
dipendenze e criteri di completamento. Non è un report di risultati ancora da
ottenere. Leggere insieme:

- [Protocolli scientifici](experimental_protocols.md): dataset, confronti,
  metriche, analisi statistiche e criteri di interpretazione.
- [Runbook e implementazione](experimental_runbook.md): comandi esistenti,
  prerequisiti, funzionalità mancanti e test da implementare.
- [Continuità operativa](continuity.md): evidenze storiche e passaggio di consegne.
- [Pipeline](pipeline.md) e [layout dei dati](data_layout.md): workflow supportato
  e invarianti di conservazione dei dati.

## 1. Decisioni concordate e ipotesi

Destinazione: journal biomedicale. L'ipotesi centrale è che la riduzione del
calcolo tramite EAF conservi segnali patologici critici, inclusi piccoli focolai
e informazione biologica distribuita. Non è una proprietà garantita dal pruning:
anche EAF può perdere questi segnali. La dimostrazione deve derivare dagli
endpoint downstream, non dalla sola concordanza con l'attenzione del teacher.

Il perimetro concordato è:

1. Tutte le **famiglie sperimentali di EAGLE** sono il requisito minimo.
2. Solo dati pubblici; eventuali registrazioni per dataset pubblici sono
   compatibili con il piano, nuove coorti private o accordi clinici no.
3. Riprodurre gli esperimenti sulle coorti originali pubbliche dove possibile;
   documentare ogni sostituzione. Non dichiarare una replica integrale di EAGLE.
4. Mantenere il metodo EAF attuale. Sono ammesse integrazioni di backbone,
   evaluator e tracciamento; non introdurre budget adattivi, nuove loss di
   protezione delle lesioni o selettori biologici come parte del metodo.
5. Nessuna nuova revisione patologica o raccolta prospettica è un prerequisito.
   Le annotazioni pubbliche possono supportare valutazioni quantitative, ma non
   sostituiscono una nuova reader study.
6. Riutilizzare asset validati. Non spostare, eliminare o rielaborare TCGA/HEST
   esistenti come parte di un refactor. Una nuova variante sperimentale richiede
   output versionati separati, senza sovrascrivere quelli precedenti.

Tile-EAF riduce i token all'interno del tile encoder. WSI-EAF riduce i token
all'interno dell'aggregatore dopo che le feature tile sono state estratte.
Il solo WSI-EAF non elimina il costo della codifica dei tile. Il sistema
combinato va valutato separatamente da entrambi i suoi componenti.

## 2. Come leggere lo stato

Usare due dimensioni indipendenti, aggiornate con evidenze:

| Stato scientifico | Significato |
| --- | --- |
| Concluso-validato | Protocollo completato, predizioni/metriche verificabili, costo e controlli pertinenti disponibili |
| Training concluso | Checkpoint e summary presenti; non dimostra qualità downstream |
| Parziale | Esistono solo alcune fasi, configurazioni o evidenze |
| Pianificato / da zero | Nessun risultato verificato per il protocollo; codice o asset generici possono già esistere |
| Non comparabile | Contaminazione, provenienza insufficiente o incompatibilità; escluso da selezione e claim |

| Eseguibilità | Significato |
| --- | --- |
| E0: subito | Comando, ambiente e input verificati per quella specifica operazione |
| E1: preparazione | Codice disponibile, ma servono cache, manifest, pesi, validazione o risorse |
| E2: implementazione | Manca un adapter, evaluator, contratto o controllo necessario |
| E3: vincolo esterno | Immagini, etichette, checkpoint o snapshot non accessibili |

Un task può avere più prerequisiti E1/E2/E3. Il catalogo automatico rileva file
e metadati, non certifica disgiunzione, qualità del modello o correttezza
clinica. Non equiparare `complete` del catalogo a `Concluso-validato`.
L'assenza nel catalogo significa «nessuna evidenza rilevata», non una prova
assoluta che un esperimento non sia mai stato eseguito.

## 3. Inventario verificato e lavoro immediato

L'audit rigenerato durante questo aggiornamento contiene **26 run: 7 complete,
18 partial e 1 non_comparable**, 20.373 file di metadati/manifest cache,
29 file di etichette, 5 profili e 27 righe nella tabella baseline legacy.
Il precedente snapshot di continuità ne riportava 36 per i metadati cache:
sono conteggi di file, non di cache validate o WSI indipendenti. Il file locale è
`$EAF_WSI_ROOT/results/experiment_catalog/catalog.json` e non va committato.

### Training catalogati come completi

| Run | Fase | Evidenza numerica riportata dal catalogo |
| --- | --- | --- |
| `conch_v15_src00` | Forecaster tile | best val KL 0.098612; max val rho 0.794784 |
| `titan_src01` | Forecaster tile CONCH, nome legacy | best val KL 0.093156; max val rho 0.805347 |
| `conch_v15_src00_pruned10pct` | Distillazione tile | best val loss 0.050346 |
| `conch_v15_src01_pruned10pct` | Distillazione tile | best val loss 0.067146 |
| `conch_v15_src01_pruned20pct` | Distillazione tile | best val loss 0.032140 |
| `conch_v15_src02_pruned10pct` | Distillazione tile | best val loss 0.048311 |
| `conch_v15_src02_pruned20pct` | Distillazione tile | best val loss 0.028293 |

I run canonici sono sotto `checkpoints/tile_eaf/conch_v15/` e
`checkpoints/pruned_finetuned/conch_v15/` del runtime root. `titan_src01` è
ancora rilevato nella directory legacy della repo; risolvere il percorso dal
catalogo senza copiarlo o spostarlo. `summaries` e `checkpoints` nel catalogo
identificano i file di evidenza. Le metriche non selezionano il modello per il
paper: loss, layer e campionamenti diversi non sono una classifica downstream.

### Stato per componente

| ID | Esperimento/componente | Stato | Eseguibilità e prossima azione |
| --- | --- | --- | --- |
| A00 | Inventario e lettura metadati | Disponibile | E0 per ispezionare il catalogo; audit rigenera solo il report locale |
| A01 | Disgiunzione e qualità etichette | Parziale | E1/E2: audit pazienti, studi HEST, profiling molecolare e armonizzazione label |
| T01 | CONCH v1.5 cache/forecaster/distillazione | Training conclusi parzialmente nel grid | E1: validare cache, split e checkpoint; poi inferenza compressa e valutazione appaiata |
| T02 | UNI2-h Tile-EAF | Cache HEST/THUNDER presente; resto da zero | E1: verificare integrità e overlap HEST; poi forecaster e distillazione |
| T03 | Virchow2 Tile-EAF | Da zero nel catalogo | E1/E2: pesi, adapter/pooling, cache, training, valutazione |
| T04 | H-Optimus-1 Tile-EAF | Da zero nel catalogo | E1/E2: stesso percorso T03 con trasformazioni native |
| T05 | Prov-GigaPath Tile-EAF | Da zero nel catalogo | E1/E2: stesso percorso T03 e compatibilità input del slide encoder |
| W01 | TITAN WSI-EAF | Cache presenti; training legacy parziali | E1: ricostruire provenienza e summary, verificare layer e coordinate, completare training |
| W02 | Tile-EAF + WSI-EAF | Nessun confronto completo verificato | E1/E2: cache input compresse, quattro bracci e target teacher coerenti |
| W03 | WSI-EAF su GigaPath | Da zero | E2: gate tecnico LongNet/attenzione; nessuna promessa di supporto attuale |
| V01 | Linear probing TITAN esistente | Baseline parziali, no confronto EAF completo | E1 per CV interna dopo validazione; E2 per valutazione esterna generale |
| V02 | EAGLE e altri comparatori WSI | Pianificati | E1/E2: release, estrazione nativa e runner comune |
| D01 | TCGA downstream | 29 file label presenti | E1/E2: controllare provenienza; non sono 29 task esterni indipendenti |
| D02 | CPTAC / PathoBench / BRACS | Manifest o split presenti | E1: inventario separato di immagini, label, cache e copertura; presenza split non certifica WSI |
| X00 | `hidden_layer2_final_CONTAMINATED_bak_20260821` | Non comparabile | Escludere da training selection, benchmark e figure; conservare evidenza del motivo |

Non è ancora verificato come concluso alcun esperimento completo del paper
comprendente confronto EAF, endpoint downstream e costo. Non dichiarare
«lanciabile subito» un training soltanto perché il suo script esiste.

## 4. Pannello dei foundation model

### Tile: dieci modelli, cinque con EAF

| Modello | Ruolo deciso | Perché includerlo |
| --- | --- | --- |
| CONCH v1.5 | Completo + Tile-EAF | Asset iniziali e pipeline TITAN |
| Virchow2 | Completo + Tile-EAF | Backbone del confronto EAGLE e input PRISM2 |
| UNI2-h | Completo + Tile-EAF | Trasferibilità a una famiglia indipendente |
| H-Optimus-1 | Completo + Tile-EAF | Compressione di un encoder di grande scala |
| Prov-GigaPath tile | Completo + Tile-EAF | Seconda pipeline tile–slide nativa |
| CTransPath | Comparatore originale | Selezione CHIEF/EAGLE; non forzare l'adapter ViT su Swin |
| Virchow | Comparatore originale | Confronto storico e input PRISM |
| CONCH originale | Comparatore originale | Confronto storico e input MADELEINE |
| H0-mini | Comparatore compatto | Alternativa pratica all'uso di un grande FM compresso |
| GigaPath-Flash tile | Comparatore compatto | Valutazione della pipeline efficiente nativa |

H0-mini deriva da H-Optimus-0, non H-Optimus-1: non è una ablation di quel
teacher. UNI2-h, Virchow2, CONCH e gli altri encoder richiedono pooling,
prefix/register token e preprocessing propri; non usare una normalizzazione
universale soltanto perché gli adapter hanno la stessa interfaccia.

### WSI: otto comparatori

| Modello | Input e valutazione |
| --- | --- |
| TITAN | CONCH v1.5; completo, solo Tile-EAF, solo WSI-EAF, combinazione |
| Prov-GigaPath | Tile encoder nativo completo/Tile-EAF; WSI-EAF solo se supera W03 |
| PRISM | Virchow nativo, versione originale |
| PRISM2 | Virchow2 CLS-only, 224 px a 0.5 MPP; base embedding principale, diagnostic embedding separato |
| CHIEF | CTransPath nativo; anche componente di EAGLE |
| COBRA | Checkpoint e coppia di feature della release EAGLE, fissati nel registro |
| MADELEINE | CONCH originale e preprocessing ufficiale |
| GigaPath-Flash | Coppia tile–slide nativa, non input GigaPath originale |

PRISM2 integra il confronto aggiornato senza sostituire PRISM. La versione
specializzata survival non entra nel confronto principale di rappresentazioni
generali. EAGLE è un sistema ulteriore, distinto dai singoli FM.

Gli adapter attuali della repo coprono TITAN, GigaPath e FEATHER; la CLI di
estrazione WSI-EAF espone TITAN/FEATHER. FEATHER è utile per verifiche tecniche,
non sostituisce uno degli otto comparatori. Le opzioni `--wsi-encoder` degli
script di training TITAN sono etichette di naming, non un dispatch di modelli.

L'adapter GigaPath non espone attualmente attenzione nativa. W03 deve verificare
target, layer, posizioni, pooling e pruning LongNet senza cambiare la definizione
di EAF. Se occorre un nuovo selettore, mantenere GigaPath come comparatore e
limitare esplicitamente il claim WSI a TITAN.

## 5. Registro degli esperimenti da completare

Ogni ID rimanda alla sezione omonima dei [protocolli](experimental_protocols.md).
Le righe sono famiglie da espandere in run, non risultati già ottenuti.

| ID | Obiettivo e contributo | Stato / readiness | Dipendenze principali |
| --- | --- | --- | --- |
| E01 | Benchmark morfologico/molecolare/N-M; confronto generale | Parziale / E1+E2 | A01, T01, W01, V02, D01–D02 |
| E02 | Survival e trattamento; utilità su outcome | Da zero / E1+E2 | WSI pubbliche PathoBench, evaluator CoxNet/classificazione |
| E03 | Few-shot e training ridotto; utilità con poche label | Da zero / E2 | E01, split e campionamenti condivisi |
| E04 | Aggregatori, budget, backbone, risoluzione, fusion, tuning | Parziale tecnico / E1+E2 | E01, runner ablation; nessun confronto sistematico concluso |
| E05 | Attenzione, regioni e artefatti | Parziale tecnico / E1+E2 | Analisi esistenti, annotazioni pubbliche, tracciamento token |
| E06 | Efficienza e frontiera qualità–costo | Profili parziali / E1+E2 | Predizioni E01 e profiler end-to-end |
| E07 | Screening molecolare ampio e riuso embedding | Da zero / E1+E2 | Intersezione label TCGA–CPTAC, FDR e costo totale |
| E08 | Retrieval e UMAP | Da zero / E2 | Embedding congelati, gallery/query disgiunte |
| E09 | Confronto visione–linguaggio | Da zero / E1+E2; snapshot storico E3 se assente | PRISM2, protocollo pubblico, disponibilità GPT-4o storico |
| B01 | Piccoli focolai CAMELYON; novità principale | Da zero / E1+E2 | Annotazioni, disgiunzione CAMELYON16/17, evaluator sensibilità |
| B02 | HEST; conservazione biologica indipendente | Da zero / E1+E2 | Audit HEST, espressione/coordinate, regressione e domini |
| B03 | BRACS; eterogeneità e sottostima | Da zero / E1+E2 | WSI/ROI e split paziente; split già presenti |
| B04 | MSI esterna; approfondimento clinico | Da zero / E1+E2 | E01 CRC, calibrazione e soglie congelate |

### Ordine e gate

1. **G0 — provenienza:** A00/A01, inventario per coorte, esclusione X00,
   validazione degli input; nessun training su downstream.
2. **G1 — baseline misurata:** T01/W01/V01, checkpoint caricabili, estrazione
   compressa e profiling su manifest di sviluppo fisso. Verificare EAGLE.
3. **G2 — infrastruttura comparativa:** E01/E06 con predizioni individuali,
   split esterni e cache versionate. Poi T03/T02, seguiti da T04/T05, uno alla volta.
4. **G3 — copertura EAGLE:** E02–E09 e otto WSI-FM. B01 può partire dopo G2;
   B02 dopo la disgiunzione HEST; B03/B04 condividono gli evaluator E01.
5. **G4 — congelamento:** fissare config, task eleggibili, seed, soglie,
   ipotesi statistiche e manifest prima di sbloccare i test finali.
6. **G5 — paper:** repliche, risultati negativi, analisi d'incertezza, figure,
   matrice pubblica di repliche/adattamenti/non riproducibili.

Non dare stime di GPU-ore o TB senza misurazione: usare E06 su un campione
stratificato per numero di tile, registrare hardware e throughput, poi stimare
ogni fase con margine e spazio libero. Cache teacher, training, estrazione
downstream e fit delle head hanno costi distinti.

### Criterio di chiusura e aggiornamento

Per promuovere un esperimento a concluso-validato servono protocollo congelato,
provenienza dei dati/pesi, esclusioni, predizioni individuali, metriche con
incertezza, tempi pertinenti e confronto appaiato. OOM, task saltati e test
inconcludenti devono rimanere nel registro. «Non significativo» non dimostra
equivalenza. La sensibilità ai piccoli focolai deve sostenere il claim specifico;
un vantaggio di sola velocità sostiene soltanto un claim di efficienza.

Dopo ogni fase aggiornare questa tabella con data, ID run, percorso relativo
del summary, stato/readiness, motivo del blocco, prossima azione e dipendenze.
Tenere risultati voluminosi solo nel runtime root. Il catalogo non aggiorna
automaticamente questa roadmap e non è un orchestratore dei run.

## 6. Fonti e versioni da congelare

Fonti consultate nella pianificazione del 16 settembre 2026. Registrare commit,
revisioni dei pesi e versioni dei dataset nel protocollo locale prima dei run;
una pagina `main` non è un riferimento riproducibile.

- [EAGLE, Nature Communications 2026](https://www.nature.com/articles/s41467-026-74918-9)
  e [codice ufficiale](https://github.com/KatherLab/EAGLE);
  [STAMP-Benchmark](https://github.com/KatherLab/STAMP-Benchmark).
- [PathoBench: split e link alle immagini](https://huggingface.co/datasets/MahmoodLab/Patho-Bench).
- [TITAN / CONCH v1.5](https://github.com/mahmoodlab/TITAN),
  [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h),
  [Virchow2](https://huggingface.co/paige-ai/Virchow2),
  [H-Optimus-1](https://huggingface.co/bioptimus/H-optimus-1),
  [H0-mini](https://huggingface.co/bioptimus/H0-mini).
- [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath),
  [GigaPath-Flash pesi](https://huggingface.co/prov-gigapath/prov-gigapath-flash),
  [GigaPath-Flash preprint](https://arxiv.org/abs/2607.18218).
- [PRISM2](https://huggingface.co/paige-ai/Prism2),
  [COBRA](https://github.com/KatherLab/COBRA),
  [MADELEINE](https://github.com/mahmoodlab/MADELEINE).
- [CAMELYON](https://camelyon17.grand-challenge.org/Data/),
  [HEST](https://github.com/mahmoodlab/HEST),
  [BRACS](https://www.bracs.icar.cnr.it/background/).
- Contesto del claim: [TAP-Path, preprint](https://arxiv.org/abs/2609.04071),
  [SpaPath-Bench, preprint](https://arxiv.org/abs/2605.25764),
  [robustezza dei FM](https://www.nature.com/articles/s41467-026-73923-2),
  [MSIntuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC10628260/).

TAP-Path e SpaPath-Bench richiedono confronto nel related work; non attribuire
a EAF la prima applicazione del pruning o la prima valutazione spaziale in
patologia. Un eventuale confronto quantitativo TAP-Path richiede codice/pesi
riproducibili e un braccio separato per la sua supervisione task-specifica.
