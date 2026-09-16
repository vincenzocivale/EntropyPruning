# Protocolli scientifici del paper EAF

Piano del 16 settembre 2026. Stato e priorità nella
[roadmap](experimental_roadmap.md); implementazione nel
[runbook](experimental_runbook.md). I valori sotto sono default prospettici,
non descrizioni di esperimenti già eseguiti.

## 1. Regole comuni

### Unità, split e leakage

- L'unità indipendente è il paziente; tutte le sue WSI, sezioni e ROI devono
  restare nello stesso split. Per HEST raggruppare anche per donatore/studio.
- Separare tre ruoli: pretraining/distillazione EAF senza label downstream,
  sviluppo del task e test finale. Non fare adattamento transduttivo EAF sul test.
- Registrare centro, preparazione FFPE/frozen, scanner, MPP e fonte label dove
  disponibili, senza inferire metadati mancanti dal nome del file.
- Verificare duplicati attraverso case ID, serie, percorsi risolti e checksum
  disponibili. Dichiarare il pretraining noto dei FM separatamente: disgiunzione
  EAF non significa disgiunzione dimostrata dei foundation model proprietari.
- Usare gli split ufficiali pubblici quando corretti per paziente; documentare
  una correzione di leakage prima di osservare le performance.
- Il test esterno non entra in scaling, tuning, scelta layer/retention, pannello
  molecolare, calibrazione, soglie o scelta delle figure illustrative.

### Matrice dei bracci

Per ogni coppia nativa supportata valutare `full`, `tile_eaf`, `wsi_eaf` e
`tile_and_wsi_eaf`; per le coppie senza WSI-EAF valutare soltanto i bracci
implementati. Includere EAGLE originale e i comparatori della roadmap.

Per TITAN il riferimento della distillazione combinata principale rimane
**TITAN completo su feature CONCH complete**. Riaddestrare il forecaster WSI
sugli hidden state generati dagli input compressi, mantenendo target full
allineati, richiede il contratto I04 del runbook. Se il training corrente usa
target TITAN su feature già compresse, etichettare il risultato come
distillazione sequenziale; non confonderlo con il riferimento full.

Due valutazioni diverse:

1. **Utilità:** head riaddestrata per ogni rappresentazione, stesso protocollo e
   budget di tuning. Classificatori e scaler apprendono solo dai dati di sviluppo.
2. **Compatibilità:** head del modello completo congelata sugli embedding EAF
   compatibili. Non applicare questa prova tra dimensioni/spazi differenti.

Non cambiare arbitrariamente tile encoder sotto un aggregatore pretrained.
Nel confronto controllato con EAGLE usare selezione CHIEF e pooling su uno
stesso tile backbone, distinguendolo sempre dall'EAGLE originale.

### Default di sviluppo e selezione

- Seed EAF di conferma: `42, 43, 44`; seed split `17`, salvo split ufficiali.
  Mantenere gli split dei run storici per la loro verifica; non rinominarli come
  nuovi seed o sommarli indiscriminatamente alle repliche prospettiche.
- Layer candidati iniziali: indici zero-based `0, 1, 2, 3`, solo se rimane almeno
  un blocco successivo. Retention `0.10, 0.20, 0.50`; full come riferimento.
- Il forecaster usa target dell'ultimo layer reale del teacher. Layer source,
  pruning point, prefix token, pooling e coordinate devono coincidere tra
  training e inferenza. Tenere i prefix/register token richiesti dal modello.
- Fare screening layer con un seed sullo sviluppo; scegliere il punto operativo
  principale più veloce con perdita media di AUROC non oltre 0.01 rispetto al
  full sul pannello di sviluppo congelato. Questa è una tolleranza ingegneristica,
  non un margine clinico di non inferiorità. Se nessuna config passa, riportare
  la frontiera completa e non dichiarare un punto equivalente al full.
- Confermare il punto scelto con tre seed; riportare anche le altre retention
  preregistrate. Nessuna scelta del seed migliore usando il test.
- Per le head partire dalle configurazioni archiviate EAGLE/PathoBench. Congelare
  la griglia prima dei run; budget identico tra metodi. Le classi eleggibili per
  classificazione hanno almeno dieci pazienti per classe in training e test;
  ciò è un filtro minimo di fattibilità, non una garanzia di potenza.

### Statistica e reporting

- AUROC, AUPRC, balanced accuracy e macro-F1 per classificazione; one-vs-rest
  macro per multiclass. Esplicitare la classe positiva e la prevalenza.
- Analisi cliniche aggiuntive: Brier score, curva di calibrazione, sensibilità e
  specificità a soglia bloccata. Salvare anche le predizioni non calibrate.
- Bootstrap appaiato per paziente, 2.000 repliche, IC 95%; tenere tutte le WSI
  dello stesso paziente insieme. Non trattare seed, fold o spot come osservazioni
  cliniche indipendenti. Dichiarare le repliche bootstrap non calcolabili.
- Test DeLong per AUROC binaria appaiata quando appropriato; bootstrap per
  differenze multiclass, sensibilità e survival. Correzione Benjamini–Hochberg
  entro famiglie dichiarate; B01 è l'ipotesi biomedicale primaria.
- Sugli esterni salvare sia ciascun modello dei cinque fold di sviluppo sia
  l'ensemble di probabilità. I cinque passaggi sullo stesso test non sono cinque
  coorti indipendenti. Per CV ripetuta usare predizioni aggregate per paziente
  con numerosità delle ripetizioni documentata, non test sui fold come campioni.
- Risultati mancanti, OOM e skip restano in tabella con motivo. Un confronto
  appaiato usa pazienti comuni, ma riportare anche copertura e tasso di fallimento
  sulla coorte originale: non nascondere i casi difficili nell'intersezione.
- Niente claim di non inferiorità senza margine giustificato e analisi di potenza
  prespecificati. Se i positivi sono pochi, conclusione «inconcludente».

## 2. Copertura EAGLE sui dati pubblici

La versione pubblicata di [EAGLE](https://www.nature.com/articles/s41467-026-74918-9)
è il riferimento. Il piano comprende benchmark, survival/trattamento, pochi
dati, ablation, attenzione, efficienza, screening molecolare, retrieval e
confronto multimodale. Le coorti riservate non vengono considerate disponibili.
Ogni task deve avere un record `replica`, `adattamento_pubblico` o
`non_riproducibile`, con protocollo di origine e motivo della differenza.
Il riferimento comprende 31 task nel benchmark principale e 12 nell'estensione
PathoBench. Il numero effettivo della replica pubblica va calcolato dal mapping
endpoint/coorte eleggibile, non forzato a 43 e non sostituito dai 29 file label
TCGA presenti localmente.

### E01 — Morfologia, biomarcatori e stato N/M

Sviluppo su TCGA-BRCA, COAD/READ, LUAD/LUSC e STAD; test esterno CPTAC-BRCA,
COAD, LUAD/LSCC quando label compatibili. Unire LUAD/LUSC per il sottotipo
NSCLC; analizzare COAD/READ con identità di paziente coerenti. Distinguere sito
di origine, centro e diagnosi per non usare la coorte come scorciatoia.

Pannello candidato da congelare dopo audit label, senza guardare performance:

| Dominio | Endpoint candidati |
| --- | --- |
| CRC | MSI, BRAF, KRAS, sidedness, N, M; CIMP solo con label validate |
| Polmone | LUAD/LUSC, EGFR, STK11, KRAS, TP53, N, M |
| Mammella | ER, PR, HER2, PIK3CA, N, M |
| Stomaco | Lauren, EBV, MSI, TP53, N, M |

Il mapping effettivo deve distinguere endpoint da coppia endpoint/coorte.
Un endpoint senza test pubblico compatibile resta interno oppure non
riproducibile; non assegnare a TCGA-STAD un test CPTAC di un altro organo.

Copertura effettiva locale, audit 2026-09-16 (`scripts/data/build_eagle_benchmark_labels.py`
per TCGA, cBioPortal `coadread_tcga_pan_can_atlas_2018`; `pathobench_v1/splits/` per il resto):

| Dominio | Endpoint | TCGA (dev) | CPTAC (test) | Stato |
| --- | --- | --- | --- | --- |
| CRC | MSI | COAD/READ `msi_status` | `cptac_coad/MSI_H` | replica |
| CRC | BRAF | COAD/READ `braf_mutation` | — | replica (solo dev) |
| CRC | KRAS | COAD/READ `kras_mutation` | `cptac_coad/KRAS_mutation` | replica |
| CRC | sidedness | COAD/READ `sidedness` (derivato da `ICD_O_3_SITE`, risolto 2026-09-16) | — | adattamento_pubblico |
| CRC | N, M | COAD/READ `n_status`/`m_status` | — | replica (solo dev) |
| CRC | CIMP | — | — | non_riproducibile: nessuna fonte strutturata pubblica |
| Polmone | LUAD/LUSC subtype | `nsclc_subtyping` | — | replica |
| Polmone | EGFR | LUAD/LUSC `egfr_mutation` | `cptac_luad/EGFR_mutation` | replica |
| Polmone | STK11 | LUAD/LUSC `stk11_mutation` | `cptac_luad/STK11_mutation` | replica |
| Polmone | TP53 | LUAD/LUSC `tp53_mutation` | `cptac_luad/TP53_mutation` | replica |
| Polmone | KRAS | — (fuori scope dev) | `cptac_luad/KRAS_mutation` | replica (solo test) |
| Polmone | N, M | LUAD/LUSC `n_status`/`m_status` | — | replica (solo dev) |
| Mammella | PIK3CA | BRCA `pik3ca_mutation` | `cptac_brca/PIK3CA_mutation` | replica |
| Mammella | TP53 | — (fuori scope dev) | `cptac_brca/TP53_mutation` | replica (solo test) |
| Mammella | ER, PR, HER2 | — (non in PanCanAtlas clinical fields) | `bcnb/{er,pr,her2}`, `bc_therapy/{er_status,her2_status}` | adattamento_pubblico (coorte non-TCGA/CPTAC) |
| Mammella | N, M | BRCA `n_status`/`m_status` | — | replica (solo dev) |
| Stomaco | MSI | STAD `msi_status` | — | replica (solo dev) |
| Stomaco | TP53 | STAD `tp53_mutation` | — | replica (solo dev) |
| Stomaco | N, M | STAD `n_status`/`m_status` | — | replica (solo dev) |
| Stomaco | Lauren, EBV | — | — | non_riproducibile: solo in tabelle supplementari del paper, fuori scope API |

Verificato anche il pannello E02 (survival/trattamento): tutti e 12 gli endpoint
del piano (BOEHMK/PFS, SURGEN/OS via alias `sr386_`, CPTAC-LUAD/HNSC/PDA/CCRCC
OS, MBC OS+RECIST, SURGEN mortalità 5 anni, POST-NAT-BRCA invasione
linfovascolare, NADT-Prostate risposta, OV-Bevacizumab risposta) sono presenti
in `pathobench_v1/splits/`; copertura 12/12.

Cinque partizioni di sviluppo TCGA, training/validation disgiunti per paziente;
test CPTAC intatto. Head MLP del protocollo EAGLE e linear probe controllato;
tile baseline con mean pooling, ABMIL, gated ABMIL e STAMP. I dieci tile-FM e
gli otto WSI-FM entrano nel benchmark, con le rispettive combinazioni native.

Prima del fit: controllare stato di profiling molecolare; un paziente assente
dall'elenco mutazioni non è automaticamente wild-type. Registrare metodo/valore
di MSI e definizione binaria; non equiparare senza verifica MSIsensor, PCR e IHC.
N/M sono stato patologico, non endpoint di sopravvivenza.

### E02 — Survival e risposta/valutazione del trattamento

Questi dodici endpoint sono il pannello di replica pianificato; verificare
immagini e label attraverso [PathoBench](https://huggingface.co/datasets/MahmoodLab/Patho-Bench),
non soltanto i file di split locali:

| Dataset | Endpoint | Alias locale osservato / atteso |
| --- | --- | --- |
| BOEHMK | PFS | `boehmk_/PFS` |
| SURGEN | OS | Da risolvere dal rilascio, non inventare l'alias |
| CPTAC-LUAD | OS | `cptac_luad/OS` |
| CPTAC-HNSC | OS | `cptac_hnsc/OS` |
| CPTAC-PDA | OS | `cptac_pda/OS` |
| CPTAC-CCRCC | OS | `cptac_ccrcc/OS` |
| MBC | OS | `mbc_/OS` |
| SURGEN | Mortalità a 5 anni | Da risolvere dal rilascio |
| MBC | RECIST | `mbc_/Recist` |
| POST-NAT-BRCA | Invasione linfovascolare | `natbrca/lymphovascular_invasion` |
| NADT-Prostate | Risposta | `nadt/response` |
| OV-Bevacizumab | Risposta | `ovarian/response` |

Questa lista implementa il pannello concordato; alcuni endpoint sono valutazioni
di malattia, non risposta farmacologica. Non inferire beneficio del trattamento
da una semplice associazione prognostica o da una singola coorte trattata.

Seguire gli split ufficiali: cinque fold per survival e SURGEN mortalità;
cinquanta split Monte Carlo per gli altri quattro endpoint classificativi se
previsti dalla release. Validation interna al training, raggruppata per paziente.
CoxNet con alpha `0.01, 0.05, 0.1` e L1 ratio `0.1, 0.5, 0.9`; classificatore
con C `0.001, 0.01, 0.1, 0.5, 1, 10`. Selezione su validation.

Output: C-index con IC, numero di eventi, censoring e orizzonte; integrated
Brier score solo negli intervalli supportati con distribuzione di censoring
stimata sul training. Curve Kaplan–Meier con soglia di rischio dal training.
Confronto supplementare con covariate cliniche comuni disponibili, mancanti
gestiti nel training. Non chiamare esterna la CV PathoBench.

### E03 — Pochi dati

Few-shot `k=1,2,4,8,16,32` pazienti per classe, dieci support set condivisi tra
metodi. Training ridotto a `75,150,300` pazienti solo su task sufficientemente
numerosi; validation e test non si riducono insieme al training. Congelare i
task in base a disponibilità e numerosità, non scegliere i migliori sul test.
Output: curve qualità–numero di label, dispersione dei support set e costo.

### E04 — Ablation e attribuzione dei guadagni

- Budget EAGLE `5,10,25,50,100` tile; media semplice/pesata e cento selezioni
  casuali per budget sullo stesso sviluppo/test. Monte Carlo sul controllo
  casuale; non selezionare la replica casuale migliore per il confronto.
- EAF: retention e layer della sezione 1; full, pruning casuale, attenzione
  precoce, forecaster, forecaster senza distillazione, LoRA senza pruning.
- EAF per cinque backbone; controllo EAGLE con backbone condiviso, distinto
  dal sistema originale. Aggregatori mean, ABMIL, gated ABMIL, STAMP.
- Preprocessing nativo come principale; risoluzioni nominali `0.5,1.14,2 MPP`
  come ablation con FOV/resize effettivi espliciti. Nessuna scelta della
  risoluzione migliore in base al test esterno.
- Early/late fusion delle WSI: per TITAN/GigaPath baseline media degli embedding
  slide; non concatenare coordinate locali di slide diverse senza identificare
  i rispettivi sistemi di riferimento. Applicare early fusion solo ai modelli
  che la supportano, riportando l'asimmetria.
- Configurazioni standard e tuning interno con budget uguale; reporting anche
  dei delta sul test, senza ritoccare la griglia.
- Controlli content-blind e senza ALiBi già presenti come analisi TITAN, per
  separare contenuto e geometria. Il coarsening morfologico già in repo rimane
  analisi ausiliaria; non è una nuova componente EAF.

E01 include tutti i comparatori. Per contenere il prodotto cartesiano, gli
sweep meccanicistici nuovi si sviluppano su CRC-MSI, NSCLC e BRCA-ER dove
eleggibili; dichiarare uno skip anziché scegliere un task favorevole. Confermare
le configurazioni congelate sull'intero pannello finale.

### E05 — Attenzione, regioni e artefatti

Lorenz/Gini, quota necessaria per il 50%/80% della massa, top-k mass, ranking
forecaster/teacher e mappe alla stessa scala su casi determinati prima del test.
Tracciare token mantenuti con coordinate e layer. Attenzione non equivale a
spiegazione causale né a importanza clinica.

Con annotazioni pubbliche misurare presenza/area di lesioni o artefatti tra le
regioni selezionate. Se mancano annotazioni di artefatti, dichiarare quel braccio
non riproducibile: un detector automatico non è ground truth patologica.
Stress test di stain/focus separati dalle osservazioni su artefatti reali.
Nessuna dichiarazione di reader study senza lettori.

### E06 — Efficienza

Separare: lettura/decodifica, segmentazione/coordinate, selezione, codifica tile,
aggregazione WSI, head e trasferimenti CPU/GPU. Misurare raw-WSI-to-prediction
e inferenza pura, distinguendo cache calda/fredda e preprocessing preesistente.
Sincronizzare CUDA; warm-up escluso e documentato; dieci ripetizioni per slide
sul campione di sviluppo fissato. Riportare mediana/p95, throughput, memoria,
FLOPs con convenzione dichiarata e tassi OOM.

Confronti sia sulla stessa macchina sia con la configurazione nativa di ciascun
metodo. Le misure temporali pubblicate altrove non sono misure locali. Campione
profiling stratificato per numero di tile, organo e preparazione; usarlo per
stimare il costo dell'intero pannello prima degli sweep.

Contabilizzare separatamente cache teacher, training EAF e inferenza; calcolare
il volume di utilizzo necessario ad ammortizzare il costo iniziale, quando il
risparmio per WSI è positivo. Mostrare qualità–tempo e sensibilità–tempo per B01.

### E07 — Screening molecolare ampio

Un embedding per paziente, molte head su alterazioni misurate in TCGA/CPTAC.
Congelare l'intersezione dei pannelli dopo audit ma prima del fit; almeno dieci
casi per classe in training/test. Medesimi split per metodi, FDR su tutto il
pannello, risultati negativi inclusi. Riportare sia screening interno sia replica
esterna. Non promettere lo stesso numero di biomarcatori della coorte privata.
Misurare tempo fino a tutte le predizioni e costo marginale di aggiungere un task.

### E08 — Rappresentazioni e retrieval

Embedding L2-normalizzati, similarità coseno, query/gallery paziente-disgiunte;
escludere altre sezioni dello stesso caso dalla gallery. Recall@1/5/10 e mAP
per diagnosi e sottotipo dove disponibili; risultati per centro e cross-coorte.
UMAP descrittiva con seed e parametri congelati, non prova quantitativa di
separabilità. Esempi scelti tramite regola riproducibile, inclusi fallimenti.

### E09 — Visione–linguaggio

Task pubblici candidati: MSI CRC, sottotipo NSCLC, ER mammella. Usare gli stessi
support set few-shot di E03, con `k=2` per la replica mirata. Congelare prompt,
ordine esempi, modalità thumbnail/top-tile, seed support e modello/snapshot.

Il braccio GPT-4o storico è condizionato all'accessibilità dello snapshot e al
budget API; l'esecuzione non è avvenuta. Se manca lo snapshot, registrare replica
non disponibile, senza attribuire a una versione diversa equivalenza storica.
PRISM2 yes/no o multiple choice è un braccio distinto e contemporaneo: riceve
input nativi, non è un sostituto diretto del protocollo thumbnail di GPT-4o.
Reportare errori di parsing, output non validi, costo e copertura.

## 3. Esperimenti biomedicali aggiuntivi

### B01 — CAMELYON: segnali focali rari

CAMELYON16 principale, CAMELYON17 con label pubbliche per trasferimento tra
centri. Verificare sovrapposizioni fra release e patient ID; nessun riuso di
immagini di training come validazione esterna. Lesioni e dimensioni dalle
annotazioni ufficiali; non convertire automaticamente un tile positivo in
una diagnosi di micrometastasi. Riportare ITC separatamente quando annotati.

Endpoint primario: sensibilità sulle WSI con micrometastasi, soglia fissata
su validation per specificità nominale 95%; riportare la specificità osservata
sul test. Secondari: AUROC/AUPRC, falsi negativi, sensibilità per dimensione e
frazione tumorale, differenza appaiata con EAGLE/full a costi misurati.

Il claim di vantaggio richiede IC della differenza di sensibilità coerente con
superiorità nel confronto prespecificato. Stimare prima la precisione ottenibile
dal numero di positivi e dalla discordanza attesa su sviluppo; se insufficiente,
considerare il risultato esplorativo. Nessun margine clinico inventato dopo il test.
Una maggiore copertura iniziale EAF è strutturale: il risultato deve essere
la predizione, non la sola intersezione dei token con la lesione.

### B02 — HEST: conservazione di informazione spaziale

Solo campioni umani tumorali, donatori/studi disgiunti dal training EAF. Se i
checkpoint esistenti hanno visto questi campioni, escluderli oppure creare
nuovi training su manifest disgiunti; non riutilizzare il risultato contaminato.
Per i nuovi training del paper preferire HISTAI/GTEx escludendo HEST interamente
finché la disgiunzione più fine non sia verificata. Preservare gli asset originali.

Tile-EAF: stessi spot, full/compressed, regressore ridge; selezionare alpha
`0.1,1,10,100` su validation per donatore. Fissare geni secondo il benchmark
HEST e programmi di proliferazione, interferone, immunità e matrice extracellulare
da una release pubblica annotata, con liste versionate prima del test.
Normalizzazione espressione e preprocessing del benchmark, documentati.

WSI-EAF: domini definiti dalla trascrittomica indipendente, rappresentazione
dei domini minoritari, bordi e diversità tra regioni mantenute. Analisi separate
dalla predizione spot-level. Per ogni paziente: correlazione/errore, copertura
dei domini e delta full–EAF, poi aggregazione tra pazienti/studi.

EAGLE può essere confrontato per copertura dei domini delle regioni selezionate.
Non assegnare espressione zero agli spot omessi, né confrontare un embedding
globale con una mappa densa come se risolvessero lo stesso task. La correlazione
con programmi biologici non dimostra un meccanismo causale.

### B03 — BRACS: eterogeneità e sottostima

Classificazione a sette classi e coarse grouping ufficiale; split paziente con
ROI collegate alle WSI. Metriche multiclass, confusione e sottostima prespecificata
tra gruppi benigno/atipico/maligno; analisi distinta DCIS/invasivo. Non imporre
un ordine clinico totale arbitrario a tutte le sette classi.
Studiare errori nelle WSI con più categorie ROI annotate, senza trattare le ROI
come annotazioni esaustive o interpretare regioni non annotate come negative.

### B04 — MSI: approfondimento clinico esterno

Sviluppo TCGA-COAD/READ, test CPTAC-COAD bloccato; PAIP2020 replica aggiuntiva
solo dopo verifica di accessibilità delle label. Il config locale CPTAC MSI_H
riporta 93 casi: contare quelli effettivamente utilizzabili e i positivi prima
di formulare claim di potenza. Non addestrare sulle partizioni interne CPTAC
e poi descriverle come test esterno TCGA.

AUROC/AUPRC, Brier, calibrazione e falsi negativi; soglia dal validation set per
sensibilità nominale 95%, riportando sensibilità/specificità effettive e quota
classificata negativa sul test. Valutazione di potenziale preselezione, non
dimostrazione che si possano omettere test molecolari in clinica.

## 4. Figure e interpretazione

Main paper: metodo/costo, benchmark pubblico EAGLE aggiornato, outcome/pochi dati,
piccoli focolai, conservazione molecolare e validazione esterna. Supplemento:
grid completo, controlli, timing dettagliati, label mapping, fallimenti e tutte
le sostituzioni di coorte. L'ordine finale dipende dalla forza delle evidenze,
non dalla selezione post hoc dei task favorevoli.

La conclusione deve distinguere efficienza, compatibilità con il teacher,
utilità downstream e conservazione di segnali critici. Nessun beneficio sui
pazienti, robustezza universale o sicurezza clinica è dimostrato da queste
sole analisi retrospettive.
