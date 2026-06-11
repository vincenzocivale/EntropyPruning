# Metodo del paper: Early Attention Forecasting for Efficient ViT Tile Encoding in Histopathology

**Paper:** *Early Attention Forecasting for Efficient ViT Tile Encoding in Histopathology*  
**Submission:** Anonymous ECCV 2026, Paper ID #14105  
**Metodo principale:** Early Attention Forecaster (EAF) pruning  
**Dominio:** istopatologia digitale, classificazione tile-level, Vision Transformer (ViT), token pruning.

---

## 1. Idea centrale del metodo

Il paper propone **Early Attention Forecaster (EAF) pruning**, un metodo per accelerare l'encoding di tile istopatologiche con Vision Transformer.

Nei workflow su Whole Slide Images (WSI), ogni slide viene suddivisa in molte tile. Ogni tile viene poi codificata da un backbone ViT, spesso un foundation model istopatologico. Questo passaggio è costoso perché il ViT processa molti patch token per ogni tile. Il metodo EAF riduce questo costo **dentro il tile encoder**, non solo nello stadio successivo di aggregazione slide-level.

L'idea è:

1. eseguire i primi blocchi del ViT con tutti i token;
2. usare un piccolo modulo, l'**Early Attention Forecaster**, collegato a un blocco iniziale;
3. predire quali patch saranno importanti secondo l'attenzione finale del ViT;
4. mantenere solo i `k` token più importanti;
5. far processare ai blocchi successivi una sequenza molto più corta.

In breve: **EAF predice presto l'importanza che normalmente emergerebbe solo tardi nel ViT**.

---

## 2. Problema affrontato

Una pipeline WSI standard può essere descritta così:

1. **Tile encoding:** ogni tile viene trasformata in un vettore di feature tramite un ViT.
2. **Slide aggregation:** le feature delle tile vengono aggregate per produrre una predizione slide-level.

Molti metodi precedenti riducono il costo nella fase di aggregazione, per esempio selezionando tile informative. Tuttavia, il costo maggiore rimane spesso l'encoding delle tile: una singola slide può produrre centinaia o migliaia di tile, ciascuna processata separatamente dal backbone.

Il metodo parte da un'osservazione patologica: l'evidenza diagnostica è spesso concentrata in una piccola frazione del tessuto. Quindi non è necessario processare uniformemente tutte le patch di una tile.

---

## 3. Glossario e notazione

| Simbolo / termine | Significato |
|---|---|
| `WSI` | Whole Slide Image, immagine istologica completa ad alta risoluzione. |
| `tile` | Crop locale estratto dalla WSI, ad esempio `224 x 224`. |
| `patch` | Sottoregione della tile usata dal ViT, ad esempio `16 x 16`. |
| `patch token` | Embedding di una patch. |
| `[CLS]` | Token globale del ViT usato per la classificazione. |
| `N_p` | Numero di patch token nella tile. Nel paper, prima del pruning viene indicato `N_p = 196`. |
| `L` | Numero di blocchi Transformer del ViT. Nel paper/figura il backbone UNI ha 24 layer. |
| `k` | Token budget: numero di patch token da mantenere dopo il pruning. |
| `ATR` | Active Token Ratio: percentuale di patch token mantenuti rispetto al baseline senza riduzione. |
| `l_s` | Blocco sorgente da cui EAF prende gli embedding iniziali. Nel paper: `l_s = 2`. |
| `l_t` | Blocco target da cui si ricava il segnale teacher. Nel paper: `l_t = L`, cioè il blocco finale. |
| `a*` | Distribuzione teacher di importanza delle patch, ottenuta dall'attenzione finale `[CLS]`-to-patch. |
| `â` | Distribuzione predetta da EAF. |

---

## 4. Pipeline generale EAF

Il metodo lavora sul ViT tile encoder.

### 4.1 Forward senza pruning

Una tile viene divisa in patch, convertita in token e processata da tutti i blocchi Transformer:

```text
Tile -> patch tokens + [CLS] -> Transformer blocks 1...L -> final [CLS] -> classifier
```

### 4.2 Forward con EAF pruning

Con EAF, la sequenza viene accorciata dopo il blocco 2:

```text
Tile
  -> patch tokens + [CLS]
  -> Transformer block 1
  -> Transformer block 2
  -> EAF predice importanza patch
  -> top-k pruning: conserva k patch + [CLS]
  -> Transformer blocks 3...L su sequenza corta
  -> final [CLS]
  -> classifier
```

I blocchi 1 e 2 processano tutti i token. I blocchi successivi processano solo `k + 1` token, includendo sempre `[CLS]`.

---

## 5. Perché predire l'attenzione finale da un layer iniziale

Il paper evidenzia un problema centrale: l'importanza affidabile dei token emerge soprattutto negli ultimi layer del ViT. In particolare, l'attenzione `[CLS]`-to-patch del blocco finale è più allineata al task e viene usata come proxy di attribuzione.

Tuttavia, se bisogna arrivare al blocco finale per sapere quali token sono importanti, non si può più fare pruning precoce. EAF risolve questo contrasto formulando il problema come **forecasting**:

> dato l'embedding delle patch dopo il blocco 2, predire la distribuzione di attenzione `[CLS]`-to-patch che il ViT avrebbe prodotto nel blocco finale.

La scelta del blocco 2 come sorgente è motivata dal fatto che:

- il blocco 1 è ancora troppo vicino a cue locali di patch;
- il blocco 2 ha già una prima contestualizzazione globale;
- il blocco 2 è ancora abbastanza presto da consentire grandi risparmi computazionali.

La scelta del blocco finale come target è motivata dal fatto che l'attenzione diventa più strutturata e task-aligned nei layer profondi.

---

## 6. Stage 1 - Adattamento del foundation model

### Obiettivo

Rendere il segnale di attenzione finale task-relevant prima di usarlo come teacher per EAF.

### Procedura

1. Si parte dal foundation model istopatologico **UNI**.
2. Si aggiunge una classification head al `[CLS]` finale.
3. Si adatta il modello al dataset target con supervised training.
4. L'adattamento è parameter-efficient tramite **LoRA**.
5. La loss usata è la cross-entropy.

### Perché serve

UNI è pretrained su dati istopatologici ampi, ma le sue rappresentazioni e attenzioni devono essere allineate alle classi specifiche del downstream task. Dopo l'adattamento, l'attenzione finale `[CLS]`-to-patch diventa un teacher più utile per addestrare EAF.

### Dettagli implementativi riportati

- Backbone: **UNI**.
- LoRA inserito in:
  - proiezioni QKV;
  - output projections;
  - MLP layers di tutti i blocchi Transformer.
- Parametri LoRA:
  - rank `r = 8`;
  - `alpha = 32`.
- Classification head:
  - LayerNorm;
  - Dropout `p = 0.1`;
  - linear layer.
- Ottimizzatore: AdamW.
- Learning rate:
  - backbone: `1e-5`;
  - head: `1e-3`.
- Scheduler: cosine schedule con 100 warm-up step.
- Loss: cross-entropy con label smoothing `epsilon = 0.1`.
- Checkpoint selection: miglior validation accuracy.

---

## 7. Stage 2 - Training dell'Early Attention Forecaster

### Obiettivo

Addestrare EAF a predire, da rappresentazioni precoci, la distribuzione di importanza delle patch osservata nel blocco finale.

### Input e target

- Input EAF: patch embeddings al blocco sorgente `l_s = 2`, escluso `[CLS]`.
- Target teacher: attenzione `[CLS]`-to-patch del blocco finale `l_t = L`.
- Encoder UNI adattato: congelato durante questo stage.

### Definizione del target di importanza

Sia:

```math
A^{(l_t)} \in \mathbb{R}^{H \times (N_p + 1) \times (N_p + 1)}
```

la matrice di self-attention del blocco target `l_t`, con `H` attention heads.

L'importanza teacher della patch `i` è la media sulle teste dell'attenzione dal token `[CLS]` verso la patch `i`:

```math
a_i^* = \frac{1}{H} \sum_{h=1}^{H} A^{(l_t)}_{h, CLS \rightarrow i}, \quad i = 1, ..., N_p
```

Poiché la self-attention è normalizzata su patch + `[CLS]`, dopo aver rimosso il termine `[CLS]` i pesi patch non sommano necessariamente a 1. Per questo il paper applica una normalizzazione `L1` su `a*`, così da ottenere una distribuzione di probabilità sulle patch.

### Output di EAF

EAF produce:

```math
\hat{a} \in \Delta^{N_p}
```

cioè una distribuzione di probabilità sulle patch, ottenuta con softmax.

### Loss di distillazione

EAF viene addestrato minimizzando la divergenza KL tra teacher e predizione:

```math
\mathcal{L}_{EAF} = D_{KL}(a^* \parallel \hat{a})
```

La scelta della KL è importante perché il pruning dipende dal ranking relativo dei token sotto un budget fisso `k`, non da score non calibrati.

### Architettura EAF

Il diagramma del paper mostra una struttura leggera:

1. input: patch embeddings del blocco 2;
2. proiezione lineare a dimensione ridotta, indicata in figura come `1024 -> 256`;
3. piccolo stack di self-attention sui patch token proiettati;
4. query apprendibile, simile a un token globale, che esegue cross-attention sui patch token raffinati;
5. MLP / score head che produce logit per patch;
6. softmax per ottenere la distribuzione `â`.

Il motivo per usare self-attention e cross-attention è combinare:

- interazioni locali e globali tra patch;
- contesto tile-level;
- scoring patch-wise più robusto rispetto all'attenzione shallow del `[CLS]`.

### Dettagli implementativi riportati

- Encoder task-adapted congelato.
- Loss: KL divergence come sopra.
- Ottimizzatore: AdamW.
- Learning rate: `1e-4`.
- Weight decay: `0.05`.
- Scheduler: cosine annealing.
- Durata: 30 epoche.
- Metrica monitorata: Spearman `rho`, come misura di accordo di ranking.
- Checkpoint selection: più bassa validation KL divergence.

---

## 8. Stage 3 - Pruning-aware fine-tuning

### Problema risolto

Dopo Stage 1 il ViT è stato addestrato con sequenze complete, cioè senza pruning. Se il pruning venisse applicato solo in inference, i blocchi successivi riceverebbero sequenze accorciate mai viste in training. Questo crea distribution shift.

### Procedura

1. Si congela il modulo EAF addestrato nello Stage 2.
2. Si applica il pruning dopo il blocco `l = 2`.
3. Si mantengono esattamente i `k` patch token con score EAF più alto.
4. Il token `[CLS]` viene sempre mantenuto.
5. I blocchi `l+1, ..., L` processano la sequenza ridotta.
6. Si aggiornano LoRA adapters e classification head.

### Top-k differenziabile in training

Il top-k è una selezione discreta. Per addestrare il modello in modo stabile, il paper usa:

- una **differentiable top-k relaxation** durante il training;
- **hard top-k** in inference.

Lo stesso budget `k` viene usato in fine-tuning e inference.

### Dettagli implementativi riportati

- EAF: congelato.
- Parametri aggiornati: LoRA adapters + classification head.
- Ottimizzatore: AdamW.
- Weight decay: `1e-2`.
- Learning rate:
  - backbone/adapters: `eta_b = 1e-4`;
  - head: `eta_h = 1e-3`.
- Scheduler: OneCycleLR.
- Durata: 20 epoche.
- Warm-up: 10%, seguito da cosine decay.
- Loss: cross-entropy con label smoothing `epsilon = 0.1`.
- Checkpoint selection: più alto validation F1 Macro.

---

## 9. Inference con EAF

Durante inference:

1. la tile viene tokenizzata in patch token + `[CLS]`;
2. i primi due blocchi Transformer sono eseguiti normalmente;
3. EAF predice lo score di importanza per ogni patch;
4. viene eseguita selezione hard top-k;
5. i patch token non selezionati sono rimossi;
6. i blocchi Transformer rimanenti processano solo `[CLS] + k token`;
7. la classification head predice la classe usando il `[CLS]` finale.

Pseudocodice:

```python
# x: tile istologica
# ViT: blocchi transformer B1...BL
# EAF: forecaster addestrato
# k: numero di patch token da mantenere

tokens = patch_embed(x)              # [CLS] + N_p patch token
tokens = B1(tokens)
tokens = B2(tokens)

patch_tokens = tokens[1:]            # esclude [CLS]
scores = EAF(patch_tokens)           # distribuzione â sulle patch
idx = top_k(scores, k)

tokens_pruned = concat(tokens[0], patch_tokens[idx])

for block in B3_to_BL:
    tokens_pruned = block(tokens_pruned)

y_pred = classifier(tokens_pruned[0])
```

---

## 10. Perché il metodo riduce il costo computazionale

Il costo della self-attention cresce quadraticamente con la lunghezza della sequenza:

```math
O(N^2)
```

Senza pruning, i blocchi Transformer profondi processano `N_p + 1` token. Con EAF, dopo il blocco 2 processano solo `k + 1` token. Poiché la maggior parte dei blocchi lavora sulla sequenza corta, il costo complessivo scende sensibilmente.

Questo è diverso dai metodi progressive pruning, che potano gradualmente e quindi costringono vari layer intermedi a lavorare ancora con molte patch. EAF concentra la selezione in un solo punto precoce.

---

## 11. Perché EAF è diverso da pruning e merging precedenti

### Rispetto al token pruning classico

Molti metodi usano attenzione `[CLS]`-to-patch come segnale diretto. Il problema è che l'attenzione dei layer iniziali è debole e poco allineata al task. EAF invece predice l'attenzione finale usando feature iniziali.

### Rispetto al progressive pruning

Il progressive pruning introduce scoring e pruning a più profondità. Questo può funzionare, ma:

- i layer prima dei punti di pruning processano ancora sequenze lunghe;
- l'overhead dei moduli di scoring si accumula;
- la selezione discreta multi-stage può introdurre complessità di training.

EAF usa una singola potatura precoce.

### Rispetto al token merging

Il token merging combina token simili. Può ridurre il costo, ma rompe la corrispondenza uno-a-uno tra token e posizione spaziale originale. In istopatologia questo è problematico perché l'evidenza diagnostica è localizzata e deve restare interpretabile.

EAF usa pruning puro: mantiene o scarta token, preservando la tracciabilità spaziale delle patch mantenute.

---

## 12. Dataset usati nella valutazione

### BreaKHis

- Benchmark di istopatologia per breast cancer.
- 7.909 tile.
- Estratte da 82 immagini microscopiche H&E.
- 8 classi:
  - benign: adenosis, fibroadenoma, phyllodes tumour, tubular adenoma;
  - malignant: ductal, lobular, mucinous, papillary carcinoma.
- Protocollo: classificazione a 8 classi.
- Split: 70/30 train-test a livello paziente, senza overlap tra pazienti.

### NCT-CRC-HE

- Benchmark di tessuto colorettale.
- 100.000 tile.
- Estratte da 86 slide H&E di colorectal cancer e tessuto normale adiacente.
- 9 classi:
  - ADI: adipose;
  - BACK: background;
  - DEB: debris;
  - LYM: lymphocytes;
  - MUC: mucus;
  - MUS: smooth muscle;
  - NORM: normal colon mucosa;
  - STR: cancer-associated stroma;
  - TUM: colorectal adenocarcinoma epithelium.
- Variante usata: non-colour-normalised.
- Test esterno: CRC-VAL-HE-7K, 7.180 tile da 50 pazienti addizionali.

---

## 13. Metriche di valutazione

### F1 Macro

Metrica principale di classificazione. Pesa tutte le classi allo stesso modo ed è adatta a dataset con class imbalance.

### Active Token Ratio (ATR)

Percentuale di patch token mantenuti rispetto al baseline senza riduzione.

Esempio: `ATR = 20%` significa che il modello mantiene circa il 20% dei patch token originali.

### FLOPs reduction / GFLOPs

Riduzione dei FLOPs dell'encoder rispetto a un forward completo con sequenza non ridotta.

### Throughput speedup

Guadagno pratico in tile processate al secondo, misurato sull'hardware target. Tiene conto anche di overhead non catturati dai FLOPs, come memory effects e costo di scoring/selezione.

---

## 14. Baseline confrontate

| Metodo | Tipo | Descrizione |
|---|---|---|
| Baseline | Nessuna riduzione | UNI completo senza token reduction. |
| ToMe | Merging training-free | Merge progressivo dei token più simili via bipartite matching. |
| MCTF | Merging training-free | Merge basato su similarità, informativeness e dimensione del token merged. |
| DynamicViT | Pruning con fine-tuning | Pruning progressivo con token-keep predictors. |
| EViT | Ibrido pruning + merging | Pruning top-k tramite attenzione `[CLS]`-to-patch e merge dei token rimossi in un summary token. |
| CropR | Pruning con fine-tuning | Scoring module con selezione differenziabile in training e hard top-k in inference. |
| EAF | Pruning con fine-tuning | Forecasting dell'attenzione finale da layer iniziale e pruning precoce. |

---

## 15. Setup sperimentale comune

- Backbone: UNI.
- Batch size: 32.
- Random seed: uguale per i metodi confrontati.
- Hardware: singola NVIDIA GeForce RTX 3080 Ti.
- Gestione class imbalance: weighted sampler.
- Training/evaluation protocol: mantenuto identico tra metodi per ciascun dataset.

---

## 16. Risultati principali

### Tabella riassuntiva dal paper

| Metodo | Tipo | BreaKHis F1 Macro (%) | NCT F1 Macro (%) | ATR (%) | GFLOPs | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | - | 98.7 | 78.8 | 100 | 60.3 | 1.00x |
| ToMe (`r=4`) | Merging | 98.7 | 78.0 | 51 | 46.6 | 1.02x |
| ToMe (`r=8`) | Merging | 97.2 | 76.1 | 2 | 31.3 | 1.46x |
| MCTF (`r=.20`) | Merging | 98.6 | 78.0 | 51 | 51.2 | 0.97x |
| MCTF (`r=.35`) | Merging | 98.4 | 77.3 | 27 | 44.1 | 1.10x |
| MCTF (`r=.50`) | Merging | 96.5 | 76.8 | 13 | 37.9 | 1.29x |
| DynamicViT | Pruning | 92.1 | 65.2 | 12 | 34.1 | 1.47x |
| EViT | Hybrid | 89.1 | 63.1 | 10 | 24.5 | 1.94x |
| CropR (`r=8`) | Pruning | 98.5 | 81.3 | 4 | 31.6 | 2.38x |
| **EAF (`k=30`)** | Pruning | **98.7** | **89.3** | 30 | 23.7 | 2.16x |
| **EAF (`k=20`)** | Pruning | **98.3** | **88.6** | 20 | 18.7 | 2.74x |

### Interpretazione dei risultati

- Su **BreaKHis**, EAF a `30% ATR` uguaglia il baseline con `98.7 F1 Macro`, riducendo il costo da `60.3` a `23.7 GFLOPs`.
- A `20% ATR`, EAF resta quasi al baseline: `98.3 F1 Macro`, `18.7 GFLOPs`, `2.74x` speedup.
- Su **NCT-CRC-HE**, EAF migliora il baseline: `89.3 F1 Macro` a `30% ATR` contro `78.8` del baseline.
- EAF domina la Pareto frontier F1 Macro vs GFLOPs su entrambi i dataset.
- Il paper riporta anche fino a `3.53x` speedup con pruning più aggressivo, mantenendo una perdita di accuratezza contenuta.

---

## 17. Analisi della strategia di selezione token

Il paper confronta tre strategie:

1. selezione casuale;
2. attenzione `[CLS]`-to-patch del layer 2;
3. EAF.

Risultato:

- EAF mantiene performance quasi baseline fino a circa `20%` di token mantenuti;
- random e layer-2 attention degradano molto prima;
- l'attenzione shallow è un proxy debole dell'evidenza task-relevant;
- predire l'importanza finale produce un ranking dei token più stabile.

Il risultato supporta la tesi principale: **non basta usare attenzione precoce; bisogna prevedere l'attenzione task-aligned dei layer finali**.

---

## 18. Analisi per classe

Il paper mostra che il pruning non danneggia uniformemente tutte le classi.

Osservazioni principali:

- le degradazioni più forti compaiono soprattutto a `ATR = 10%`;
- le classi con alta variabilità intra-classe o contenuto visivamente cluttered sono più sensibili;
- nel regime `20-30% ATR`, la F1 per classe resta molto vicina ai regimi con più token;
- su NCT-CRC-HE alcune classi migliorano con il pruning, ad esempio:
  - smooth muscle (MUS);
  - lymphocytes (LYM);
  - tumour epithelium (TUM);
  - stroma (STR).

Interpretazione del paper: il pruning può agire come regolarizzazione, rimuovendo token poco informativi e facendo dipendere il `[CLS]` da evidenza morfologica più discriminativa.

---

## 19. Interpretabilità

EAF non produce solo efficienza, ma anche mappe di attribuzione token-level.

### 19.1 Attention forecasting fidelity

Il paper visualizza:

1. tile H&E originale;
2. attenzione `[CLS]`-to-patch del layer 2;
3. attenzione target del layer finale;
4. distribuzione forecasted da EAF.

Le mappe EAF seguono da vicino le mappe target finali. Alcune correlazioni Spearman riportate negli esempi:

| Dataset | Classe | Spearman `rho` tra EAF e target finale |
|---|---|---:|
| BreaKHis | Adenosis | 0.950 |
| BreaKHis | Lobular Carcinoma | 0.939 |
| BreaKHis | Ductal Carcinoma | 0.937 |
| NCT-CRC-HE | ADI | 0.972 |
| NCT-CRC-HE | TUM | 0.961 |
| NCT-CRC-HE | MUC | 0.926 |

### 19.2 Confronto con Gradient x Input

Il paper confronta EAF con Gradient x Input, un metodo post-hoc di saliency.

Differenza principale:

- Gradient x Input richiede full forward + backward pass;
- EAF produce mappe comparabili dopo solo i primi due blocchi Transformer e senza backward.

Quindi EAF fornisce interpretabilità a costo aggiuntivo trascurabile, insieme alla selezione efficiente dei token.

---

## 20. Vantaggi del metodo

1. **Riduce il collo di bottiglia corretto:** agisce sul tile encoder, non solo sull'aggregatore slide-level.
2. **Pruning precoce:** accorcia la sequenza prima dei blocchi più costosi.
3. **Segnale più affidabile:** predice l'attenzione finale, più allineata al task, invece di usare attenzione shallow.
4. **Budget fisso:** il top-k garantisce costo prevedibile per tile.
5. **Preserva la tracciabilità spaziale:** a differenza del token merging, ogni token mantenuto corrisponde a una patch originale.
6. **Compatibile con pipeline WSI:** può essere combinato con metodi di selezione tile o compressione slide-level.
7. **Interpretabilità integrata:** gli score EAF generano evidence map token-level senza un passaggio post-hoc separato.

---

## 21. Limiti dichiarati

Il paper indica tre limiti principali:

1. **Training multi-stage:** EAF richiede più fasi di addestramento, quindi ha un costo upfront maggiore rispetto a metodi training-free.
2. **Speedup marginale a keep ratio alti:** se si rimuovono pochi token, il guadagno pratico può essere limitato e dipendere dall'efficienza runtime dello scoring e della selezione.
3. **Valutazione tile-level:** il paper valuta classificazione a livello tile; l'integrazione completa in pipeline WSI con aggregazione slide-level e tile selection è lasciata a lavoro futuro.

---

## 22. Algoritmo completo di training

```python
# Stage 1: task adaptation
initialize UNI backbone
insert LoRA adapters into QKV, output projections, and MLP layers
attach classification head to final CLS
train on full token sequences with cross_entropy
save best checkpoint by validation accuracy

# Stage 2: EAF distillation
freeze task-adapted UNI
for each tile:
    run full forward
    collect X^(2) patch embeddings
    collect final-block CLS-to-patch attention
    average over heads
    remove CLS entry
    L1-normalize to get a_star
    predict a_hat = EAF(X^(2))
    minimize KL(a_star || a_hat)
save EAF checkpoint by validation KL / Spearman monitoring

# Stage 3: pruning-aware fine-tuning
freeze EAF
for each tile:
    run blocks 1 and 2
    score patch tokens with EAF
    select top-k patches with differentiable top-k relaxation
    keep CLS token
    run remaining blocks on shortened sequence
    classify using final CLS
    update LoRA adapters and classification head
save checkpoint by validation F1 Macro
```

---

## 23. Algoritmo completo di inference

```python
# Inference EAF
input tile x
patchify x -> N_p patch tokens
prepend CLS

run ViT blocks 1 and 2
patch_embeddings = tokens_without_CLS
importance = EAF(patch_embeddings)
selected_indices = top_k(importance, k)

short_sequence = CLS + selected patch tokens
run ViT blocks 3...L on short_sequence
prediction = classifier(final_CLS)

# optional interpretability
attribution_map = reshape(importance over patch grid)
selection_mask = binary mask from selected_indices
```

---

## 24. Informazioni minime per reimplementare il metodo

Per reimplementare EAF servono:

1. un ViT tile encoder, nel paper UNI;
2. dataset tile-level con label supervisionate;
3. capacità di estrarre attention maps dal blocco finale;
4. capacità di leggere patch embeddings dopo il blocco 2;
5. implementazione EAF:
   - proiezione degli embedding;
   - self-attention leggera;
   - learnable query con cross-attention;
   - MLP score head;
   - softmax sui patch score;
6. loss di distillazione KL;
7. top-k pruning dopo il blocco 2;
8. fine-tuning pruning-aware con differenziable top-k relaxation;
9. hard top-k in inference;
10. valutazione con F1 Macro, ATR, GFLOPs e throughput.

---

## 25. Nota su materiale supplementare

Il paper afferma che dettagli aggiuntivi, inclusi ablation su source/target layer, varianti architetturali EAF, loss alternative e impatto dello Stage 3, sono forniti nel materiale supplementare. Questi dettagli non sono inclusi nel PDF principale analizzato qui.
