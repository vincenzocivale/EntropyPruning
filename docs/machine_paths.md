# Path critici per macchina

---

# HAL (hostname `hal`) — server GPU principale

Macchina con GPU dedicata su cui girano i training correnti (branch `unsupervised-pruning`).

## Ambiente Python

| Voce | Path |
|---|---|
| Conda base | `/data2/home/vcivale/miniconda3/` |
| Environment | `eaf_env` |
| Attivazione | `conda activate eaf_env` |
| Esecuzione senza attivazione | `conda run --no-capture-output -n eaf_env python ...` |
| PyTorch installato | `2.5.1+cu124` |

## Sorgente codice

| Voce | Path |
|---|---|
| Progetto EAF (questo repo) | `/data2/home/vcivale/projects/imaging/EAF/` |
| Branch attivo | `unsupervised-pruning` |
| Thunder (installato in eaf_env) | incluso in `eaf_env` (non un repo separato) |

## Dati Thunder (`THUNDER_BASE_DATA_FOLDER`)

Impostare **sempre** questa variabile prima di ogni run:
```bash
export THUNDER_BASE_DATA_FOLDER=/data2/home/vcivale/projects/imaging/data/thunder-tiles
```

> **Importante**: `THUNDER_BASE_DATA_FOLDER` punta alla root (contenente `pretrained_ckpts/`),
> mentre `--base-data-folder` punta alla sottocartella `datasets/` (contenente `data_splits/`).
> Sono **due percorsi diversi** — non confonderli.

| Voce | Path |
|---|---|
| `THUNDER_BASE_DATA_FOLDER` (root) | `/data2/home/vcivale/projects/imaging/data/thunder-tiles/` |
| `--base-data-folder` (da passare agli script) | `/data2/home/vcivale/projects/imaging/data/thunder-tiles/datasets/` |
| Pesi modelli pre-addestrati | `/data2/home/vcivale/projects/imaging/data/thunder-tiles/pretrained_ckpts/` |
| Split JSON | `...datasets/data_splits/{dataset}.json` |
| Checkpoint EAF supervisato | `/data2/home/vcivale/projects/imaging/data/thunder-tiles/checkpoints/` |
| Cache e checkpoint non-supervisionato | `/data2/home/vcivale/projects/imaging/EAF/checkpoints/unsupervised/` |
| Log training | `/data2/home/vcivale/projects/imaging/EAF/logs/` |

## Dataset scaricati (15 su 16 — manca `mhist`)

```
bach  bracs  break_his  ccrcc  crc  esca  patch_camelyon
spider_breast  spider_colorectal  spider_skin  spider_thorax
tcga_crc_msi  tcga_tils  tcga_uniform  wilds
```

> **Nota SPIDER**: i dataset `spider_*` usano solo la sottocartella `center_crop/` (patch pre-elaborate).
> Le cartelle raw `SPIDER-{organ}/` (download HuggingFace originale, ~241 GB ciascuna) sono state
> **eliminate** — non servono alla pipeline.

## Checkpoint non-supervisionato (stato attuale)

| File | Descrizione |
|---|---|
| `checkpoints/unsupervised/*_uni_attn_features.h5` | Cache HDF5 per tutti i 15 dataset |
| `checkpoints/unsupervised/uni_forecaster/forecaster_uni_src02_attn23_universal.pt` | Modello universale (naming post-refactor: include model_name nel filename) |
| `checkpoints/unsupervised/per_dataset/{ds}/forecaster_src02_attn23.pt` | Modelli per-dataset |

> **Nota naming**: a partire dal commit `3e20bc1` il filename del forecaster universale include il nome
> dell'encoder: `forecaster_{model}_{src_tag}_attn{T:02d}_universal.pt`.
> I checkpoint precedenti al refactor usavano `forecaster_src{L:02d}_attn{T:02d}_universal.pt`.

## Risultati esperimenti

| File | Descrizione |
|---|---|
| `results/linear_probe_pruned/{model_name}.csv` | Metriche linear probe con EAF pruning (acc, F1, AUROC, TAR@FAR) |

## Comandi di avvio rapido (HAL)

```bash
cd /data2/home/vcivale/projects/imaging/EAF
export THUNDER_BASE_DATA_FOLDER=/data2/home/vcivale/projects/imaging/data/thunder-tiles
BASE_DATA=/data2/home/vcivale/projects/imaging/data/thunder-tiles/datasets

# Pipeline supervisionata (Fase 1 → 2 → 3) su un dataset
bash scripts/run_thunder_crc_tests.sh  # vedi run_experiment.md per la versione parametrica

# Pipeline non-supervisionata universale (build cache + train)
THUNDER_BASE_DATA_FOLDER=$THUNDER_BASE_DATA_FOLDER \
conda run --no-capture-output -n eaf_env python scripts/build_unsupervised_cache.py \
    --model-name uni --base-data-folder $BASE_DATA

THUNDER_BASE_DATA_FOLDER=$THUNDER_BASE_DATA_FOLDER \
conda run --no-capture-output -n eaf_env python scripts/train_forecaster_unsupervised.py \
    --model-name uni --base-data-folder $BASE_DATA --layers-source 2 --epochs 30 \
    --forecaster-dir checkpoints/unsupervised/uni_forecaster_h512 \
    --hidden 512 --wandb-project eaf-forecaster

# Pipeline non-supervisionata per-dataset (tutti i 15 in sequenza)
THUNDER_BASE_DATA_FOLDER=$THUNDER_BASE_DATA_FOLDER \
conda run --no-capture-output -n eaf_env python scripts/train_per_dataset_eaf.py
```

## Disco

| Mount | Note |
|---|---|
| `/data2/` | Disco principale dati e checkpoint; usare sempre questo per download grandi |

---

# Nanopore-PC (hostname `Nanopore-PC`)

Questa macchina si chiama **Nanopore-PC** (hostname `Nanopore-PC`).
Documento qui dove risiedono i file critici per la pipeline EAF.

---

## Ambiente Python

| Voce | Path |
|---|---|
| Virtual environment | `/home/oem/EAF/EntropyPruning/.venv/` |
| Attivazione | `source /home/oem/EAF/EntropyPruning/.venv/bin/activate` |
| PyTorch installato | `2.5.1+cu121` (compatibile con driver CUDA 12.2 / driver 535.x) |

> **Nota CUDA**: il driver NVIDIA (535.230.02) supporta CUDA ≤ 12.2.
> Non installare `torch>=2.6.0` (richiedono cu124/cu126 che richiedono driver ≥ 550).
> Usare sempre `--index-url https://download.pytorch.org/whl/cu121`.

---

## Sorgente codice

| Voce | Path |
|---|---|
| Progetto EAF (questo repo) | `/home/oem/EAF/EntropyPruning/` |
| Branch attivo | `thunder-integration` |
| Thunder (sorgente editable) | `/home/oem/EAF/thunder/` |

---

## Dati Thunder (`THUNDER_BASE_DATA_FOLDER`)

Variabile d'ambiente da impostare prima di ogni run:
```bash
export THUNDER_BASE_DATA_FOLDER=/data/EAF_data/thunder
```

| Voce | Path |
|---|---|
| Base data folder (da passare a `--base-data-folder`) | `/data/EAF_data/thunder/datasets/` |
| Pesi modelli pre-addestrati | `/data/EAF_data/thunder/pretrained_ckpts/` |
| Dataset CRC scaricato | `/data/EAF_data/thunder/datasets/crc/` |
| Split JSON CRC | `/data/EAF_data/thunder/datasets/data_splits/crc.json` |
| Checkpoint EAF (nuovi) | `/home/oem/EAF/EntropyPruning/checkpoints/` (creato a runtime) |

> **Nota**: Thunder `download-datasets` mette sia le immagini che i data_splits dentro `datasets/`.
> Passare sempre `--base-data-folder /data/EAF_data/thunder/datasets` (non `.../thunder`).

---

## Pesi modelli fondazionali

| Modello | Path effettivo | Symlink Thunder |
|---|---|---|
| UNI (MahmoodLab/UNI) | `/home/oem/.cache/huggingface/hub/models--MahmoodLab--uni/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin` | `/data/EAF_data/thunder/pretrained_ckpts/uni/pytorch_model.bin → (symlink)` |

Per altri modelli che richiedono download manuale da HuggingFace, creare la struttura:
```
/data/EAF_data/thunder/pretrained_ckpts/{model_name}/
```

---

## Dati legacy (vecchia pipeline, pre-Thunder integration)

| Voce | Path |
|---|---|
| Dataset NCT-CRC-HE (HuggingFace Arrow) | `/data/EAF_data/NCT-CRC-HE/` |
| Dataset BREAKHIS (HuggingFace Arrow) | `/data/EAF_data/BREAKHIS/` |
| Checkpoint vecchia pipeline | `/data/EAF_data/checkpoints-Attention-Pruning/` |

I dati legacy sono in formato HuggingFace Arrow (`.arrow`), **non** nel formato Thunder (JSON+immagini).
Non sono direttamente usabili con `build_thunder_loaders`.

---

## Cache HuggingFace

| Voce | Path |
|---|---|
| Cache modelli HF | `/home/oem/.cache/huggingface/hub/` |

---

## Disco

| Mount | Dimensione | Utilizzo | Note |
|---|---|---|---|
| `/` (sda2) | 1.8 TB | ~98% | Quasi pieno — evitare scritture grandi |
| `/data` (sdb1) | 7.3 TB | ~57% | Usare per dataset e checkpoint |

> **Attenzione**: `/` è quasi pieno (98%). Tutti i download di dataset e checkpoint
> devono andare su `/data/EAF_data/thunder/` o `/data/EAF_data/`.

---

## Comandi di avvio rapido

```bash
# Attiva venv + imposta Thunder data folder
source /home/oem/EAF/EntropyPruning/.venv/bin/activate
export THUNDER_BASE_DATA_FOLDER=/data/EAF_data/thunder
cd /home/oem/EAF/EntropyPruning

# Fase 1 — Linear Probing su CRC con UNI
python scripts/train_classifier.py \
    --model-name uni \
    --dataset-name crc \
    --base-data-folder /data/EAF_data/thunder/datasets \
    --adaptation linear_probing \
    --epochs 20 \
    --batch-size 64 \
    --output-dir /data/EAF_data/thunder/checkpoints/crc/uni_linear_probing

# Fase 2 — AttentionForecaster
python scripts/train_forecaster.py \
    --model-name uni \
    --dataset-name crc \
    --base-data-folder /data/EAF_data/thunder/datasets \
    --adaptation linear_probing \
    --layers-source 2 \
    --output-dir /data/EAF_data/thunder/checkpoints/crc/uni_forecaster

# Fase 3 — Fine-tune pruned
python scripts/finetune_pruned.py \
    --model-name uni \
    --dataset-name crc \
    --base-data-folder /data/EAF_data/thunder/datasets \
    --prune-layer 2 \
    --keep-ratio 0.1 \
    --output-dir /data/EAF_data/thunder/checkpoints/crc/uni_pruned
```
