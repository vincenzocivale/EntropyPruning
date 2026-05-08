# Path critici — Nanopore-PC

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
