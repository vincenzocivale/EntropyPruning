"""
evaluate_uni_cropr.py
─────────────────────
Valutazione standalone del modello UNI + CropR addestrato.
Logga su Weights & Biases:
  - Accuracy, F1 macro, F1 per classe
  - TAR@FAR (default FAR=1e-4)
  - GFLOPs per immagine
  - Tempo di inferenza medio per immagine (ms)
  - Confusion matrix
  - Classification report completo come tabella

Uso:
    python evaluate_uni_cropr.py
"""

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
import timm
import wandb

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("⚠️  fvcore non trovato — GFLOPs non calcolati. "
          "Installa con: pip install fvcore")

sys.path.append(".")

from src.vision_transformer_copr import VisionTransformer as CroprVisionTransformer
from src.dataset import HistologicalImageDataset

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — adatta questi valori al tuo setup
# ──────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Dati
    data_dir         = "/data/NCT-CRC-HE/",
    img_size         = 224,
    batch_size       = 16,
    num_workers      = 4,

    # Checkpoint da valutare
    checkpoint       = "/data/checkpoints-Attention-Pruning/NCT-CRC-HE/uni_cropr/best_model.pt",

    # Backbone UNI
    patch_size       = 16,
    embed_dim        = 1024,
    depth            = 24,
    num_heads        = 16,
    mlp_ratio        = 4.0,
    init_values      = 1e-5,
    drop_path        = 0.2,
    global_pool      = "avg",

    # CropR — devono corrispondere a quelli usati in training
    use_cropr           = True,
    cropr_pruning_rate  = 8,
    cropr_llf           = False,
    cropr_num_queries   = 1,
    cropr_num_heads     = 1,
    cropr_pre_attn_norm = False,
    cropr_q_proj        = False,
    cropr_k_proj        = False,
    cropr_v_proj        = False,
    cropr_mlp           = True,
    cropr_mlp_ratio     = 4.0,

    # Metriche
    far_threshold    = 1e-4,
    n_warmup_batches = 10,

    # WandB
    wandb_project    = "uni-cropr",
    wandb_run_name   = "eval_NCTCRCHE_pr8",
)
CFG["dataset_name"] = Path(CFG["data_dir"]).name


# ──────────────────────────────────────────────────────────────────────────────
# MODELLO — identico alla definizione usata in training
# ──────────────────────────────────────────────────────────────────────────────
class UNICroprVisionTransformer(CroprVisionTransformer):
    def __init__(self, cropr_cfg, init_values: float = 1e-5, **kwargs):
        super().__init__(cropr_cfg, **kwargs)
        dpr = [x.item() for x in torch.linspace(0, kwargs["drop_path_rate"], len(self.blocks))]
        if cropr_cfg["use_cropr"]:
            dpr[-1] = 0.0
        for blk_idx in range(len(self.blocks)):
            self.blocks[blk_idx] = timm.models.vision_transformer.Block(
                dim=self.embed_dim,
                num_heads=kwargs["num_heads"],
                qkv_bias=True,
                init_values=init_values,
                drop_path=dpr[blk_idx],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )


def build_model(n_classes: int, cfg: dict) -> nn.Module:
    cropr_cfg = {
        "use_cropr"     : cfg["use_cropr"],
        "pruning_rate"  : cfg["cropr_pruning_rate"],
        "llf"           : cfg["cropr_llf"],
        "num_queries"   : cfg["cropr_num_queries"],
        "num_heads"     : cfg["cropr_num_heads"],
        "pre_attn_norm" : cfg["cropr_pre_attn_norm"],
        "q_proj"        : cfg["cropr_q_proj"],
        "k_proj"        : cfg["cropr_k_proj"],
        "v_proj"        : cfg["cropr_v_proj"],
        "mlp"           : cfg["cropr_mlp"],
        "mlp_ratio"     : cfg["cropr_mlp_ratio"],
        "training"      : True, 
    }
    model_kwargs = dict(
        num_classes    = n_classes,
        img_size       = cfg["img_size"],
        patch_size     = cfg["patch_size"],
        embed_dim      = cfg["embed_dim"],
        depth          = cfg["depth"],
        num_heads      = cfg["num_heads"],
        mlp_ratio      = cfg["mlp_ratio"],
        drop_path_rate = cfg["drop_path"],
        global_pool    = cfg["global_pool"],
        class_token    = True,
    )
    return UNICroprVisionTransformer(
        cropr_cfg, init_values=cfg["init_values"], **model_kwargs
    )


# ──────────────────────────────────────────────────────────────────────────────
# FUNZIONI DI VALUTAZIONE
# ──────────────────────────────────────────────────────────────────────────────
def compute_tar_at_far(scores, is_correct, far_threshold=1e-4):
    correct   = np.asarray(is_correct, dtype=bool)
    incorrect = ~correct
    if incorrect.sum() == 0:
        return 1.0, float("nan")
    n_far_allowed = max(1, int(np.ceil(incorrect.sum() * far_threshold)))
    sorted_wrong  = np.sort(scores[incorrect])[::-1]
    threshold     = sorted_wrong[min(n_far_allowed - 1, len(sorted_wrong) - 1)]
    tar = (scores[correct] >= threshold).mean()
    return float(tar), float(threshold)


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    for imgs, labels in tqdm(loader, desc="Inference", leave=False):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(imgs.to(device, non_blocking=True))
        probs  = logits.float().softmax(-1)
        all_preds.append(probs.argmax(-1).cpu())
        all_labels.append(labels)
        all_scores.append(probs.max(-1).values.cpu())
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_scores).numpy(),
    )


def compute_gflops(model, loader, device):
    if not HAS_FVCORE:
        return None
    dummy = next(iter(loader))[0][:1].to(device, dtype=torch.bfloat16)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fa = FlopCountAnalysis(model, dummy)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 1e9
    except Exception as e:
        print(f"⚠️  FLOPs non calcolabili: {e}")
        return None


def compute_ms_per_image(model, loader, device, n_warmup=10):
    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(imgs.to(device))
            if i >= n_warmup:
                break
    torch.cuda.synchronize()
    t0, n_imgs = time.perf_counter(), 0
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Timing", leave=False):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(imgs.to(device))
            torch.cuda.synchronize()
            n_imgs += len(imgs)
    return (time.perf_counter() - t0) / n_imgs * 1000


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──
    eval_tf = T.Compose([
        T.Resize((CFG["img_size"], CFG["img_size"])),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_ds = HistologicalImageDataset(f"{CFG['data_dir']}/test", transform=eval_tf)
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True,
    )
    CLASS_NAMES = test_ds.class_names
    N_CLASSES   = len(CLASS_NAMES)
    print(f"Classi ({N_CLASSES}): {CLASS_NAMES}")
    print(f"Test samples: {len(test_ds)}")

    # ── Modello ──
    model = build_model(N_CLASSES, CFG).to(device, dtype=torch.bfloat16)
    ckpt  = torch.load(CFG["checkpoint"], map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print(f"✅ Checkpoint caricato: {CFG['checkpoint']}")

    # ── Inference ──
    all_preds, all_labels, all_scores = run_inference(model, test_loader, device)

    # ── Metriche ──
    acc          = float((all_preds == all_labels).mean())
    f1_macro     = float(f1_score(all_labels, all_preds, average="macro",  zero_division=0))
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    report_dict  = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True
    )
    tar, thr = compute_tar_at_far(all_scores, all_preds == all_labels, CFG["far_threshold"])

    print("\nCalcolo GFLOPs...")
    gflops = compute_gflops(model, test_loader, device)

    print("Calcolo tempo di inferenza...")
    ms_per_img = compute_ms_per_image(model, test_loader, device, CFG["n_warmup_batches"])

    # ── Stampa ──
    print("\n" + "─" * 52)
    print(f"  Accuracy               : {acc:.4f}")
    print(f"  F1 macro               : {f1_macro:.4f}")
    print(f"  TAR@FAR={CFG['far_threshold']:.0e}        : {tar:.4f}  (thr={thr:.4f})")
    if gflops:
        print(f"  GFLOPs/img             : {gflops:.2f}")
    print(f"  ms/img                 : {ms_per_img:.2f}")
    print("─" * 52)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # ── Confusion matrix ──
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    fig_cm, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — UNI+CropR ({CFG['dataset_name']})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # ── WandB ──
    # ── WandB ──
    wandb.init(
        project = CFG["wandb_project"],
        name    = CFG["wandb_run_name"],
        config  = CFG,
        tags    = [CFG["dataset_name"], "eval", f"pr{CFG['cropr_pruning_rate']}"],
    )

    log_dict = {
        "test/accuracy"                               : acc,
        "test/f1_macro"                               : f1_macro,
        f"test/tar_at_far_{CFG['far_threshold']:.0e}" : tar,
        "test/ms_per_img"                             : ms_per_img,
    }
    if gflops:
        log_dict["test/gflops"] = gflops

    wandb.log(log_dict)
    wandb.log({"test/confusion_matrix": wandb.Image(fig_cm)})
    plt.close(fig_cm)

    wandb.summary.update({
        "test/accuracy"  : acc,
        "test/f1_macro"  : f1_macro,
        "test/tar_at_far": tar,
        "test/ms_per_img": ms_per_img,
        **({"test/gflops": gflops} if gflops else {}),
    })

    wandb.finish()
    print("\n✅ Valutazione completata e loggata su WandB.")


if __name__ == "__main__":
    main()
