# %%
import sys
import time
import copy
import json
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
from tome.patch import timm as tome_patch_timm
from peft import LoraConfig
from peft.tuners.lora import LoraModel

sys.path.append(".")
from src.dataset import HistologicalImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"timm   : {timm.__version__}")

# %% [markdown]
# ## 1 · Configurazione

# %%
CFG = dict(
    data_dir      = "/data/BREAKHIS",
    img_size      = 224,
    batch_size    = 32,
    num_workers   = 8,
    embed_dim     = 1024,
    patch_size    = 16,
    tome_r_values = [0, 4, 8, 16, 24, 32],
    seed          = 42,
)

CFG["dataset_name"] = Path(CFG["data_dir"]).name
CFG["num_patches"]  = (CFG["img_size"] // CFG["patch_size"]) ** 2

FINETUNED_CKPT = (Path("/data/checkpoints-Attention-Pruning")
                  / CFG["dataset_name"] / "uni_finetuned" / "best_model.pt")

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(f"Dataset    : {CFG['dataset_name']}")
print(f"Checkpoint : {FINETUNED_CKPT}")

# %% [markdown]
# ## 2 · Dati

# %%
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
print(f"Test: {len(test_ds)} immagini")

# %% [markdown]
# ## 3 · Architettura
#
# Con timm >= 0.9, init_values=1e-5 attiva LayerScale nativamente,
# eliminando tutto il patching manuale del forward dei blocchi.

# %%
class UNILoRAClassifier(nn.Module):
    """
    UNI con LoRA — identica al training di Stage 1.
    Con timm aggiornato, LayerScale è gestita internamente da timm.
    """
    def __init__(self, n_classes, dropout=0.1):
        super().__init__()
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained       = True,
            init_values      = 1e-5,
            dynamic_img_size = True,
            num_classes      = 0,
        )
        lora_config = LoraConfig(
            r              = 8,
            lora_alpha     = 32,
            target_modules = ["qkv", "proj", "fc1", "fc2"],
            lora_dropout   = 0.1,
            bias           = "none",
        )
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_uni_classifier(n_classes, checkpoint_path):
    model = UNILoRAClassifier(n_classes)
    path  = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")
    state = torch.load(path, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Checkpoint caricato: {path}")
    return model


base_model = build_uni_classifier(N_CLASSES, FINETUNED_CKPT)

# %% [markdown]
# ## 4 · Funzioni di valutazione e throughput

# %%
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Eval", leave=False):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(imgs.to(device, non_blocking=True))
        all_preds.append(logits.float().argmax(1).cpu())
        all_labels.append(labels)
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc    = float((preds == labels).mean())
    f1_mac = float(f1_score(labels, preds, average="macro", zero_division=0))
    return acc, f1_mac, preds, labels


def measure_throughput(model, device, img_size=224, batch_size=32,
                       n_warmup=20, n_runs=100):
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size,
                        device=device, dtype=torch.bfloat16)
    for _ in range(n_warmup):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(dummy)
    torch.cuda.synchronize()
    return (n_runs * batch_size) / (time.perf_counter() - t0)

from fvcore.nn import FlopCountAnalysis

def measure_gflops(model, device, img_size=224):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)
    return flops.total() / 1e9  # GFLOPs

# %% [markdown]
# ## 5 · Sweep su r

# %%
results = []

for r in CFG["tome_r_values"]:
    print(f"\n{'='*50}")
    print(f"ToMe  r = {r}  (token mergiati per blocco)")

    model_r = copy.deepcopy(base_model).to(device)

    if r > 0:
        # ToMe va applicato al ViT interno, non al wrapper UNILoRAClassifier
        # model_r.backbone      → LoraModel (PEFT wrapper)
        # model_r.backbone.model → ViT timm puro
        tome_patch_timm(model_r.backbone.model, trace_source=False, prop_attn=True)
        model_r.backbone.model.r = r
        print(f"  ToMe applicato con r={r}")
    else:
        print("  Baseline (nessun merging)")

    thr         = measure_throughput(model_r, device,
                                     img_size=CFG["img_size"],
                                     batch_size=CFG["batch_size"])
    tokens_left = max(CFG["num_patches"] - r * 24, 1)
    acc, f1_mac, preds, labels = evaluate(model_r, test_loader, device)

    res = dict(r=r, acc=acc, f1_macro=f1_mac,
               throughput=thr, tokens_left=tokens_left)
    results.append(res)

    gflops = measure_gflops(model_r, device, img_size=CFG["img_size"])

    res = dict(r=r, acc=acc, f1_macro=f1_mac,
           throughput=thr, tokens_left=tokens_left, gflops=gflops)

    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 macro   : {f1_mac:.4f}")
    print(f"  Throughput : {thr:.1f} img/s")
    print(f"  Token left : ~{tokens_left}")
    print(f"  GFLOPs     : {gflops:.1f}")
    print(classification_report(labels, preds,
                                 target_names=CLASS_NAMES, zero_division=0))
    
    
    del model_r
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6 · Tabella riassuntiva

# %%
baseline_thr = results[0]["throughput"]
print(f"\n{'r':>4} {'Acc':>8} {'F1 macro':>10} {'Throughput':>12} "
      f"{'Tokens':>8} {'Speedup':>8}")
print("-" * 58)
for res in results:
    speedup = res["throughput"] / baseline_thr
    print(f"{res['r']:>4} {res['acc']:>8.4f} {res['f1_macro']:>10.4f} "
          f"{res['throughput']:>12.1f} {res['tokens_left']:>8} {speedup:>8.2f}x")

# %% [markdown]
# ## 7 · Curva accuracy/speedup

# %%
accs     = [r["acc"]      for r in results]
f1s      = [r["f1_macro"] for r in results]
speedups = [r["throughput"] / baseline_thr for r in results]
r_vals   = [r["r"]        for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, ys, ylabel, color in zip(
        axes,
        [accs, f1s],
        ["Test Accuracy", "F1 Macro"],
        ["steelblue", "darkorange"]):
    ax.plot(speedups, ys, marker="o", linewidth=2, color=color)
    for i, rv in enumerate(r_vals):
        ax.annotate(f"r={rv}", (speedups[i], ys[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Speedup (×)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"ToMe — {ylabel} vs Speedup ({CFG['dataset_name']})")
    ax.grid(alpha=0.3)

plt.tight_layout()
out_dir = Path(f"/data/checkpoints-Attention-Pruning/{CFG['dataset_name']}/tome")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "tome_tradeoff.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8 · Confusion Matrix per il best r (speedup ≥ 1.5×)

# %%
candidates = [(res["f1_macro"], res) for res in results
              if res["throughput"] / baseline_thr >= 1.5]

if candidates:
    best = max(candidates, key=lambda x: x[0])[1]
    print(f"Best r con speedup ≥ 1.5×: r={best['r']} "
          f"(F1={best['f1_macro']:.4f}, "
          f"speedup={best['throughput']/baseline_thr:.2f}×)")

    model_best = copy.deepcopy(base_model).to(device)
    tome_patch_timm(model_best.backbone.model, trace_source=False, prop_attn=True)
    model_best.backbone.model.r = best["r"]

    _, _, preds_best, labels_best = evaluate(model_best, test_loader, device)

    cm = confusion_matrix(labels_best, preds_best, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — ToMe r={best['r']} ({CFG['dataset_name']})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_r{best['r']}.png", dpi=150)
    plt.show()
    del model_best
    torch.cuda.empty_cache()
else:
    print("Nessun r raggiunge speedup ≥ 1.5×. Prova valori di r più grandi.")

# %% [markdown]
# ## 9 · Salvataggio risultati

# %%
with open(out_dir / "tome_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Risultati salvati in {out_dir / 'tome_results.json'}")