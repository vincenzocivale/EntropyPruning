# %% [markdown]
# # DynamicViT — Training su UNI (wrapper approach)
# 
# Integra i moduli `PredictorLG` del metodo DynamicViT direttamente nel classificatore UNI LoRA
# **senza modificare** la classe backbone. L'approccio è analogo a `IntegratedPrunedClassifier`:
# 
# - `base_model.backbone` e `base_model.head` sono **condivisi per riferimento**
# - `UNIDynamicViT` aggiunge solo i `PredictorLG` e ridefinisce `forward()` manualmente
# - Il ViT backbone **non viene toccato**: si chiama ogni blocco a mano dentro il forward
# 
# ## Strategia training (come DynamicViT originale)
# 
# | Fase | Meccanismo | Differenziabilità |
# |---|---|---|
# | **Training** | Gumbel-Softmax hard | Differenziabile (straight-through) |
# | **Inference** | Top-k hard + gather | Sequenza fisicamente ridotta |
# 
# ## Loss (4 componenti)
# 
# ```
# L = clf_w  * CE(pred, labels)                    # classificazione
#   + ratio_w * MSE(keep_ratio_actual, keep_ratio_target) / n_stages  # budget
#   + dist_w  * KL(pred_logits, teacher_logits)    # distillazione class
#   + dist_w  * MSE(pred_tokens, teacher_tokens)   # distillazione token
# ```
# 
# ## Stadi di pruning
# 
# Per UNI ViT-L (depth=24): blocchi **[6, 12, 18]**, keep ratio target **[0.7, 0.49, 0.343]**
# (ogni stadio mantiene 70% del precedente → keep cumulativo: 70% → 49% → 34%)
# 

# %% [markdown]
# ## 0 · Imports

# %%
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
import timm
from peft import LoraConfig
from peft.tuners.lora import LoraModel
import wandb

sys.path.append(".")
from src.dataset import HistologicalImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {device} | timm: {timm.__version__}")

# %% [markdown]
# ## 1 · Configurazione

# %%
DATASET_NAME = "NCT-CRC-HE"
CKPT_BASE    = Path(f"/raid/DATASETS/checkpoints-Attention-Pruning/{DATASET_NAME}")

CFG = dict(
    # ---- Dati ----
    data_dir        = f"/raid/DATASETS/{DATASET_NAME}",
    img_size        = 224,
    patch_size      = 16,
    batch_size      = 16,
    num_workers     = 4,

    # ---- Backbone UNI LoRA ----
    embed_dim       = 1024,
    depth           = 24,
    num_heads       = 16,

    # ---- DynamicViT: stadi di pruning ----
    # Per ViT-L depth=24: pruning a blocco 6, 12, 18
    # ogni stadio mantiene il 70% dei token del precedente
    pruning_loc     = [6, 12, 18],
    keep_ratio      = [0.7, 0.49, 0.343],   # cumulativo: 70% -> 49% -> 34%

    # ---- Loss weights ----
    clf_weight      = 1.0,     # weight CE loss classificazione
    ratio_weight    = 2.0,     # weight ratio regression loss
    distill_weight  = 0.5,     # weight KL + MSE distillation

    # ---- Ottimizzazione ----
    lr              = 5e-5,    # più basso del baseline: solo PredictorLG trainable
    min_lr          = 1e-7,
    weight_decay    = 0.05,
    epochs          = 30,
    warmup_epochs   = 3,
    accum_steps     = 2,
    max_norm        = 1.0,
    label_smoothing = 0.1,

    seed            = 42,

    # ---- Checkpoints ----
    ckpt_base       = CKPT_BASE / "uni_finetuned" / "best_model.pt",
    output_dir      = CKPT_BASE / "uni_dynamicvit",
)
Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print(f"Dataset      : {DATASET_NAME}")
print(f"Pruning loc  : {CFG['pruning_loc']}")
print(f"Keep ratio   : {CFG['keep_ratio']}")
print(f"Output dir   : {CFG['output_dir']}")

# %% [markdown]
# ## 2 · Dataset

# %%
train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
    T.RandomApply([T.ColorJitter(0.2, 0.2, 0.1, 0.05)], p=0.5),
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
eval_tf = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_ds = HistologicalImageDataset(f"{CFG['data_dir']}/train", transform=train_tf)
val_ds   = HistologicalImageDataset(f"{CFG['data_dir']}/val",   transform=eval_tf)
test_ds  = HistologicalImageDataset(f"{CFG['data_dir']}/test",  transform=eval_tf)

counts  = np.bincount(train_ds.labels)
weights = torch.from_numpy((1.0 / counts)[train_ds.labels]).double()
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

kw = dict(batch_size=CFG["batch_size"], num_workers=CFG["num_workers"],
          pin_memory=True, persistent_workers=True)
train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **kw)
val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

CLASS_NAMES = train_ds.class_names
N_CLASSES   = len(CLASS_NAMES)
print(f"Classi ({N_CLASSES}): {CLASS_NAMES}")
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

# %% [markdown]
# ## 3 · Architetture
# 
# ### 3a. Base UNILoRAClassifier (teacher e backbone condiviso)
# 
# ### 3b. PredictorLG — da DynamicViT
# 
# Modulo leggero che combina feature locali e globali per stimare
# la probabilità keep/prune di ogni token:
# 
# ```
# local_x  = first C//2 dims per token
# global_x = policy-weighted mean of last C//2 dims (global context)
# output   = LogSoftmax(MLP(cat[local_x, global_x]))  # shape: (B, N, 2)
# ```
# 
# ### 3c. UNIDynamicViT — wrapper sul classificatore standard
# 
# Accetta un `UNILoRAClassifier` già caricato e aggiunge `PredictorLG`
# ai layer `pruning_loc` **senza modificare il backbone**.

# %%
# ── Base LoRA classifier ────────────────────────────────────────────────────
class UNILoRAClassifier(nn.Module):
    def __init__(self, n_classes, dropout=0.1):
        super().__init__()
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, dynamic_img_size=True, num_classes=0,
        )
        lora_cfg = LoraConfig(
            r=8, lora_alpha=32,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1, bias="none",
        )
        self.backbone = LoraModel(backbone, lora_cfg, adapter_name="default")
        self.head = nn.Sequential(
            nn.LayerNorm(1024), nn.Dropout(dropout), nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ── PredictorLG — da DynamicViT (Rao et al. 2021) ──────────────────────────
class PredictorLG(nn.Module):
    """
    Predittore leggero per la decisione keep/prune di ogni token.
    Combina feature locali (prima metà canali) con un contesto globale
    pesato dalla decisione precedente (seconda metà canali).

    Input:
        x      : (B, N, embed_dim)  — token patch (senza CLS)
        policy : (B, N, 1)          — decisione precedente (1=vivo, 0=potato)
    Output:
        (B, N, 2)  — log-prob [keep, prune] per ogni token
    """
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x  = x[:, :, :C // 2]
        # context globale: media pesata dalla policy precedente
        global_x = (x[:, :, C // 2:] * policy).sum(dim=1, keepdim=True) \
                   / torch.sum(policy, dim=1, keepdim=True).clamp(min=1e-6)
        x = torch.cat([local_x, global_x.expand(B, N, C // 2)], dim=-1)
        return self.out_conv(x)


# ── UNIDynamicViT — wrapper ─────────────────────────────────────────────────
class UNIDynamicViT(nn.Module):
    """
    Wrapper su UNILoRAClassifier che integra PredictorLG ai layer pruning_loc.

    NON modifica il backbone: i blocchi ViT vengono eseguiti manualmente
    dentro forward(), inserendo la scoring logic tra i blocchi specificati.

    Training:
        - Gumbel-Softmax hard (straight-through) → sequenza rimane N token
        - token zeroed-out (moltiplicati per 0 nella policy)
        - Restituisce (logits, token_features, prev_decision, out_pred_prob)

    Inference:
        - Hard top-k + torch.gather → sequenza fisicamente ridotta
        - Restituisce solo logits
    """
    def __init__(self, base_model: UNILoRAClassifier, pruning_loc, keep_ratio):
        super().__init__()
        # Condividiamo backbone e head per riferimento (no deepcopy)
        self.backbone     = base_model.backbone
        self.head         = base_model.head
        self.pruning_loc  = pruning_loc
        self.keep_ratio   = keep_ratio

        # Alloca un PredictorLG per ogni stadio
        self.predictors = nn.ModuleList([
            PredictorLG(embed_dim=CFG["embed_dim"])
            for _ in pruning_loc
        ])

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _run_block(blk, x):
        """Esegue un blocco timm ViT-L (con LayerScale ls1/ls2)."""
        x_n1 = blk.norm1(x)
        B, N, C = x_n1.shape
        qkv = blk.attn.qkv(x_n1).reshape(B, N, 3, blk.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = blk.attn.q_norm(q), blk.attn.k_norm(k)
        attn = (q @ k.transpose(-2, -1) * blk.attn.scale).softmax(-1)
        attn_out = (blk.attn.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = blk.attn.proj_drop(blk.attn.proj(attn_out))
        if hasattr(blk, "ls1"):
            x = x + blk.drop_path1(blk.ls1(attn_out))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        else:
            x = x + blk.drop_path1(attn_out)
            x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))
        return x

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x):
        vit = self.backbone.model
        B   = x.shape[0]

        # Preamble ViT
        x = vit.patch_embed(x)
        x = vit._pos_embed(x)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)

        N_patches = x.shape[1] - 1   # numero di patch token (senza CLS)

        # Stato per il tracking della decisione
        prev_decision = torch.ones(B, N_patches, 1, dtype=x.dtype, device=x.device)
        out_pred_prob = []          # per la ratio loss
        p_count = 0

        for i, blk in enumerate(vit.blocks):
            if i in self.pruning_loc:
                spatial_x = x[:, 1:]   # solo patch token (senza CLS)

                # ── Score prediction ─────────────────────────────────────────
                pred_score = self.predictors[p_count](spatial_x, prev_decision)
                # pred_score shape: (B, N_current, 2)  — log-prob

                if self.training:
                    # Gumbel-Softmax hard: differenziabile, sequenza invariata
                    hard_keep = F.gumbel_softmax(pred_score, hard=True, dim=-1)[:, :, 0:1]
                    # hard_keep: (B, N_current, 1)  — 1=keep, 0=prune
                    hard_keep = hard_keep * prev_decision  # propaga decisioni precedenti
                    out_pred_prob.append(hard_keep.squeeze(-1))  # per ratio loss

                    # Esegui blocco PRIMA del masking
                    x = self._run_block(blk, x)

                    # Zero-out token pruned: OUT-OF-PLACE (no inplace su slice view)
                    cls_tok = x[:, :1, :]
                    masked  = x[:, 1:, :] * hard_keep   # out-of-place multiply
                    x       = torch.cat([cls_tok, masked], dim=1)
                    prev_decision = hard_keep

                else:
                    # Inference: top-k hard selection → sequenza ridotta
                    score = pred_score[:, :, 0].exp()  # prob keep
                    num_keep = int(N_patches * self.keep_ratio[p_count])
                    keep_idx = score.topk(num_keep, dim=1).indices   # (B, num_keep)
                    keep_idx_sorted = keep_idx.sort(dim=1).values    # mantieni ordine spaziale

                    cls_tok  = x[:, :1, :]
                    patches  = x[:, 1:, :]
                    C = patches.shape[-1]
                    kept     = torch.gather(patches, 1,
                                            keep_idx_sorted.unsqueeze(-1).expand(-1, -1, C))
                    x = torch.cat([cls_tok, kept], dim=1)

                    # Aggiorna N_patches per il prossimo stadio
                    prev_decision = torch.gather(
                        prev_decision, 1,
                        keep_idx_sorted.unsqueeze(-1)
                    )

                    # Esegui blocco sulla sequenza ridotta
                    x = self._run_block(blk, x)

                p_count += 1

            else:
                # Blocco normale
                x = self._run_block(blk, x)

        x = vit.norm(x)
        features = x[:, 1:]    # tutti i patch token finali

        # Pooling: media sui patch token
        cls_repr = x[:, 1:].mean(dim=1)
        logits = self.head(cls_repr)

        if self.training:
            return logits, features, prev_decision.detach(), out_pred_prob
        else:
            return logits


print("Architetture definite.")

# %% [markdown]
# ## 4 · Costruzione modelli (student + teacher)

# %%
def load_ckpt(model, path, strict=False):
    state = torch.load(Path(path), map_location="cpu")
    if "state_dict" in state: state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    miss, unex = model.load_state_dict(state, strict=strict)
    if miss:  print(f"  [WARN] {len(miss)} missing keys")
    if unex:  print(f"  [WARN] {len(unex)} unexpected keys")
    return model


# ── Student: UNIDynamicViT ───────────────────────────────────────────────────
print("Caricamento base classifier (student)...")
base_clf = UNILoRAClassifier(N_CLASSES)
load_ckpt(base_clf, CFG["ckpt_base"], strict=False)

student = UNIDynamicViT(
    base_model   = base_clf,
    pruning_loc  = CFG["pruning_loc"],
    keep_ratio   = CFG["keep_ratio"],
).to(device)

# Congela il backbone (LoRA inclusa) e la head — allena SOLO i PredictorLG
for p in student.backbone.parameters():
    p.requires_grad_(False)
for p in student.head.parameters():
    p.requires_grad_(False)
for p in student.predictors.parameters():
    p.requires_grad_(True)

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6
total     = sum(p.numel() for p in student.parameters()) / 1e6
print(f"Parametri totali    : {total:.1f}M")
print(f"Parametri trainable : {trainable:.1f}M  (solo PredictorLG)")


# ── Teacher: lo stesso classificatore base, frozen ──────────────────────────
# Il teacher restituisce (logits, token_features) dal forward modificato
print("\nCostruzione teacher model...")

class TeacherWrapper(nn.Module):
    """Wrapper sul classificatore base: restituisce (logits, token_features)."""
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model.backbone
        self.head     = base_model.head

    def forward(self, x):
        vit = self.backbone.model
        x   = vit.patch_embed(x)
        x   = vit._pos_embed(x)
        x   = vit.patch_drop(x)
        x   = vit.norm_pre(x)
        for blk in vit.blocks:
            x = blk(x)
        x         = vit.norm(x)
        features  = x[:, 1:]              # patch token
        cls_repr  = x[:, 1:].mean(dim=1)
        logits    = self.head(cls_repr)
        return logits, features

teacher = TeacherWrapper(base_clf).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

print("Teacher: frozen UNILoRAClassifier")
del base_clf   # base_clf ora non serve più: backbone/head sono referenziati
gc.collect()

# %% [markdown]
# ## 5 · Loss DynamicViT
# 
# 4 componenti:
# 1. **CE** — cross-entropy con label smoothing
# 2. **Ratio MSE** — penalizza deviazioni dal keep ratio target
# 3. **KL logits** — distillazione sulla distribuzione di classe del teacher
# 4. **MSE tokens** — distillazione sui token features del teacher (sui token superstiti)

# %%
class DynamicViTLoss(nn.Module):
    """
    Loss DynamicViT adattata per UNIDynamicViT.

    Attende come outputs: (logits, token_features, prev_decision, out_pred_prob)
    Il teacher_model deve restituire: (cls_logits, token_features)
    """
    def __init__(self, teacher_model, n_stages, keep_ratio,
                 clf_weight=1.0, ratio_weight=2.0, distill_weight=0.5,
                 label_smoothing=0.1, print_every=100):
        super().__init__()
        self.teacher      = teacher_model
        self.n_stages     = n_stages
        self.keep_ratio   = keep_ratio
        self.clf_weight   = clf_weight
        self.ratio_weight = ratio_weight
        self.dist_weight  = distill_weight
        self.base_crit    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Moving average per stampa debug
        self._count = 0
        self._sums  = {"cls": 0, "ratio": 0, "kl": 0, "mse": 0}
        self._print_every = print_every

    def forward(self, inputs, outputs, labels):
        logits, token_pred, prev_decision, out_pred_prob = outputs

        # ── 1. CE loss ───────────────────────────────────────────────────────
        cls_loss = self.base_crit(logits, labels)

        # ── 2. Ratio regression loss ─────────────────────────────────────────
        # out_pred_prob[i]: (B, N)  — 1 se token tenuto, 0 altrimenti (Gumbel hard)
        ratio_loss = 0.0
        for i, pred_prob in enumerate(out_pred_prob):
            actual_ratio = pred_prob.mean()   # media su tutti i token e il batch
            ratio_loss   = ratio_loss + (actual_ratio - self.keep_ratio[i]) ** 2

        # ── 3. Distillazione (teacher, no grad) ──────────────────────────────
        with torch.no_grad():
            cls_t, token_t = self.teacher(inputs)

        cls_kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.log_softmax(cls_t,  dim=-1),
            reduction="batchmean",
            log_target=True,
        )

        # ── 4. Token distillation: solo sui token superstiti ─────────────────
        # token_pred e token_t hanno shape (B, N_alive, C)
        # prev_decision: (B, N_patches, 1)  — ma N può essere diverso da token_t
        # Usiamo distillazione media su tutti i token rimasti (approx)
        B, N_student, C = token_pred.shape
        _, N_teacher, _ = token_t.shape

        if N_student == N_teacher:
            # Tutti i token presenti: usa prev_decision come maschera
            bool_mask = prev_decision.reshape(B * N_student) > 0.5
            tp_flat   = token_pred.reshape(B * N_student, C)
            tt_flat   = token_t.reshape(B * N_teacher, C)
            if bool_mask.sum() > 0:
                token_mse = torch.pow(tp_flat[bool_mask] - tt_flat[bool_mask], 2).mean()
            else:
                token_mse = token_pred.new_zeros(1)
        else:
            # Durante training (gumbel) la sequenza non si riduce,
            # usiamo media globale come approssimazione
            N_min = min(N_student, N_teacher)
            token_mse = torch.pow(token_pred[:, :N_min] - token_t[:, :N_min], 2).mean()

        # ── Loss totale ───────────────────────────────────────────────────────
        loss = (
            self.clf_weight   * cls_loss
            + self.ratio_weight * ratio_loss / self.n_stages
            + self.dist_weight  * cls_kl_loss
            + self.dist_weight  * token_mse
        )

        # Debug print ogni print_every iterazioni
        self._count += 1
        self._sums["cls"]   += cls_loss.item()
        self._sums["ratio"] += ratio_loss.item() if isinstance(ratio_loss, torch.Tensor) else ratio_loss
        self._sums["kl"]    += cls_kl_loss.item()
        self._sums["mse"]   += token_mse.item()
        if self._count % self._print_every == 0:
            n = self._print_every
            print(f"  loss_info: cls={self._sums['cls']/n:.4f}  "
                  f"ratio={self._sums['ratio']/n:.4f}  "
                  f"kl={self._sums['kl']/n:.4f}  "
                  f"mse={self._sums['mse']/n:.4f}")
            self._count = 0
            for k in self._sums: self._sums[k] = 0.0

        return loss, [cls_loss, ratio_loss, cls_kl_loss, token_mse]


criterion = DynamicViTLoss(
    teacher_model  = teacher,
    n_stages       = len(CFG["pruning_loc"]),
    keep_ratio     = CFG["keep_ratio"],
    clf_weight     = CFG["clf_weight"],
    ratio_weight   = CFG["ratio_weight"],
    distill_weight = CFG["distill_weight"],
    label_smoothing= CFG["label_smoothing"],
)
print("Loss DynamicViT pronta.")

# %% [markdown]
# ## 6 · Ottimizzatore & Scheduler
# 
# Si allena **solo** il `predictors` (i `PredictorLG`).
# AdamW + cosine warmup senza LLRD (pochi parametri).

# %%
optimizer = torch.optim.AdamW(
    student.predictors.parameters(),
    lr           = CFG["lr"],
    weight_decay = CFG["weight_decay"],
)

steps_per_epoch = len(train_loader) // CFG["accum_steps"]
total_steps     = CFG["epochs"] * steps_per_epoch
warmup_steps    = CFG["warmup_epochs"] * steps_per_epoch


def cosine_schedule_with_warmup(base_lr, min_lr, total_steps, warmup_steps):
    schedule = []
    for t in range(total_steps):
        if t < warmup_steps:
            lr = base_lr * (t + 1) / max(1, warmup_steps)
        else:
            progress = (t - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        schedule.append(lr)
    return np.array(schedule)


lr_schedule = cosine_schedule_with_warmup(
    CFG["lr"], CFG["min_lr"], total_steps, warmup_steps
)
print(f"Steps totali: {total_steps}  |  Warmup: {warmup_steps}")
print(f"LR iniziale : {CFG['lr']:.2e}  |  LR min: {CFG['min_lr']:.2e}")

# %% [markdown]
# ## 7 · WandB

# %%
wandb.init(
    project = "uni-dynamicvit",
    name    = (f"{DATASET_NAME}_dynamicvit"
               f"_loc{'_'.join(str(l) for l in CFG['pruning_loc'])}"
               f"_kr{'_'.join(f'{r:.2f}' for r in CFG['keep_ratio'])}"),
    config  = CFG,
    tags    = [DATASET_NAME, "dynamicvit", "uni", "lora"],
)

# %% [markdown]
# ## 8 · Training Loop

# %%
@torch.no_grad()
def evaluate(model, loader):
    """Valutazione sul val set (inference mode: no Gumbel)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ce = nn.CrossEntropyLoss()
    for imgs, lbls in tqdm(loader, leave=False, desc="Val"):
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(imgs)   # inference: restituisce solo logits
        total_loss += ce(logits, lbls).item() * len(lbls)
        correct    += (logits.argmax(1) == lbls).sum().item()
        total      += len(lbls)
    return total_loss / total, correct / total


def train_one_epoch(model, criterion, loader, optimizer, epoch, grad_offset):
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches    = 0
    grad_i       = grad_offset
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{CFG['epochs']}", leave=False)
    for batch_i, (imgs, lbls) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        # Aggiorna LR
        if batch_i % CFG["accum_steps"] == 0:
            current_lr = lr_schedule[min(grad_i, len(lr_schedule) - 1)]
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(imgs)   # (logits, token_features, prev_decision, out_pred_prob)
            loss, _ = criterion(imgs, outputs, lbls)
            loss    = loss / CFG["accum_steps"]

        loss.backward()

        if (batch_i + 1) % CFG["accum_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(
                student.predictors.parameters(), CFG["max_norm"]
            )
            optimizer.step()
            optimizer.zero_grad()
            grad_i += 1

        with torch.no_grad():
            acc = (outputs[0].detach().argmax(1) == lbls).float().mean().item()
        running_loss += loss.item() * CFG["accum_steps"]
        running_acc  += acc
        n_batches    += 1
        pbar.set_postfix(loss=f"{running_loss/n_batches:.4f}",
                         acc=f"{running_acc/n_batches:.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    # flush ultimo accum incompleto
    if len(loader) % CFG["accum_steps"] != 0:
        torch.nn.utils.clip_grad_norm_(student.predictors.parameters(), CFG["max_norm"])
        optimizer.step()
        optimizer.zero_grad()

    return running_loss / n_batches, running_acc / n_batches, grad_i


print("Funzioni di training pronte.")

# %%
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc, best_epoch = 0.0, 0
grad_step = 0

epoch_bar = tqdm(range(1, CFG["epochs"] + 1), desc="Training", unit="epoch")

for epoch in epoch_bar:
    tr_loss, tr_acc, grad_step = train_one_epoch(
        student, criterion, train_loader, optimizer, epoch, grad_step
    )
    vl_loss, vl_acc = evaluate(student, val_loader)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    epoch_bar.set_postfix(
        tr_loss=f"{tr_loss:.4f}", tr_acc=f"{tr_acc:.4f}",
        vl_loss=f"{vl_loss:.4f}", vl_acc=f"{vl_acc:.4f}",
        best=f"{best_val_acc:.4f}"
    )

    wandb.log({
        "epoch"     : epoch,
        "train/loss": tr_loss, "train/acc": tr_acc,
        "val/loss"  : vl_loss, "val/acc"  : vl_acc,
        "lr"        : optimizer.param_groups[0]["lr"],
    })

    if vl_acc > best_val_acc:
        best_val_acc, best_epoch = vl_acc, epoch
        torch.save(student.predictors.state_dict(),
                   CFG["output_dir"] / "best_predictors.pt")
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["best_epoch"]   = best_epoch
        epoch_bar.write(f"  ✅ Epoch {epoch:02d} — nuovo best val_acc={best_val_acc:.4f}")

epoch_bar.write(f"\nBest val_acc={best_val_acc:.4f} @ epoch {best_epoch}")

# %% [markdown]
# ## 9 · Curve di Apprendimento

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, len(history["train_loss"]) + 1)
axes[0].plot(ep, history["train_loss"], label="train")
axes[0].plot(ep, history["val_loss"],   label="val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep, history["train_acc"], label="train")
axes[1].plot(ep, history["val_acc"],   label="val")
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle(f"DynamicViT UNI — {DATASET_NAME}\n"
             f"pruning_loc={CFG['pruning_loc']}  keep={CFG['keep_ratio']}")
plt.tight_layout()
plt.savefig(CFG["output_dir"] / "training_curves.png", dpi=150)
plt.show()

# %% [markdown]
# ## 10 · Valutazione Test Set

# %%
# Ricarica i migliori predictors
student.predictors.load_state_dict(
    torch.load(CFG["output_dir"] / "best_predictors.pt", map_location=device)
)
student.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="Test"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = student(imgs.to(device))
        all_preds.append(logits.float().argmax(1).cpu())
        all_labels.append(lbls)

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

acc      = float((all_preds == all_labels).mean())
f1_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
print(f"Test accuracy : {acc:.4f}  |  F1 macro : {f1_macro:.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# %% [markdown]
# ## 11 · Confusion Matrix

# %%
cm = confusion_matrix(all_labels, all_preds, normalize="true")
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — DynamicViT UNI ({DATASET_NAME})")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(CFG["output_dir"] / "confusion_matrix.png", dpi=150)
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.show()

# %% [markdown]
# ## 12 · Token Budget per stadio

# %%
num_patches = (CFG["img_size"] // CFG["patch_size"]) ** 2
stages_n    = [num_patches]
for kr in CFG["keep_ratio"]:
    stages_n.append(int(stages_n[-1] * kr))

print(f"Token budget per stadio (inference):")
print(f"  Ingresso    : {num_patches}")
for i, (loc, kr, n) in enumerate(zip(CFG["pruning_loc"], CFG["keep_ratio"], stages_n[1:])):
    print(f"  Dopo blocco {loc:2d} (keep={kr:.0%}): {n} token")

# Visualizza
fig, ax = plt.subplots(figsize=(8, 4))
x_labels = ["Input"] + [f"Post-block {l}" for l in CFG["pruning_loc"]]
ax.bar(range(len(stages_n)), stages_n, color=["#6b7280"] + ["#e63946"] * len(CFG["pruning_loc"]))
ax.set_xticks(range(len(stages_n)))
ax.set_xticklabels(x_labels)
ax.set_ylabel("Token rimanenti")
ax.set_title(f"Token Budget — DynamicViT UNI (keep={CFG['keep_ratio']})")
for i, n in enumerate(stages_n):
    ax.text(i, n + 1, str(n), ha="center", va="bottom", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(CFG["output_dir"] / "token_budget.png", dpi=150)
plt.show()

# %% [markdown]
# ## 13 · Salvataggio finale & WandB

# %%
# Salva anche la configurazione dei predictors per il caricamento in speed_comparison
torch.save(
    {
        "predictors": student.predictors.state_dict(),
        "cfg": {
            "pruning_loc" : CFG["pruning_loc"],
            "keep_ratio"  : CFG["keep_ratio"],
            "embed_dim"   : CFG["embed_dim"],
        }
    },
    CFG["output_dir"] / "dynamicvit_checkpoint.pt"
)
print(f"Checkpoint salvato: {CFG['output_dir'] / 'dynamicvit_checkpoint.pt'}")

wandb.log({"test/accuracy": acc, "test/f1_macro": f1_macro})
wandb.summary.update({"test/accuracy": acc, "test/f1_macro": f1_macro})
wandb.finish()
print(f"Test accuracy : {acc:.4f}  |  F1 macro : {f1_macro:.4f}")


