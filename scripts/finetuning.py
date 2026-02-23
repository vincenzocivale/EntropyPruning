# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import timm, types, sys, wandb, time
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from peft import LoraConfig
from peft.tuners.lora import LoraModel
from sklearn.metrics import f1_score
from fvcore.nn import FlopCountAnalysis
import torchvision.transforms as T

sys.path.append(".")
from src.dataset import HistologicalImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%
from pathlib import Path
import torch
import numpy as np
import random

# ===== Dataset =====
DATA_DIR = "/data/BREAKHIS"
DATASET_NAME = Path(DATA_DIR).name

BASE_CHECKPOINT_DIR = Path("checkpoints")

# ===== Config =====
CFG = dict(
    # Dataset
    data_dir        = DATA_DIR,
    dataset_name    = DATASET_NAME,
    img_size        = 224,
    batch_size      = 16,
    num_workers     = 8,
    seed            = 42,

    # Checkpoint specifici per dataset
    classifier_ckpt = BASE_CHECKPOINT_DIR / DATASET_NAME / "uni_finetuned" / "best_model.pt",
    forecaster_ckpt = BASE_CHECKPOINT_DIR / DATASET_NAME / "forecaster" / "forecaster_src02_tgt23.pt",

    # Output pruned specifico per dataset
    output_dir      = BASE_CHECKPOINT_DIR / DATASET_NAME / "pruned_finetuned",

    # Pruning
    prune_layer     = 2,
    keep_ratio      = 0.5,

    # Training
    epochs          = 20,
    lr_head         = 1e-3,
    lr_backbone     = 1e-4,
    weight_decay    = 0.01,
    label_smoothing = 0.1,
    far_threshold   = 1e-4,

    # Logging
    wandb_project   = f"pruned-finetuning",
)

# ===== Directory creation =====
CFG["output_dir"].mkdir(parents=True, exist_ok=True)

# ===== Reproducibility =====
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Configurazione pruning caricata per dataset: {DATASET_NAME}")
print(f"Classifier ckpt: {CFG['classifier_ckpt']}")
print(f"Forecaster ckpt: {CFG['forecaster_ckpt']}")
print(f"Output dir: {CFG['output_dir']}")

# %%
train_tf = T.Compose([
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation((90,90))], p=0.5),
    T.RandomApply([T.ColorJitter(0.2,0.2,0.1,0.05)], p=0.5),
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
eval_tf = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

train_ds = HistologicalImageDataset(f"{CFG['data_dir']}/train", transform=train_tf)
val_ds   = HistologicalImageDataset(f"{CFG['data_dir']}/val",   transform=eval_tf)
test_ds  = HistologicalImageDataset(f"{CFG['data_dir']}/test",  transform=eval_tf)

# Sampler bilanciato
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
print(f"Classi: {CLASS_NAMES}")
print(f"Counts per classe: {counts}")

# %%
def compute_tar_at_far(scores, is_correct, far_threshold=1e-4):
    """
    TAR@FAR per classificazione multiclasse.
    score   = max softmax probability
    correct = predizione corretta (bool)
    FAR     = fraction of wrong predictions above threshold
    TAR     = fraction of correct predictions above threshold
    """
    scores    = np.array(scores)
    correct   = np.array(is_correct).astype(bool)
    incorrect = ~correct

    if incorrect.sum() == 0:
        return 1.0, float('nan')

    # Soglia: al FAR target
    n_incorrect    = incorrect.sum()
    n_far_allowed  = max(1, int(np.ceil(n_incorrect * far_threshold)))
    sorted_wrong   = np.sort(scores[incorrect])[::-1]
    threshold      = sorted_wrong[min(n_far_allowed - 1, len(sorted_wrong) - 1)]

    tar = (scores[correct] >= threshold).mean()
    return float(tar), float(threshold)


def evaluate(model, loader, device, far_threshold=1e-4):
    """Valuta: accuracy, F1 macro, TAR@FAR."""
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs  = logits.softmax(-1)
            preds  = probs.argmax(-1).cpu()
            score  = probs.max(-1).values.cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            all_scores.append(score)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.cat(all_scores).numpy()

    acc        = (all_preds == all_labels).mean()
    f1         = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    is_correct = (all_preds == all_labels)
    tar, thr   = compute_tar_at_far(all_scores, is_correct, far_threshold)

    return {"acc": acc, "f1_macro": f1, "tar_at_far": tar, "threshold": thr}



# %%
def benchmark_model(model, loader, device, n_warmup=10, label="model"):
    """Misura tempo di inferenza medio per immagine e FLOPs."""
    model.eval()

    # FLOPs su un singolo sample
    dummy = next(iter(loader))[0][:1].to(device)
    try:
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total_flops = flops.total()
    except Exception as e:
        print(f"FLOPs non calcolabili: {e}")
        total_flops = None

    # Warmup GPU
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            model(imgs.to(device))
            if i >= n_warmup:
                break

    # Timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_imgs = 0
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
            torch.cuda.synchronize()
            n_imgs += len(imgs)
    t1 = time.perf_counter()

    ms_per_img = (t1 - t0) / n_imgs * 1000

    print(f"\n── Benchmark: {label} ──")
    print(f"  Tempo medio per immagine: {ms_per_img:.2f} ms")
    if total_flops:
        print(f"  FLOPs per immagine:       {total_flops/1e9:.2f} GFLOPs")

    return {"ms_per_img": ms_per_img, "gflops": total_flops/1e9 if total_flops else None}


# %%
class AttentionForecaster(nn.Module):
    def __init__(self, embed_dim=1024, hidden=256,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(embed_dim, hidden)
        self.cls_query   = nn.Parameter(torch.randn(1,1,hidden)*0.02)
        self.self_attn   = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads,
                dim_feedforward=hidden*2, dropout=dropout,
                batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.cross_attn  = nn.ModuleList([
            nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.norm        = nn.LayerNorm(hidden)
        self.score_head  = nn.Sequential(
            nn.Linear(hidden*2,128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128,1))

    def forward(self, patch_embeddings):
        B, N, D = patch_embeddings.shape
        x = self.input_proj(patch_embeddings)
        for sa in self.self_attn:
            x = sa(x)
        cls = self.cls_query.expand(B,-1,-1)
        for ca, norm in zip(self.cross_attn, self.cross_norms):
            cls_out, _ = ca(cls, x, x)
            cls = norm(cls + cls_out)
        x_norm  = self.norm(x)
        cls_exp = cls.expand(-1,N,-1)
        scores  = self.score_head(torch.cat([x_norm, cls_exp], dim=-1)).squeeze(-1)
        return scores.softmax(-1)


class UNILoRAClassifier(nn.Module):
    """Classificatore base senza pruning — usato per il confronto."""
    def __init__(self, n_classes, dropout=0.1):
        super().__init__()
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, dynamic_img_size=True)
        lora_config = LoraConfig(r=8, lora_alpha=32,
            target_modules=["qkv","proj","fc1","fc2"],
            lora_dropout=0.1, bias="none")
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(
            nn.LayerNorm(1024), nn.Dropout(dropout), nn.Linear(1024, n_classes))
    def forward(self, x):
        return self.head(self.backbone(x))


class UNILoRAWithForecasterPruning(nn.Module):
    def __init__(self, n_classes, forecaster, prune_layer, keep_ratio, dropout=0.1):
        super().__init__()
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, dynamic_img_size=True)
        lora_config = LoraConfig(r=8, lora_alpha=32,
            target_modules=["qkv","proj","fc1","fc2"],
            lora_dropout=0.1, bias="none")
        self.backbone    = LoraModel(backbone, lora_config, adapter_name="default")
        self.head        = nn.Sequential(
            nn.LayerNorm(1024), nn.Dropout(dropout), nn.Linear(1024, n_classes))
        self.forecaster  = forecaster
        self.prune_layer = prune_layer
        self.keep_ratio  = keep_ratio

    def forward(self, x):
        make_block_hook = self._make_block_hook()
        orig_fwd = self.backbone.model.blocks[self.prune_layer].forward
        self.backbone.model.blocks[self.prune_layer].forward = \
            make_block_hook(self.prune_layer)
        out = self.head(self.backbone(x))
        self.backbone.model.blocks[self.prune_layer].forward = orig_fwd
        return out

    def _make_block_hook(self):
        def make_block_hook(idx):
            orig_fwd = self.backbone.model.blocks[idx].forward
            training  = self.training
            forecaster = self.forecaster
            keep_ratio = self.keep_ratio

            def block_fwd(x):
                x = orig_fwd(x)
                B, N, D = x.shape
                patch_emb = x[:, 1:]
                with torch.no_grad():
                    scores = forecaster(patch_emb)
                k_keep    = max(1, int((N-1) * keep_ratio))
                topk_vals = scores.topk(k_keep, dim=-1).values
                threshold = topk_vals[:, -1:]
                soft_mask = torch.sigmoid((scores - threshold) / 0.05)
                hard_mask = (scores >= threshold).float()
                st_mask   = hard_mask - soft_mask.detach() + soft_mask
                cls_tok   = x[:, :1, :]
                patches   = x[:, 1:, :]
                topk_idx  = scores.topk(k_keep, dim=-1).indices
                if training:
                    masked_patches = patches * st_mask.unsqueeze(-1)
                    kept = torch.stack([masked_patches[b][topk_idx[b]] for b in range(B)])
                else:
                    kept = torch.stack([patches[b][topk_idx[b]] for b in range(B)])
                return torch.cat([cls_tok, kept], dim=1)
            return block_fwd
        return make_block_hook


# %%
# baseline_model = UNILoRAClassifier(N_CLASSES).to(device)
# ckpt = torch.load(CFG["classifier_ckpt"], map_location=device)
# baseline_model.load_state_dict(ckpt, strict=False)
# baseline_model.eval()
# for p in baseline_model.parameters():
#     p.requires_grad_(False)

# # Metriche test
# baseline_metrics = evaluate(baseline_model, test_loader, device, CFG["far_threshold"])
# # Benchmark
# baseline_bench   = benchmark_model(baseline_model, test_loader, device, label="Baseline (no pruning)")

# print(f"\n── Baseline Test ──────────────────────────")
# print(f"  Accuracy:         {baseline_metrics['acc']:.3f}")
# print(f"  F1 macro:         {baseline_metrics['f1_macro']:.3f}")
# print(f"  TAR@FAR={CFG['far_threshold']:.0e}: {baseline_metrics['tar_at_far']:.3f}")
# print(f"  ms/img:           {baseline_bench['ms_per_img']:.2f}")
# print(f"  GFLOPs:           {baseline_bench['gflops']:.2f}")

# wandb.init(project=CFG["wandb_project"], name="baseline_no_pruning", config=CFG)
# wandb.log({
#     "test/acc"      : baseline_metrics["acc"],
#     "test/f1_macro" : baseline_metrics["f1_macro"],
#     "test/tar_at_far": baseline_metrics["tar_at_far"],
#     "test/ms_per_img"    : baseline_bench["ms_per_img"],
#     "test/gflops"        : baseline_bench["gflops"],
# })
# wandb.finish()

# %%
forecaster = AttentionForecaster().to(device)
forecaster.load_state_dict(torch.load(CFG["forecaster_ckpt"], map_location=device))
forecaster.eval()
for p in forecaster.parameters():
    p.requires_grad_(False)
print("Forecaster caricato e frozen")

model = UNILoRAWithForecasterPruning(
    n_classes   = N_CLASSES,
    forecaster  = forecaster,
    prune_layer = CFG["prune_layer"],
    keep_ratio  = CFG["keep_ratio"],
).to(device)

ckpt = torch.load(CFG["classifier_ckpt"], map_location=device)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")

# Pre fine-tuning
pre_metrics = evaluate(model, val_loader, device, CFG["far_threshold"])
print(f"\nPre fine-tuning val — acc={pre_metrics['acc']:.3f} "
      f"f1={pre_metrics['f1_macro']:.3f} tar={pre_metrics['tar_at_far']:.3f}")


# %%
backbone_params = [p for n,p in model.backbone.named_parameters() if p.requires_grad]
head_params     = list(model.head.parameters())

opt = torch.optim.AdamW([
    {"params": backbone_params, "lr": CFG["lr_backbone"]},
    {"params": head_params,     "lr": CFG["lr_head"]},
], weight_decay=CFG["weight_decay"])

total_steps = CFG["epochs"] * len(train_loader)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[CFG["lr_backbone"], CFG["lr_head"]],
    total_steps=total_steps, pct_start=0.1)

criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])


# %%
run_name = f"prune_layer{CFG['prune_layer']}_keep{int(CFG['keep_ratio']*100)}"
wandb.init(project=CFG["wandb_project"], name=run_name, config=CFG, tags=[f"prune_layer{CFG['prune_layer']}", f"keep{int(CFG['keep_ratio']*100)}", DATASET_NAME])

best_val_f1 = 0.
history = {"train_loss": [], "train_f1": [], "val_f1": [], "val_acc": [], "val_tar": []}

for epoch in range(CFG["epochs"]):
    # ── Train ──
    model.train()
    total_loss = 0.
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(train_loader, leave=False, desc=f"Ep{epoch+1}"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        total_loss += loss.item() * len(labels)
        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(labels.cpu())

    train_f1   = f1_score(torch.cat(all_labels).numpy(),
                          torch.cat(all_preds).numpy(),
                          average='macro', zero_division=0)
    train_loss = total_loss / len(train_ds)

    # ── Val ──
    val_metrics = evaluate(model, val_loader, device, CFG["far_threshold"])

    history["train_loss"].append(train_loss)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_metrics["f1_macro"])
    history["val_acc"].append(val_metrics["acc"])
    history["val_tar"].append(val_metrics["tar_at_far"])

    wandb.log({
        "epoch"          : epoch+1,
        "train/loss"     : train_loss,
        "train/f1_macro" : train_f1,
        "val/acc"        : val_metrics["acc"],
        "val/f1_macro"   : val_metrics["f1_macro"],
        "val/tar_at_far" : val_metrics["tar_at_far"],
        "lr_backbone"    : opt.param_groups[0]["lr"],
        "lr_head"        : opt.param_groups[1]["lr"],
    })

    # Best model per F1 macro
    if val_metrics["f1_macro"] > best_val_f1:
        best_val_f1 = val_metrics["f1_macro"]
        torch.save(model.state_dict(),
                   CFG["output_dir"] / f"best_{run_name}.pt")

    print(f"Ep {epoch+1:02d} | loss={train_loss:.4f} | "
          f"train_f1={train_f1:.3f} | val_f1={val_metrics['f1_macro']:.3f} | "
          f"val_tar={val_metrics['tar_at_far']:.3f} | best_f1={best_val_f1:.3f}")


# %%
model.load_state_dict(torch.load(CFG["output_dir"] / f"best_{run_name}.pt",
                                  map_location=device))
model.eval()

test_metrics = evaluate(model, test_loader, device, CFG["far_threshold"])
test_bench   = benchmark_model(model, test_loader, device,
                               label=f"Pruned (layer={CFG['prune_layer']}, keep={int(CFG['keep_ratio']*100)}%)")

print(f"\n── Test Results ─────────────────────────")
print(f"  Accuracy:         {test_metrics['acc']:.3f}")
print(f"  F1 macro:         {test_metrics['f1_macro']:.3f}")
print(f"  TAR@FAR={CFG['far_threshold']:.0e}: {test_metrics['tar_at_far']:.3f}")
print(f"  ms/img:           {test_bench['ms_per_img']:.2f}")
print(f"  GFLOPs:           {test_bench['gflops']:.2f}")

wandb.log({
    "test/acc"       : test_metrics["acc"],
    "test/f1_macro"  : test_metrics["f1_macro"],
    "test/tar_at_far": test_metrics["tar_at_far"],
    "test/ms_per_img": test_bench["ms_per_img"],
    "test/gflops"    : test_bench["gflops"],
})
wandb.finish()


