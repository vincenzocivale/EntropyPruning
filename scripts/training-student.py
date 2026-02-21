# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timm
import types
import sys
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from peft import LoraConfig
from peft.tuners.lora import LoraModel
import h5py
import wandb


sys.path.append(".")
from src.dataset import HistologicalImageDataset

import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%
CFG = dict(
    data_dir        = "/data/BREAKHIS",
    img_size        = 224,
    batch_size      = 128,
    num_workers     = 8,
    seed            = 42,
    output_dir      = Path("checkpoints/uni_finetuned"),
    dataset_cache   = Path("/data/data_cache/forecaster_dataset.h5"),
    forecaster_dir  = Path("checkpoints/forecaster"),

    # Layer target fisso: ultimo blocco significativo
    layer_target    = 23,

    # Layer sorgente da testare sistematicamente
    layers_source   = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],

    # Architettura student
    hidden          = 256,
    n_heads         = 4,
    n_layers        = 2,
    dropout         = 0.2,

    # Training
    epochs          = 30,
    lr              = 1e-4,
    weight_decay    = 0.05,

    # W&B
    wandb_project   = "attention-forecaster",
)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
CFG["dataset_cache"].parent.mkdir(parents=True, exist_ok=True)
CFG["forecaster_dir"].mkdir(parents=True, exist_ok=True)


# %%
train_tf = T.Compose([
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
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
train_loader = DataLoader(train_ds, sampler=sampler, drop_last=False, **kw)
val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

CLASS_NAMES = train_ds.class_names
N_CLASSES   = len(CLASS_NAMES)
print(f"Classi: {CLASS_NAMES}")
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")


# %%
class UNILoRAClassifier(nn.Module):
    def __init__(self, n_classes, dropout=0.1):
        super().__init__()
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, dynamic_img_size=True
        )
        lora_config = LoraConfig(
            r=8, lora_alpha=32,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1, bias="none"
        )
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(
            nn.LayerNorm(1024), nn.Dropout(dropout), nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

model = UNILoRAClassifier(N_CLASSES).to(device)
ckpt  = torch.load(CFG["output_dir"] / "best_model.pt", map_location=device)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

with torch.no_grad():
    imgs, labs = next(iter(val_loader))
    acc = (model(imgs.to(device)).argmax(-1).cpu() == labs).float().mean()
print(f"Val batch accuracy: {acc:.3f}")

# %%
def collect_and_save_dataset(model, loaders_dict, device,
                              layers_source, layer_target, save_path):
    with h5py.File(save_path, 'w') as f:
        for split_name, loader in loaders_dict.items():
            print(f"\nRaccolta {split_name}...")

            n_total   = len(loader.dataset)
            n_patches = 196
            embed_dim = 1024

            grp      = f.create_group(split_name)
            ds_label = grp.create_dataset("labels",
                shape=(n_total,), maxshape=(None,),
                dtype='i4', compression="gzip")
            ds_attn  = grp.create_dataset(f"attn_layer{layer_target}",
                shape=(n_total, n_patches), maxshape=(None, n_patches),
                dtype='f2', compression="gzip", chunks=(64, n_patches))
            ds_embs  = {l: grp.create_dataset(f"emb_layer{l}",
                shape=(n_total, n_patches, embed_dim),
                maxshape=(None, n_patches, embed_dim),
                dtype='f2', compression="gzip",
                chunks=(64, n_patches, embed_dim))
                for l in layers_source}

            orig  = {}
            cache = {}

            def make_hook(idx):
                def fwd(self, x):
                    B, N, C = x.shape
                    qkv  = self.qkv(x).reshape(B,N,3,self.num_heads,
                                     self.head_dim).permute(2,0,3,1,4)
                    q, k, v = qkv.unbind(0)
                    q, k   = self.q_norm(q), self.k_norm(k)
                    attn   = (q @ k.transpose(-2,-1) * self.scale).softmax(-1)
                    if idx in layers_source:
                        cache[f"emb_{idx}"] = x[:,1:].detach().cpu().half()
                    if idx == layer_target:
                        cache["attn"] = attn[:,:,0,1:].mean(1).detach().cpu().half()
                    x = (self.attn_drop(attn) @ v).transpose(1,2).reshape(B,N,C)
                    return self.proj_drop(self.proj(x))
                return fwd

            for i, block in enumerate(model.backbone.model.blocks):
                orig[i] = block.attn.forward
                block.attn.forward = types.MethodType(make_hook(i), block.attn)

            FLUSH_EVERY = 16
            buf_labels  = []
            buf_attn    = []
            buf_embs    = {l: [] for l in layers_source}

            def flush(ptr):
                if not buf_labels:
                    return ptr
                B = sum(len(x) for x in buf_labels)
                ds_label[ptr:ptr+B] = np.concatenate(buf_labels)
                ds_attn[ptr:ptr+B]  = torch.cat(buf_attn).numpy()
                for l in layers_source:
                    ds_embs[l][ptr:ptr+B] = torch.cat(buf_embs[l]).numpy()
                buf_labels.clear(); buf_attn.clear()
                for l in layers_source:
                    buf_embs[l].clear()
                return ptr + B

            ptr = 0
            with torch.no_grad():
                for i_batch, (imgs, labels) in enumerate(tqdm(loader, desc=split_name)):
                    cache.clear()
                    model(imgs.to(device))
                    buf_labels.append(labels.numpy())
                    buf_attn.append(cache["attn"])
                    for l in layers_source:
                        buf_embs[l].append(cache[f"emb_{l}"])
                    if (i_batch + 1) % FLUSH_EVERY == 0:
                        ptr = flush(ptr)

            ptr = flush(ptr)  # residuo finale

            for i, block in enumerate(model.backbone.model.blocks):
                block.attn.forward = orig[i]

            if ptr < n_total:
                ds_label.resize(ptr, axis=0)
                ds_attn.resize(ptr, axis=0)
                for l in layers_source:
                    ds_embs[l].resize(ptr, axis=0)

            print(f"  {split_name}: {ptr} sample salvati")


# Esegui raccolta solo se il file non esiste già
if not CFG["dataset_cache"].exists():
    collect_and_save_dataset(
        model,
        {"train": train_loader, "val": val_loader, "test": test_loader},
        device,
        layers_source=CFG["layers_source"],
        layer_target=CFG["layer_target"],
        save_path=CFG["dataset_cache"]
    )
else:
    print(f"Dataset cache trovata: {CFG['dataset_cache']}")
    with h5py.File(CFG["dataset_cache"], 'r') as f:
        for split in f.keys():
            n = len(f[split]["labels"])
            print(f"  {split}: {n} sample")

# %%
class H5ForecastDataset(torch.utils.data.Dataset):
    """
    Dataset lazy-loading da HDF5.
    Carica embedding @ layer_source e attenzione @ layer_target.
    """
    def __init__(self, h5_path, split, layer_source, layer_target):
        self.h5_path      = str(h5_path)
        self.split        = split
        self.layer_source = layer_source
        self.layer_target = layer_target
        self._file        = None

        # Leggi solo la lunghezza senza caricare tutto in RAM
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[split]["labels"])

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f   = self._get_file()
        grp = f[self.split]
        emb    = torch.from_numpy(grp[f"emb_layer{self.layer_source}"][idx]).float()
        target = torch.from_numpy(grp[f"attn_layer{self.layer_target}"][idx]).float()
        label  = int(grp["labels"][idx])
        return emb, target, label

# %%
class AttentionForecaster(nn.Module):
    def __init__(self, embed_dim=1024, hidden=256,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.cls_query  = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        self.self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        self.norm       = nn.LayerNorm(hidden)
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, patch_embeddings):
        B, N, D = patch_embeddings.shape
        x = self.input_proj(patch_embeddings)

        for sa in self.self_attn:
            x = sa(x)

        cls = self.cls_query.expand(B, -1, -1)
        for ca, norm in zip(self.cross_attn, self.cross_norms):
            cls_out, _ = ca(cls, x, x)
            cls = norm(cls + cls_out)

        x_norm  = self.norm(x)
        cls_exp = cls.expand(-1, N, -1)
        scores  = self.score_head(torch.cat([x_norm, cls_exp], dim=-1)).squeeze(-1)
        return scores.softmax(-1)

# %%
def train_forecaster(layer_source, layer_target, cfg, device):
    """
    Addestra un AttentionForecaster per la coppia (layer_source, layer_target).
    Logga su W&B con nome run = f"src{layer_source}_tgt{layer_target}"
    """
    run_name = f"src{layer_source:02d}_tgt{layer_target:02d}"
    print(f"\n{'='*60}")
    print(f"  Esperimento: {run_name}")
    print(f"{'='*60}")

    # W&B init
    wandb.init(
        project = cfg["wandb_project"],
        name    = run_name,
        config  = {
            "layer_source"  : layer_source,
            "layer_target"  : layer_target,
            "hidden"        : cfg["hidden"],
            "n_heads"       : cfg["n_heads"],
            "n_layers"      : cfg["n_layers"],
            "dropout"       : cfg["dropout"],
            "epochs"        : cfg["epochs"],
            "lr"            : cfg["lr"],
            "weight_decay"  : cfg["weight_decay"],
        },
        reinit = True
    )

    # Dataset
    train_ds_h5 = H5ForecastDataset(cfg["dataset_cache"], "train",
                                     layer_source, layer_target)
    val_ds_h5   = H5ForecastDataset(cfg["dataset_cache"], "val",
                                     layer_source, layer_target)
    test_ds_h5  = H5ForecastDataset(cfg["dataset_cache"], "test",
                                     layer_source, layer_target)

    kw_h5 = dict(batch_size=64, num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader_h5 = DataLoader(train_ds_h5, shuffle=True,  **kw_h5)
    val_loader_h5   = DataLoader(val_ds_h5,   shuffle=False, **kw_h5)
    test_loader_h5  = DataLoader(test_ds_h5,  shuffle=False, **kw_h5)

    # Modello
    forecaster = AttentionForecaster(
        embed_dim = 1024,
        hidden    = cfg["hidden"],
        n_heads   = cfg["n_heads"],
        n_layers  = cfg["n_layers"],
        dropout   = cfg["dropout"],
    ).to(device)

    opt   = torch.optim.AdamW(forecaster.parameters(),
                               lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best_val_kl  = float('inf')
    best_val_rho = -1.0
    save_path    = cfg["forecaster_dir"] / f"forecaster_{run_name}.pt"

    for epoch in range(cfg["epochs"]):
        # ── Train ──
        forecaster.train()
        train_kl, train_rho_list = 0., []

        for emb, target, _ in tqdm(train_loader_h5, leave=False,
                                    desc=f"Ep{epoch+1} train"):
            emb, target = emb.to(device), target.to(device)
            pred = forecaster(emb)

            loss_kl  = F.kl_div((pred + 1e-8).log(), target + 1e-8,
                                  reduction='batchmean')
            loss_mse = F.mse_loss(pred, target)
            loss     = loss_kl + 0.1 * loss_mse

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
            opt.step()

            train_kl += loss_kl.item()
            with torch.no_grad():
                for b in range(len(emb)):
                    rho, _ = spearmanr(pred[b].cpu().numpy(), target[b].cpu().numpy())
                    train_rho_list.append(rho)

        # ── Val ──
        forecaster.eval()
        val_kl, val_rho_list = 0., []
        with torch.no_grad():
            for emb, target, _ in val_loader_h5:
                emb, target = emb.to(device), target.to(device)
                pred = forecaster(emb)
                val_kl += F.kl_div((pred + 1e-8).log(), target + 1e-8,
                                    reduction='batchmean').item()
                for b in range(len(emb)):
                    rho, _ = spearmanr(pred[b].cpu().numpy(), target[b].cpu().numpy())
                    val_rho_list.append(rho)

        sched.step()

        train_kl  /= len(train_loader_h5)
        val_kl    /= len(val_loader_h5)
        train_rho  = np.nanmean(train_rho_list)
        val_rho    = np.nanmean(val_rho_list)

        # Log W&B
        wandb.log({
            "epoch"     : epoch + 1,
            "train/kl"  : train_kl,
            "val/kl"    : val_kl,
            "train/rho" : train_rho,
            "val/rho"   : val_rho,
            "lr"        : sched.get_last_lr()[0],
        })

        if val_kl < best_val_kl:
            best_val_kl  = val_kl
            best_val_rho = val_rho
            torch.save(forecaster.state_dict(), save_path)

        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:02d} | "
                  f"train_kl={train_kl:.4f} val_kl={val_kl:.4f} | "
                  f"train_ρ={train_rho:.3f} val_ρ={val_rho:.3f}")

    # ── Test con best checkpoint ──
    forecaster.load_state_dict(torch.load(save_path))
    forecaster.eval()

    test_rho_forecaster = []
    test_rho_cls_x      = []   # baseline: CLS attn @ layer_source (dal file H5)
    test_rho_token_norm = []   # baseline: norma token @ layer_source

    with h5py.File(cfg["dataset_cache"], 'r') as f_h5:
        grp         = f_h5["test"]
        emb_all     = torch.from_numpy(grp[f"emb_layer{layer_source}"][:]).float()
        target_all  = torch.from_numpy(grp[f"attn_layer{layer_target}"][:]).float()

    test_batch_ds = DataLoader(TensorDataset(emb_all, target_all),
                               batch_size=64, shuffle=False)

    with torch.no_grad():
        for emb, target in test_batch_ds:
            emb = emb.to(device)
            pred = forecaster(emb).cpu()
            for b in range(len(emb)):
                t = target[b].numpy()
                rho_f,  _ = spearmanr(pred[b].numpy(), t)
                rho_n,  _ = spearmanr(emb[b].cpu().norm(dim=-1).numpy(), t)
                test_rho_forecaster.append(rho_f)
                test_rho_token_norm.append(rho_n)

    test_rho_f = np.nanmean(test_rho_forecaster)
    test_rho_n = np.nanmean(test_rho_token_norm)

    wandb.log({
        "test/rho_forecaster" : test_rho_f,
        "test/rho_token_norm" : test_rho_n,
        "test/delta_vs_norm"  : test_rho_f - test_rho_n,
        "best_val_rho"        : best_val_rho,
        "best_val_kl"         : best_val_kl,
    })

    print(f"\n  Test ρ forecaster:   {test_rho_f:.3f}")
    print(f"  Test ρ token norm:   {test_rho_n:.3f}")
    print(f"  Δ vs baseline:      {test_rho_f - test_rho_n:+.3f}")

    wandb.finish()

    return {
        "layer_source"       : layer_source,
        "layer_target"       : layer_target,
        "best_val_rho"       : best_val_rho,
        "best_val_kl"        : best_val_kl,
        "test_rho_forecaster": test_rho_f,
        "test_rho_token_norm": test_rho_n,
    }

# %%
all_results = []

for layer_source in CFG["layers_source"]:
    result = train_forecaster(
        layer_source = layer_source,
        layer_target = CFG["layer_target"],
        cfg          = CFG,
        device       = device,
    )
    all_results.append(result)

# %%
layers   = [r["layer_source"]        for r in all_results]
rho_fore = [r["test_rho_forecaster"] for r in all_results]
rho_norm = [r["test_rho_token_norm"] for r in all_results]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(layers, rho_fore, marker='o', linewidth=2, color='tomato',
        label="AttentionForecaster (student)")
ax.plot(layers, rho_norm, marker='s', linewidth=2, color='gray',
        linestyle='--', label="Token norm baseline")
ax.fill_between(layers, rho_norm, rho_fore, alpha=0.15, color='tomato')
ax.set_xlabel("Layer sorgente (input student)", fontsize=13)
ax.set_ylabel("Spearman ρ (test)", fontsize=13)
ax.set_title(f"Predizione importanza patch @ layer {CFG['layer_target']}\n"
             f"da layer sorgente variabile", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xticks(layers)
plt.tight_layout()
plt.savefig("checkpoints/forecaster/rho_vs_layer.png", dpi=150)
plt.show()

# Stampa tabella riassuntiva
print(f"\n{'Layer':>8} {'ρ Forecaster':>14} {'ρ Baseline':>12} {'Δ':>8}")
print("─" * 46)
for r in all_results:
    delta = r["test_rho_forecaster"] - r["test_rho_token_norm"]
    print(f"{r['layer_source']:>8} {r['test_rho_forecaster']:>14.3f} "
          f"{r['test_rho_token_norm']:>12.3f} {delta:>+8.3f}")


