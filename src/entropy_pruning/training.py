from __future__ import annotations

import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .models import AttentionForecaster, UNILoRAWithForecasterPruning


class H5ForecastDataset(Dataset):
    def __init__(self, h5_path, split, layer_source, layer_target):
        self.h5_path = str(h5_path)
        self.split = split
        self.layer_source = layer_source
        self.layer_target = layer_target
        self._file = None
        with self._open_h5_read() as f:
            requested = split
            if requested in f:
                resolved = requested
            elif requested == "val" and "validation" in f:
                resolved = "validation"
            elif requested == "validation" and "val" in f:
                resolved = "val"
            else:
                available = list(f.keys())
                raise KeyError(
                    f"Split `{requested}` not found in {self.h5_path}. "
                    f"Available splits: {available}"
                )

            grp = f[resolved]
            if "labels" not in grp:
                raise KeyError(
                    f"Split `{resolved}` exists in {self.h5_path} but dataset `labels` is missing."
                )

            self.split = resolved
            self.length = len(grp["labels"])

    def _open_h5_read(self):
        try:
            return h5py.File(self.h5_path, "r")
        except BlockingIOError:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            try:
                return h5py.File(self.h5_path, "r", locking=False)
            except TypeError:
                # Older h5py versions may not support `locking=...`.
                return h5py.File(self.h5_path, "r")

    def _get_file(self):
        if self._file is None:
            try:
                self._file = h5py.File(self.h5_path, "r", swmr=True)
            except BlockingIOError:
                os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
                try:
                    self._file = h5py.File(self.h5_path, "r", swmr=True, locking=False)
                except TypeError:
                    self._file = h5py.File(self.h5_path, "r", swmr=True)
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()
        grp = f[self.split]
        emb = torch.from_numpy(grp[f"emb_layer{self.layer_source}"][idx]).float()
        target = torch.from_numpy(grp[f"attn_layer{self.layer_target}"][idx]).float()
        label = int(grp["labels"][idx])
        return emb, target, label


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_classifier(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            preds = logits.argmax(-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    acc = float((y_pred == y_true).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"acc": acc, "f1_macro": f1}


def train_classifier(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr_backbone=1e-5,
    lr_head=1e-3,
    weight_decay=0.01,
    label_smoothing=0.1,
    save_path=None,
):
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr_backbone},
            {"params": model.head.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(-1) == labels).sum().item()
            total_count += len(labels)

        model.eval()
        val_loss, val_correct, val_count = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(-1) == labels).sum().item()
                val_count += len(labels)

        sched.step()
        tr_loss = total_loss / max(1, total_count)
        tr_acc = total_correct / max(1, total_count)
        vl_loss = val_loss / max(1, val_count)
        vl_acc = val_correct / max(1, val_count)

        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_loss"].append(float(vl_loss))
        history["val_acc"].append(float(vl_acc))

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save(model.state_dict(), save_path)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return {"best_val_acc": float(best_val_acc), "history": history}


def train_forecaster(
    h5_cache_path,
    layer_source,
    layer_target,
    device,
    hidden=256,
    n_heads=4,
    n_layers=2,
    dropout=0.2,
    batch_size=64,
    num_workers=0,
    epochs=30,
    lr=1e-4,
    weight_decay=0.05,
    save_path=None,
):
    train_ds = H5ForecastDataset(h5_cache_path, "train", layer_source, layer_target)
    val_ds = H5ForecastDataset(h5_cache_path, "val", layer_source, layer_target)
    test_ds = H5ForecastDataset(h5_cache_path, "test", layer_source, layer_target)

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    forecaster = AttentionForecaster(
        embed_dim=1024,
        hidden=hidden,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(forecaster.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_kl = float("inf")
    best_val_rho = -1.0

    for _ in range(epochs):
        forecaster.train()
        for emb, target, _ in tqdm(train_loader, leave=False):
            emb, target = emb.to(device), target.to(device)
            pred = forecaster(emb)

            loss_kl = F.kl_div((pred + 1e-8).log(), target + 1e-8, reduction="batchmean")
            loss_mse = F.mse_loss(pred, target)
            loss = loss_kl + 0.1 * loss_mse

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
            opt.step()

        forecaster.eval()
        val_kl, val_rhos = 0.0, []
        with torch.no_grad():
            for emb, target, _ in val_loader:
                emb, target = emb.to(device), target.to(device)
                pred = forecaster(emb)
                val_kl += F.kl_div((pred + 1e-8).log(), target + 1e-8, reduction="batchmean").item()
                pred_np = pred.cpu().numpy()
                tgt_np = target.cpu().numpy()
                for p, t in zip(pred_np, tgt_np):
                    rho, _ = spearmanr(p, t)
                    val_rhos.append(rho)

        val_kl /= max(1, len(val_loader))
        val_rho = float(np.nanmean(val_rhos))

        if val_kl < best_val_kl:
            best_val_kl = val_kl
            best_val_rho = val_rho
            if save_path is not None:
                torch.save(forecaster.state_dict(), save_path)

        sched.step()

    if save_path is not None:
        forecaster.load_state_dict(torch.load(save_path, map_location=device))

    forecaster.eval()
    test_rho_forecaster, test_rho_norm = [], []
    with torch.no_grad():
        for emb, target, _ in test_loader:
            emb_dev = emb.to(device)
            pred = forecaster(emb_dev).cpu().numpy()
            tgt = target.numpy()
            norms = emb.norm(dim=-1).numpy()
            for p, t, n in zip(pred, tgt, norms):
                rho_f, _ = spearmanr(p, t)
                rho_n, _ = spearmanr(n, t)
                test_rho_forecaster.append(rho_f)
                test_rho_norm.append(rho_n)

    return {
        "layer_source": layer_source,
        "layer_target": layer_target,
        "best_val_kl": float(best_val_kl),
        "best_val_rho": float(best_val_rho),
        "test_rho_forecaster": float(np.nanmean(test_rho_forecaster)),
        "test_rho_token_norm": float(np.nanmean(test_rho_norm)),
        "model": forecaster,
    }


def finetune_pruned_classifier(
    n_classes,
    classifier_ckpt,
    forecaster,
    prune_layer,
    keep_ratio,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr_backbone=1e-4,
    lr_head=1e-3,
    weight_decay=0.01,
    label_smoothing=0.1,
):
    model = UNILoRAWithForecasterPruning(
        n_classes=n_classes,
        forecaster=forecaster,
        prune_layer=prune_layer,
        keep_ratio=keep_ratio,
    ).to(device)

    ckpt = torch.load(classifier_ckpt, map_location=device)
    model.load_state_dict(ckpt, strict=False)

    for p in model.forecaster.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr_backbone},
            {"params": model.head.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    total_steps = max(1, epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[lr_backbone, lr_head],
        total_steps=total_steps,
        pct_start=0.1,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_f1 = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for imgs, labels in tqdm(train_loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        val_metrics = evaluate_classifier(model, val_loader, device)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_metrics = evaluate_classifier(model, test_loader, device)
    return {
        "val_best_f1": float(best_val_f1),
        "test_acc": test_metrics["acc"],
        "test_f1_macro": test_metrics["f1_macro"],
    }
