"""Train one AttentionForecaster per dataset (unsupervised, frozen backbone target).

Loops sequentially through all available datasets, trains for --epochs epochs,
saves the best checkpoint to checkpoints/unsupervised/per_dataset/{dataset}/,
evaluates on the test split, and appends per-dataset Spearman rho to a CSV.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import csv
import sys
from pathlib import Path

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import set_seed, get_device
from src.models import AttentionForecaster
from src.data import MultiH5ForecastDataset


def spearman_correlation(y_pred, y_true):
    r_pred = y_pred.argsort(dim=-1).argsort(dim=-1).float()
    r_true = y_true.argsort(dim=-1).argsort(dim=-1).float()
    r_pred_m = r_pred - r_pred.mean(dim=-1, keepdim=True)
    r_true_m = r_true - r_true.mean(dim=-1, keepdim=True)
    num = (r_pred_m * r_true_m).sum(dim=-1)
    den = torch.sqrt((r_pred_m ** 2).sum(dim=-1) * (r_true_m ** 2).sum(dim=-1))
    return num / (den + 1e-8)


@torch.no_grad()
def _evaluate(forecaster, loader, device):
    forecaster.eval()
    rho_list = []
    for emb, target, _, _ in loader:
        emb, target = emb.to(device), target.to(device)
        logits = forecaster(emb)
        rho_list.append(spearman_correlation(logits, target).cpu())
    return torch.cat(rho_list).mean().item()


def train_one(dataset_name, cache_path, save_path, layer_source, layer_target,
              embed_dim, hidden, n_heads, n_layers, dropout,
              epochs, lr, weight_decay, batch_size, num_workers, device, seed):
    set_seed(seed)
    cache = {dataset_name: cache_path}
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=(num_workers > 0))
    train_ds = MultiH5ForecastDataset(cache, "train", layer_source, layer_target)
    val_ds   = MultiH5ForecastDataset(cache, "val",   layer_source, layer_target)
    test_ds  = MultiH5ForecastDataset(cache, "test",  layer_source, layer_target)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    n_steps = len(train_loader)
    print(f"  [{dataset_name}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"steps/ep={n_steps}")

    forecaster = AttentionForecaster(
        embed_dim=embed_dim, hidden=hidden, n_heads=n_heads,
        n_layers=n_layers, dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(forecaster.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_val_rho = -1.0
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        forecaster.train()
        for emb, target, _, _ in tqdm(train_loader, leave=False,
                                       desc=f"[{dataset_name}] Ep{epoch+1}"):
            emb, target = emb.to(device), target.to(device)
            with torch.amp.autocast("cuda"):
                logits = forecaster(emb)
                loss = F.kl_div(logits.log_softmax(-1), target, reduction='batchmean')
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        val_rho = _evaluate(forecaster, val_loader, device)
        sched.step()
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            torch.save(forecaster.state_dict(), save_path)

    forecaster.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_rho = _evaluate(forecaster, test_loader, device)
    print(f"  [{dataset_name}] best_val_rho={best_val_rho:.4f}  test_rho={test_rho:.4f}  ckpt={save_path}")
    return best_val_rho, test_rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir",    type=str, default="checkpoints/unsupervised")
    parser.add_argument("--layer-source", type=int, default=2)
    parser.add_argument("--layer-target", type=int, default=23)
    parser.add_argument("--embed-dim",    type=int, default=1024)
    parser.add_argument("--hidden",       type=int, default=256)
    parser.add_argument("--n-heads",      type=int, default=4)
    parser.add_argument("--n-layers",     type=int, default=2)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--batch-size",   type=int, default=128)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/unsupervised/per_dataset")
    parser.add_argument("--out-csv",        type=str, default="logs/per_dataset_eaf_rho.csv")
    args = parser.parse_args()

    device = get_device()
    cache_dir = Path(args.cache_dir)
    cache_paths = {p.stem.split("_uni_attn_features")[0]: p
                   for p in sorted(cache_dir.glob("*_uni_attn_features.h5"))}
    print(f"Datasets: {list(cache_paths.keys())}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "val_rho", "test_rho"])

        ckpt_base = Path(args.checkpoint_dir)
        for name, path in cache_paths.items():
            print(f"\n=== Training per-dataset EAF: {name} ===")
            ckpt_path = ckpt_base / name / f"forecaster_src{args.layer_source:02d}_attn{args.layer_target:02d}.pt"
            try:
                val_rho, test_rho = train_one(
                    name, path, ckpt_path, args.layer_source, args.layer_target,
                    args.embed_dim, args.hidden, args.n_heads, args.n_layers, args.dropout,
                    args.epochs, args.lr, args.weight_decay,
                    args.batch_size, args.num_workers, device, args.seed,
                )
                writer.writerow([name, round(val_rho, 6), round(test_rho, 6)])
                f.flush()
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")
                writer.writerow([name, "ERROR", "ERROR"])
                f.flush()

    print(f"\nDone. Results saved to {out_csv}")


if __name__ == "__main__":
    main()
