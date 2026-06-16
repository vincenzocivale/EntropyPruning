"""Quick evaluation of a saved AttentionForecaster checkpoint — generates a spider plot
of per-dataset Spearman rho on the test set."""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import argparse
from pathlib import Path

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import AttentionForecaster
from src.data import MultiH5ForecastDataset


def spearman_correlation(y_pred, y_true):
    r_pred = y_pred.argsort(dim=-1).argsort(dim=-1).float()
    r_true = y_true.argsort(dim=-1).argsort(dim=-1).float()
    r_pred_m = r_pred - r_pred.mean(dim=-1, keepdim=True)
    r_true_m = r_true - r_true.mean(dim=-1, keepdim=True)
    num = (r_pred_m * r_true_m).sum(dim=-1)
    den = torch.sqrt((r_pred_m**2).sum(dim=-1) * (r_true_m**2).sum(dim=-1))
    return num / (den + 1e-8)


@torch.no_grad()
def eval_per_dataset(checkpoint_path, cache_dir, layer_source, layer_target,
                     embed_dim, hidden, n_heads, n_layers, dropout,
                     batch_size, num_workers, device):
    cache_dir = Path(cache_dir)
    cache_paths = {p.stem.split("_uni_attn_features")[0]: p
                   for p in sorted(cache_dir.glob("*_uni_attn_features.h5"))}
    if not cache_paths:
        raise FileNotFoundError(f"No HDF5 caches found in {cache_dir}")
    print(f"Datasets found: {list(cache_paths.keys())}")

    test_ds = MultiH5ForecastDataset(cache_paths, "test", layer_source, layer_target)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0))

    forecaster = AttentionForecaster(
        embed_dim=embed_dim, hidden=hidden, n_heads=n_heads,
        n_layers=n_layers, dropout=dropout,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    forecaster.load_state_dict(state)
    forecaster.eval()

    rho_list, norm_rho_list, ds_idx_list = [], [], []
    for emb, target, _, ds_idx in loader:
        emb, target = emb.to(device), target.to(device)
        logits = forecaster(emb)
        rho_list.append(spearman_correlation(logits, target).cpu())
        norm_rho_list.append(spearman_correlation(emb.norm(dim=-1), target).cpu())
        ds_idx_list.append(ds_idx)

    rho_all = torch.cat(rho_list)
    norm_rho_all = torch.cat(norm_rho_list)
    ds_idx_all = torch.cat(ds_idx_list)

    names = list(cache_paths.keys())
    per_ds = {}
    for i, name in enumerate(names):
        mask = ds_idx_all == i
        if mask.any():
            per_ds[name] = {
                "forecaster": rho_all[mask].mean().item(),
                "baseline": norm_rho_all[mask].mean().item(),
            }
    return per_ds


def spider_plot(per_ds, out_path):
    names = list(per_ds.keys())
    rho_f = [per_ds[n]["forecaster"] for n in names]
    rho_b = [per_ds[n]["baseline"] for n in names]

    N = len(names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    rho_f_plot = rho_f + rho_f[:1]
    rho_b_plot = rho_b + rho_b[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    ax.fill(angles, rho_f_plot, alpha=0.25, color="#2196F3")
    ax.plot(angles, rho_f_plot, color="#2196F3", linewidth=2, label="Forecaster")
    ax.fill(angles, rho_b_plot, alpha=0.15, color="#FF5722")
    ax.plot(angles, rho_b_plot, color="#FF5722", linewidth=1.5, linestyle="--", label="Token-norm baseline")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, size=10)
    ax.set_rlabel_position(30)
    ax.set_ylim(min(min(rho_f), min(rho_b)) - 0.05, 1.0)

    avg_f = np.mean(rho_f)
    avg_b = np.mean(rho_b)
    ax.set_title(
        f"Per-dataset Spearman ρ — AttentionForecaster (UNI, h256)\n"
        f"avg forecaster={avg_f:.3f}  avg baseline={avg_b:.3f}  Δ={avg_f - avg_b:+.3f}",
        size=12, pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Spider plot saved to {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/unsupervised/uni_forecaster/forecaster_src02_attn23_universal_backup.pt")
    parser.add_argument("--cache-dir", type=str, default="checkpoints/unsupervised")
    parser.add_argument("--layer-source", type=int, default=2)
    parser.add_argument("--layer-target", type=int, default=23)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="logs/spider_rho_h256.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Checkpoint: {args.checkpoint}")

    per_ds = eval_per_dataset(
        args.checkpoint, args.cache_dir, args.layer_source, args.layer_target,
        args.embed_dim, args.hidden, args.n_heads, args.n_layers, args.dropout,
        args.batch_size, args.num_workers, device,
    )

    print(f"\n{'Dataset':>25} {'Forecaster':>10} {'Baseline':>10} {'Delta':>8}")
    print("-" * 58)
    for name, v in per_ds.items():
        print(f"{name:>25} {v['forecaster']:>10.3f} {v['baseline']:>10.3f} {v['forecaster']-v['baseline']:>+8.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spider_plot(per_ds, out_path)


if __name__ == "__main__":
    main()
