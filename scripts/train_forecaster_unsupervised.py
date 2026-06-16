"""Train ONE AttentionForecaster on a merged multi-dataset corpus to imitate
a frozen foundation model's own last-layer CLS->patch attention.

Unlike ``train_forecaster.py`` (Phase 2, per-dataset, requires a Phase-1
fine-tuned checkpoint), this script:
  - Skips Phase 1 entirely -- the backbone is the raw pretrained FM.
  - Builds/reuses per-dataset attention caches via ``build_attention_cache``.
  - Trains a single universal forecaster on the concatenation of all caches.
  - Reports an aggregate test metric AND a per-dataset breakdown.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
from pathlib import Path

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, save_results
from src.models import AttentionForecaster
from src.data import MultiH5ForecastDataset
from src.collection import build_attention_cache, build_frozen_model

DEFAULT_DATASETS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
    "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
    "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds",
]


def spearman_correlation(y_pred, y_true):
    """Vectorized Spearman rank correlation in PyTorch."""
    r_pred = y_pred.argsort(dim=-1).argsort(dim=-1).float()
    r_true = y_true.argsort(dim=-1).argsort(dim=-1).float()
    r_pred_m = r_pred - r_pred.mean(dim=-1, keepdim=True)
    r_true_m = r_true - r_true.mean(dim=-1, keepdim=True)
    num = (r_pred_m * r_true_m).sum(dim=-1)
    den = torch.sqrt((r_pred_m**2).sum(dim=-1) * (r_true_m**2).sum(dim=-1))
    return num / (den + 1e-8)


def _per_dataset_mean(values, ds_idx, dataset_names):
    out = {}
    for i, name in enumerate(dataset_names):
        mask = ds_idx == i
        if mask.any():
            out[name] = values[mask].mean().item()
    return out


@torch.no_grad()
def _evaluate(forecaster, loader, device, dataset_names, baseline=False):
    forecaster.eval()
    total_kl, n_batches = 0., 0
    rho_f_list, rho_n_list, ds_idx_list = [], [], []
    for emb, target, _, ds_idx in loader:
        emb, target = emb.to(device), target.to(device)
        logits = forecaster(emb)
        total_kl += F.kl_div(logits.log_softmax(-1), target, reduction='batchmean').item()
        n_batches += 1
        rho_f_list.append(spearman_correlation(logits, target).cpu())
        if baseline:
            rho_n_list.append(spearman_correlation(emb.norm(dim=-1), target).cpu())
        ds_idx_list.append(ds_idx)

    rho_f = torch.cat(rho_f_list)
    ds_idx_t = torch.cat(ds_idx_list)
    result = {
        "kl": total_kl / n_batches,
        "rho": rho_f.mean().item(),
        "per_dataset_rho": _per_dataset_mean(rho_f, ds_idx_t, dataset_names),
    }
    if baseline:
        rho_n = torch.cat(rho_n_list)
        result["rho_token_norm"] = rho_n.mean().item()
        result["per_dataset_rho_token_norm"] = _per_dataset_mean(rho_n, ds_idx_t, dataset_names)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train a universal, multi-dataset AttentionForecaster")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--layers-source", type=int, nargs="+", default=[2],
                        help="Block indices to extract patch embeddings from (e.g. 1 2 3 4 5). "
                             "When multiple are given their embeddings are concatenated.")
    parser.add_argument("--layer-target", type=int, default=None,
                        help="Defaults to last block (n_blocks-1).")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Per-dataset cache dir (default: checkpoints/unsupervised).")
    parser.add_argument("--max-samples-per-split", type=int, default=None,
                        help="Debug cap on samples per split (default: full datasets).")
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="attention-forecaster")
    parser.add_argument("--forecaster-dir", type=str, default=None,
                        help="Override forecaster checkpoint output dir.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Model: {args.model_name} | Datasets requested: {len(args.datasets)}")

    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    model, adapter = build_frozen_model(args.model_name, raw_backbone, device)
    layer_target = args.layer_target if args.layer_target is not None else adapter.n_blocks - 1
    forecaster_embed_dim = adapter.embed_dim * len(args.layers_source)
    print(f"embed_dim={adapter.embed_dim} n_blocks={adapter.n_blocks} n_patches={adapter.n_patches} "
          f"| layers_source={args.layers_source} layer_target={layer_target} "
          f"forecaster_embed_dim={forecaster_embed_dim}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("checkpoints") / "unsupervised"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = {}
    for dataset_name in args.datasets:
        split_path = Path(args.base_data_folder) / "data_splits" / f"{dataset_name}.json"
        if not split_path.exists():
            print(f"[{dataset_name}] SKIP: missing data split {split_path} "
                  f"(thunder download {dataset_name} --make-splits "
                  f"--base-data-folder {args.base_data_folder})")
            continue
        save_path = cache_dir / f"{dataset_name}_{args.model_name}_attn_features.h5"
        build_attention_cache(
            model, adapter, transform, dataset_name, args.base_data_folder, save_path, device,
            layers_source=args.layers_source, layer_target=layer_target,
            batch_size=args.cache_batch_size, num_workers=args.cache_num_workers,
            max_samples_per_split=args.max_samples_per_split,
        )
        cache_paths[dataset_name] = save_path

    if not cache_paths:
        raise RuntimeError("No dataset caches available -- check --base-data-folder / data splits.")
    print(f"Training corpus ({len(cache_paths)} datasets): {list(cache_paths.keys())}")

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
              persistent_workers=(args.num_workers > 0))
    train_ds = MultiH5ForecastDataset(cache_paths, "train", args.layers_source, layer_target)
    val_ds = MultiH5ForecastDataset(cache_paths, "val", args.layers_source, layer_target)
    test_ds = MultiH5ForecastDataset(cache_paths, "test", args.layers_source, layer_target)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Steps/epoch: {len(train_loader)}")

    src_tag = "src" + "+".join(f"{ls:02d}" for ls in args.layers_source)
    run_name = f"{args.model_name}_universal_{src_tag}_attn{layer_target:02d}"
    wandb.init(
        project=args.wandb_project, name=run_name, job_type="phase2_unsupervised",
        group=f"unsupervised/{args.model_name}",
        config={"model_name": args.model_name, "embed_dim": adapter.embed_dim,
                "forecaster_embed_dim": forecaster_embed_dim,
                "hidden": args.hidden, "n_heads": args.n_heads, "n_layers": args.n_layers,
                "dropout": args.dropout, "epochs": args.epochs, "lr": args.lr,
                "weight_decay": args.weight_decay, "layers_source": args.layers_source,
                "layer_target": layer_target, "datasets": list(cache_paths.keys())},
        tags=[args.model_name, src_tag, f"tgt{layer_target}", "phase2", "unsupervised"],
        reinit=True,
    )

    forecaster = AttentionForecaster(
        embed_dim=forecaster_embed_dim, hidden=args.hidden, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(forecaster.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    forecaster_dir = Path(args.forecaster_dir) if args.forecaster_dir else \
        cache_dir / f"{args.model_name}_forecaster"
    forecaster_dir.mkdir(parents=True, exist_ok=True)
    save_path = forecaster_dir / f"forecaster_{args.model_name}_{src_tag}_attn{layer_target:02d}_universal.pt"

    best_val_kl, best_val_rho = float('inf'), -1.0
    for epoch in range(args.epochs):
        forecaster.train()
        train_kl = 0.
        for emb, target, _, _ in tqdm(train_loader, leave=False, desc=f"Ep{epoch+1} train"):
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
            train_kl += loss.item()
        train_kl /= len(train_loader)

        val_metrics = _evaluate(forecaster, val_loader, device, list(cache_paths.keys()))
        sched.step()

        wandb.log({"epoch": epoch + 1, "train/kl": train_kl, "val/kl": val_metrics["kl"],
                   "val/rho": val_metrics["rho"], "lr": sched.get_last_lr()[0]})
        if val_metrics["kl"] < best_val_kl:
            best_val_kl, best_val_rho = val_metrics["kl"], val_metrics["rho"]
            torch.save(forecaster.state_dict(), save_path)
        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:02d} | train_kl={train_kl:.4f} val_kl={val_metrics['kl']:.4f} "
                  f"val_rho={val_metrics['rho']:.3f}")

    # Test
    forecaster.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = _evaluate(forecaster, test_loader, device, list(cache_paths.keys()), baseline=True)

    print(f"\n{'Dataset':>20} {'rho Forecaster':>14} {'rho TokenNorm':>14} {'Delta':>8}")
    print("-" * 60)
    per_dataset_results = {}
    for name in cache_paths:
        rf = test_metrics["per_dataset_rho"].get(name, float('nan'))
        rn = test_metrics["per_dataset_rho_token_norm"].get(name, float('nan'))
        per_dataset_results[name] = {
            "rho_forecaster": round(rf, 6), "rho_token_norm": round(rn, 6),
            "delta": round(rf - rn, 6),
        }
        print(f"{name:>20} {rf:>14.3f} {rn:>14.3f} {rf - rn:>+8.3f}")

    results = {
        "model_name": args.model_name,
        "datasets": list(cache_paths.keys()),
        "layers_source": args.layers_source,
        "layer_target": layer_target,
        "embed_dim": adapter.embed_dim,
        "best_val_rho": round(best_val_rho, 6),
        "best_val_kl": round(best_val_kl, 6),
        "test_rho_forecaster": round(test_metrics["rho"], 6),
        "test_rho_token_norm": round(test_metrics["rho_token_norm"], 6),
        "test_delta_vs_norm": round(test_metrics["rho"] - test_metrics["rho_token_norm"], 6),
        "per_dataset": per_dataset_results,
    }
    wandb.log({
        "test/rho_forecaster": test_metrics["rho"],
        "test/rho_token_norm": test_metrics["rho_token_norm"],
        "test/delta_vs_norm": test_metrics["rho"] - test_metrics["rho_token_norm"],
        "val/best_rho": best_val_rho, "val/best_kl": best_val_kl,
    })
    print(f"\n  Test rho forecaster (avg): {test_metrics['rho']:.3f}")
    print(f"  Test rho token norm (avg): {test_metrics['rho_token_norm']:.3f}")
    print(f"  Delta vs baseline:         {test_metrics['rho'] - test_metrics['rho_token_norm']:+.3f}")
    wandb.finish()

    results_path = save_results(save_path.parent / f"results_{save_path.stem}.json", results)
    print(f"Forecaster saved to: {save_path}")
    print(f"Results saved to:    {results_path}")


if __name__ == "__main__":
    main()
