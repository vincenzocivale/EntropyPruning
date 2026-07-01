"""Phase 2 (multi-dataset): Train AttentionForecaster across all Thunder training datasets.

Reuses the holdout plan from Phase 1 (holdout_plan.json) to ensure the same
datasets are excluded. Feature extraction runs per-dataset using build_thunder_loaders;
the resulting HDF5 caches are concatenated for forecaster training.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, save_results
from src.models import AttentionForecaster, ThunderBackboneAdapter, build_classifier, STRATEGIES
from src.data.thunder_loaders import build_thunder_loaders
from src.data.h5_dataset import H5ForecastDataset
from src.data.thunder_multi import ThunderDatasetRegistry
from src.collection import collect_and_save_dataset


def _spearman(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r_pred = y_pred.argsort(-1).argsort(-1).float()
    r_true = y_true.argsort(-1).argsort(-1).float()
    rp = r_pred - r_pred.mean(-1, keepdim=True)
    rt = r_true - r_true.mean(-1, keepdim=True)
    return (rp * rt).sum(-1) / (
        torch.sqrt((rp ** 2).sum(-1) * (rt ** 2).sum(-1)) + 1e-8
    )


def _collect_features(
    dataset_name: str,
    base_data_folder: str,
    transform,
    model,
    device: torch.device,
    cache_path: Path,
    layers_source: list,
    layer_target: int,
    batch_size: int,
    num_workers: int,
):
    train_loader, val_loader, test_loader, _, n_classes = build_thunder_loaders(
        dataset_name, base_data_folder, transform,
        batch_size, num_workers, drop_last_train=False,
    )
    collect_and_save_dataset(
        model,
        {"train": train_loader, "val": val_loader, "test": test_loader},
        device,
        layers_source=layers_source,
        layers_target=[layer_target],
        save_path=cache_path,
    )
    print(f"  Cached: {cache_path}")


def _train_forecaster(layer_source: int, layer_target: int, cfg: dict,
                      cache_paths: list, device: torch.device) -> dict:
    run_name = (f"{cfg['model_name']}_multi_phase2"
                f"_src{layer_source:02d}_tgt{layer_target:02d}")
    print(f"\n{'='*60}\n  {run_name}\n{'='*60}")

    kw = dict(batch_size=128, num_workers=4, pin_memory=True, persistent_workers=True)
    train_ds = ConcatDataset([
        H5ForecastDataset(p, "train", layer_source, layer_target) for p in cache_paths
    ])
    val_ds = ConcatDataset([
        H5ForecastDataset(p, "val", layer_source, layer_target) for p in cache_paths
    ])
    test_ds = ConcatDataset([
        H5ForecastDataset(p, "test", layer_source, layer_target) for p in cache_paths
    ])
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    if cfg.get("wandb_project"):
        wandb.init(
            project=cfg["wandb_project"], name=run_name, job_type="phase2_multi",
            config={**{k: cfg[k] for k in ("model_name", "embed_dim", "hidden",
                                             "n_heads", "n_layers", "dropout",
                                             "epochs", "lr", "weight_decay")},
                    "layer_source": layer_source, "layer_target": layer_target,
                    "n_train_datasets": cfg["n_train_datasets"]},
            tags=[cfg["model_name"], f"src{layer_source}", f"tgt{layer_target}",
                  "phase2", "multi_dataset"],
            reinit=True,
        )

    forecaster = AttentionForecaster(
        embed_dim=cfg["embed_dim"],
        hidden=cfg["hidden"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(forecaster.parameters(),
                             lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda")
    best_val_kl, best_val_rho = float("inf"), -1.0
    save_path = cfg["forecaster_dir"] / f"forecaster_{run_name}.pt"

    for epoch in range(cfg["epochs"]):
        forecaster.train()
        train_kl = 0.
        for emb, target, _ in tqdm(train_loader, leave=False, desc=f"Ep{epoch+1} train"):
            emb, target = emb.to(device), target.to(device)
            with torch.amp.autocast("cuda"):
                logits = forecaster(emb)
                loss = F.kl_div(logits.log_softmax(-1), target, reduction="batchmean")
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_kl += loss.item()

        forecaster.eval()
        val_kl, rho_list = 0., []
        with torch.no_grad():
            for emb, target, _ in val_loader:
                emb, target = emb.to(device), target.to(device)
                logits = forecaster(emb)
                val_kl += F.kl_div(logits.log_softmax(-1), target, reduction="batchmean").item()
                rho_list.append(_spearman(logits, target))
        sched.step()

        train_kl /= len(train_loader)
        val_kl /= len(val_loader)
        val_rho = torch.cat(rho_list).mean().item()

        if cfg.get("wandb_project"):
            wandb.log({"epoch": epoch + 1, "train/kl": train_kl,
                       "val/kl": val_kl, "val/rho": val_rho,
                       "lr": sched.get_last_lr()[0]})

        if val_kl < best_val_kl:
            best_val_kl, best_val_rho = val_kl, val_rho
            torch.save(forecaster.state_dict(), save_path)
        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:02d} | kl={train_kl:.4f} val_kl={val_kl:.4f} "
                  f"val_rho={val_rho:.3f}")

    # Test
    forecaster.load_state_dict(torch.load(save_path, map_location=device))
    forecaster.eval()
    rho_f, rho_n = [], []
    with torch.no_grad():
        for emb, target, _ in test_loader:
            emb, target = emb.to(device), target.to(device)
            logits = forecaster(emb)
            rho_f.append(_spearman(logits, target))
            rho_n.append(_spearman(emb.norm(dim=-1), target))
    test_rho_f = torch.cat(rho_f).mean().item()
    test_rho_n = torch.cat(rho_n).mean().item()

    results = {
        "model_name": cfg["model_name"],
        "layer_source": layer_source, "layer_target": layer_target,
        "embed_dim": cfg["embed_dim"],
        "n_train_datasets": cfg["n_train_datasets"],
        "best_val_rho": round(best_val_rho, 6),
        "best_val_kl": round(best_val_kl, 6),
        "test_rho_forecaster": round(test_rho_f, 6),
        "test_rho_token_norm": round(test_rho_n, 6),
        "test_delta_vs_norm": round(test_rho_f - test_rho_n, 6),
    }
    if cfg.get("wandb_project"):
        wandb.log({"test/rho_forecaster": test_rho_f, "test/rho_token_norm": test_rho_n,
                   "test/delta_vs_norm": test_rho_f - test_rho_n,
                   "val/best_rho": best_val_rho, "val/best_kl": best_val_kl})
        wandb.finish()

    print(f"  Test rho forecaster: {test_rho_f:.3f}")
    print(f"  Test rho token norm: {test_rho_n:.3f}")
    print(f"  Delta vs baseline:  {test_rho_f - test_rho_n:+.3f}")
    save_results(save_path.parent / f"results_{run_name}.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 (multi-dataset): Train AttentionForecaster on Thunder datasets")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    # Holdout (one of these is required)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--holdout-plan", type=str,
                     help="Path to holdout_plan.json saved by Phase 1 (recommended)")
    grp.add_argument("--n-holdout", type=int,
                     help="Re-compute holdout from scratch using N smallest datasets")
    parser.add_argument("--holdout-datasets", type=str, nargs="+", default=None,
                        help="Explicit holdout names (only used with --n-holdout)")
    # Phase 1 checkpoint
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--adaptation", type=str, default="lora", choices=STRATEGIES)
    # Feature extraction / caching
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size-extract", type=int, default=64,
                        help="Batch size for feature extraction (default 64)")
    parser.add_argument("--num-workers", type=int, default=4)
    # Forecaster architecture
    parser.add_argument("--layers-source", type=int, nargs="+", default=[2])
    parser.add_argument("--layer-target", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # Output / W&B
    parser.add_argument("--forecaster-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # --- Registry ---
    if args.holdout_plan:
        registry = ThunderDatasetRegistry.from_plan(
            args.holdout_plan, args.base_data_folder)
    else:
        registry = ThunderDatasetRegistry(
            args.base_data_folder,
            n_holdout=args.n_holdout,
            holdout_datasets=args.holdout_datasets,
        )
    print(f"Train datasets ({len(registry.train_datasets)}): {registry.train_datasets}")
    print(f"Holdout datasets ({len(registry.holdout_datasets)}): {registry.holdout_datasets}")

    # --- Backbone ---
    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    print(f"embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}  "
          f"n_patches={adapter.n_patches}")

    layer_target = args.layer_target if args.layer_target is not None else adapter.n_blocks - 1
    for ls in args.layers_source:
        assert ls < adapter.n_blocks, f"--layers-source {ls} >= n_blocks"
    assert layer_target < adapter.n_blocks, f"--layer-target {layer_target} >= n_blocks"

    # --- Paths ---
    cache_dir = Path(args.cache_dir) if args.cache_dir else \
        Path("checkpoints") / "multi_thunder" / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    forecaster_dir = Path(args.forecaster_dir) if args.forecaster_dir else \
        Path("checkpoints") / "multi_thunder" / f"{args.model_name}_forecaster"
    forecaster_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Phase 1 classifier for feature extraction ---
    # Use a dummy n_classes=2 just to load the backbone (head is ignored)
    dummy_cls = build_classifier(args.adaptation, raw_backbone, adapter, n_classes=2)
    if args.classifier_ckpt:
        dummy_cls.load_state_dict(
            torch.load(args.classifier_ckpt, map_location=device), strict=False)
    dummy_cls.eval()
    for p in dummy_cls.parameters():
        p.requires_grad_(False)
    dummy_cls = dummy_cls.to(device)

    # --- Extract features per training dataset ---
    cache_paths: list = []
    for name in registry.train_datasets:
        cache_path = cache_dir / f"{name}_{args.model_name}_features.h5"
        cache_paths.append(cache_path)
        if cache_path.exists():
            print(f"Feature cache found: {cache_path}")
            continue
        print(f"\nExtracting features: {name}")
        _collect_features(
            name, args.base_data_folder, transform, dummy_cls, device,
            cache_path, args.layers_source, layer_target,
            args.batch_size_extract, args.num_workers,
        )

    # --- Train forecaster for each source layer ---
    cfg = dict(
        model_name=args.model_name, embed_dim=adapter.embed_dim,
        hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, wandb_project=args.wandb_project,
        forecaster_dir=forecaster_dir,
        n_train_datasets=len(registry.train_datasets),
    )
    all_results = []
    for ls in args.layers_source:
        all_results.append(_train_forecaster(ls, layer_target, cfg, cache_paths, device))

    print(f"\n{'Layer':>8} {'rho Forecaster':>14} {'rho Baseline':>12} {'Delta':>8}")
    print("-" * 46)
    for r in all_results:
        delta = r["test_rho_forecaster"] - r["test_rho_token_norm"]
        print(f"{r['layer_source']:>8} {r['test_rho_forecaster']:>14.3f} "
              f"{r['test_rho_token_norm']:>12.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
