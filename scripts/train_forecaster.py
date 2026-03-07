"""Phase 2: Train AttentionForecaster to predict target-layer attention
from source-layer embeddings."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import h5py
import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import set_seed, get_device
from src.models import UNILoRAClassifier, AttentionForecaster
from src.data.loaders import build_loaders
from src.data.h5_dataset import H5ForecastDataset
from src.collection import collect_and_save_dataset


def train_forecaster(layer_source, layer_target, cfg, device):
    run_name = f"src{layer_source:02d}_tgt{layer_target:02d}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {run_name}")
    print(f"{'='*60}")

    wandb.init(
        project=cfg["wandb_project"], name=run_name,
        config={
            "layer_source": layer_source, "layer_target": layer_target,
            "hidden": cfg["hidden"], "n_heads": cfg["n_heads"],
            "n_layers": cfg["n_layers"], "dropout": cfg["dropout"],
            "epochs": cfg["epochs"], "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
        },
        tags=[f"src{layer_source}", f"tgt{layer_target}",
              cfg["dataset_name"]],
        reinit=True,
    )

    kw_h5 = dict(batch_size=64, num_workers=4, pin_memory=True,
                 persistent_workers=True)
    train_loader = DataLoader(
        H5ForecastDataset(cfg["dataset_cache"], "train",
                          layer_source, layer_target),
        shuffle=True, **kw_h5)
    val_loader = DataLoader(
        H5ForecastDataset(cfg["dataset_cache"], "val",
                          layer_source, layer_target),
        shuffle=False, **kw_h5)

    forecaster = AttentionForecaster(
        embed_dim=1024, hidden=cfg["hidden"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(forecaster.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                        T_max=cfg["epochs"])

    best_val_kl = float('inf')
    best_val_rho = -1.0
    save_path = cfg["forecaster_dir"] / f"forecaster_{run_name}.pt"

    for epoch in range(cfg["epochs"]):
        # Train
        forecaster.train()
        train_kl, train_rho_list = 0., []

        for emb, target, _ in tqdm(train_loader, leave=False,
                                   desc=f"Ep{epoch+1} train"):
            emb, target = emb.to(device), target.to(device)
            pred = forecaster(emb)

            loss_kl = F.kl_div((pred + 1e-8).log(), target + 1e-8,
                               reduction='batchmean')
            opt.zero_grad()
            loss_kl.backward()
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
            opt.step()

            train_kl += loss_kl.item()
            with torch.no_grad():
                for b in range(len(emb)):
                    rho, _ = spearmanr(pred[b].cpu().numpy(),
                                       target[b].cpu().numpy())
                    train_rho_list.append(rho)

        # Val
        forecaster.eval()
        val_kl, val_rho_list = 0., []
        with torch.no_grad():
            for emb, target, _ in val_loader:
                emb, target = emb.to(device), target.to(device)
                pred = forecaster(emb)
                val_kl += F.kl_div((pred + 1e-8).log(), target + 1e-8,
                                   reduction='batchmean').item()
                for b in range(len(emb)):
                    rho, _ = spearmanr(pred[b].cpu().numpy(),
                                       target[b].cpu().numpy())
                    val_rho_list.append(rho)

        sched.step()

        train_kl /= len(train_loader)
        val_kl /= len(val_loader)
        train_rho = np.nanmean(train_rho_list)
        val_rho = np.nanmean(val_rho_list)

        wandb.log({
            "epoch": epoch + 1, "train/kl": train_kl, "val/kl": val_kl,
            "train/rho": train_rho, "val/rho": val_rho,
            "lr": sched.get_last_lr()[0],
        })

        if val_kl < best_val_kl:
            best_val_kl = val_kl
            best_val_rho = val_rho
            torch.save(forecaster.state_dict(), save_path)

        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:02d} | "
                  f"train_kl={train_kl:.4f} val_kl={val_kl:.4f} | "
                  f"train_rho={train_rho:.3f} val_rho={val_rho:.3f}")

    # Test with best checkpoint
    forecaster.load_state_dict(torch.load(save_path, map_location=device))
    forecaster.eval()

    test_rho_forecaster = []
    test_rho_token_norm = []

    with h5py.File(cfg["dataset_cache"], 'r') as f_h5:
        grp = f_h5["test"]
        emb_all = torch.from_numpy(
            grp[f"emb_layer{layer_source}"][:]).float()
        target_all = torch.from_numpy(
            grp[f"attn_layer{layer_target}"][:]).float()

    test_batch_ds = DataLoader(TensorDataset(emb_all, target_all),
                               batch_size=64, shuffle=False)

    with torch.no_grad():
        for emb, target in test_batch_ds:
            emb = emb.to(device)
            pred = forecaster(emb).cpu()
            for b in range(len(emb)):
                t = target[b].numpy()
                rho_f, _ = spearmanr(pred[b].numpy(), t)
                rho_n, _ = spearmanr(
                    emb[b].cpu().norm(dim=-1).numpy(), t)
                test_rho_forecaster.append(rho_f)
                test_rho_token_norm.append(rho_n)

    test_rho_f = np.nanmean(test_rho_forecaster)
    test_rho_n = np.nanmean(test_rho_token_norm)

    wandb.log({
        "test/rho_forecaster": test_rho_f,
        "test/rho_token_norm": test_rho_n,
        "test/delta_vs_norm": test_rho_f - test_rho_n,
        "best_val_rho": best_val_rho,
        "best_val_kl": best_val_kl,
    })

    print(f"\n  Test rho forecaster:  {test_rho_f:.3f}")
    print(f"  Test rho token norm: {test_rho_n:.3f}")
    print(f"  Delta vs baseline:   {test_rho_f - test_rho_n:+.3f}")

    wandb.finish()

    return {
        "layer_source": layer_source, "layer_target": layer_target,
        "best_val_rho": best_val_rho, "best_val_kl": best_val_kl,
        "test_rho_forecaster": test_rho_f,
        "test_rho_token_norm": test_rho_n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Train attention forecaster")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--layers-source", type=int, nargs="+", default=[2])
    parser.add_argument("--layer-target", type=int, default=23)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str,
                        default="attention-forecaster")
    parser.add_argument("--cache-dir", type=str, default="/data/data_cache")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    dataset_name = Path(args.data_dir).name
    base_ckpt = Path("checkpoints")

    cfg = dict(
        dataset_name=dataset_name,
        output_dir=base_ckpt / dataset_name / "uni_finetuned",
        dataset_cache=Path(args.cache_dir) / f"{dataset_name}_forecaster_dataset.h5",
        forecaster_dir=base_ckpt / dataset_name / "forecaster",
        layers_source=args.layers_source,
        layer_target=args.layer_target,
        hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, wandb_project=args.wandb_project,
    )

    cfg["dataset_cache"].parent.mkdir(parents=True, exist_ok=True)
    cfg["forecaster_dir"].mkdir(parents=True, exist_ok=True)

    # Extract features if cache doesn't exist
    if not cfg["dataset_cache"].exists():
        classifier_ckpt = args.classifier_ckpt or str(
            cfg["output_dir"] / "best_model.pt")

        train_loader, val_loader, test_loader, class_names, n_classes = \
            build_loaders(args.data_dir, args.img_size, args.batch_size,
                          args.num_workers, drop_last_train=False)

        model = UNILoRAClassifier(n_classes).to(device)
        ckpt = torch.load(classifier_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        collect_and_save_dataset(
            model,
            {"train": train_loader, "val": val_loader, "test": test_loader},
            device,
            layers_source=cfg["layers_source"],
            layer_target=cfg["layer_target"],
            save_path=cfg["dataset_cache"],
        )
    else:
        print(f"Dataset cache found: {cfg['dataset_cache']}")

    # Train forecasters
    all_results = []
    for layer_source in cfg["layers_source"]:
        result = train_forecaster(
            layer_source=layer_source,
            layer_target=cfg["layer_target"],
            cfg=cfg, device=device,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'Layer':>8} {'rho Forecaster':>14} {'rho Baseline':>12} "
          f"{'Delta':>8}")
    print("-" * 46)
    for r in all_results:
        delta = r["test_rho_forecaster"] - r["test_rho_token_norm"]
        print(f"{r['layer_source']:>8} {r['test_rho_forecaster']:>14.3f} "
              f"{r['test_rho_token_norm']:>12.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
