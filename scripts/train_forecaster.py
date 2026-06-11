"""Phase 2: Train AttentionForecaster to predict target-layer attention from source-layer embeddings.

The teacher is the final-block CLS->patch attention of the backbone with Phase 1's
full-network LoRA adapters applied (scripts/train_phase1_lora.py): the LoRA-adapted
backbone is frozen and used only to extract features/attention. If no Phase-1 checkpoint
is found, falls back to the frozen pretrained backbone's final attention. The
distillation objective is the KL divergence between the forecaster's softmax and the
L1-normalized teacher attention; see src/losses.py.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.utils import set_seed, get_device, save_results
from src.models import (AttentionForecaster, ThunderBackboneAdapter, FrozenBackbone,
                        load_lora_adapted_backbone)
from src.losses import distillation_loss
from src.data.thunder_loaders import build_thunder_loaders
from src.data.h5_dataset import H5ForecastDataset
from src.collection import collect_and_save_dataset
from src.evaluation.ranking_metrics import spearman_rho, topk_overlap

# Top-k overlap is reported at these deploy keep-ratios.
OVERLAP_KEEP_RATIOS = (0.1, 0.2, 0.3)


def train_forecaster(layer_source, layer_target, cfg, device):
    run_name = (f"{cfg['model_name']}_{cfg['dataset_name']}_phase2"
                f"_src{layer_source:02d}_tgt{layer_target:02d}")
    print(f"\n{'='*60}\n  Experiment: {run_name}\n{'='*60}")

    wandb.init(
        project=cfg["wandb_project"],
        name=run_name,
        job_type="phase2",
        group=f"{cfg['dataset_name']}/{cfg['model_name']}",
        config={**{k: cfg[k] for k in ("model_name", "embed_dim", "hidden",
                                        "n_heads", "n_layers", "dropout",
                                        "epochs", "lr", "weight_decay",
                                        "ckpt_metric", "use_adapted_teacher")},
                "layer_source": layer_source, "layer_target": layer_target},
        tags=[cfg["model_name"], cfg["dataset_name"],
              f"src{layer_source}", f"tgt{layer_target}", "phase2",
              "adapted_teacher" if cfg["use_adapted_teacher"] else "pretrained_teacher"],
        reinit=True,
    )

    kw = dict(batch_size=128, num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(
        H5ForecastDataset(cfg["dataset_cache"], "train", layer_source, layer_target),
        shuffle=True, **kw)
    val_loader = DataLoader(
        H5ForecastDataset(cfg["dataset_cache"], "val", layer_source, layer_target),
        shuffle=False, **kw)

    forecaster = AttentionForecaster(
        embed_dim=cfg["embed_dim"],
        hidden=cfg["hidden"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(forecaster.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda")

    ckpt_metric = cfg["ckpt_metric"]  # "kl" (lower better) or "rho" (higher better)
    best_val_kl, best_val_rho = float('inf'), -1.0
    best_sel = float('inf') if ckpt_metric == "kl" else -1.0
    save_path = cfg["forecaster_dir"] / f"forecaster_{run_name}.pt"

    def better(cur, best):
        return cur < best if ckpt_metric == "kl" else cur > best

    for epoch in range(cfg["epochs"]):
        forecaster.train()
        train_loss = 0.
        for emb, target, _ in tqdm(train_loader, leave=False, desc=f"Ep{epoch+1} train"):
            emb, target = emb.to(device), target.to(device)
            with torch.amp.autocast("cuda"):
                logits = forecaster(emb)
            loss = distillation_loss(logits, target)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()

        forecaster.eval()
        val_kl, val_rho_list = 0., []
        with torch.no_grad():
            for emb, target, _ in val_loader:
                emb, target = emb.to(device), target.to(device)
                logits = forecaster(emb)
                val_kl += distillation_loss(logits, target).item()
                val_rho_list.append(spearman_rho(logits, target))
        sched.step()

        train_loss /= len(train_loader)
        val_kl /= len(val_loader)
        val_rho = torch.cat(val_rho_list).mean().item()
        wandb.log({"epoch": epoch+1, "train/loss": train_loss,
                   "val/kl": val_kl, "val/rho": val_rho, "lr": sched.get_last_lr()[0]})

        sel = val_kl if ckpt_metric == "kl" else val_rho
        if better(sel, best_sel):
            best_sel, best_val_kl, best_val_rho = sel, val_kl, val_rho
            torch.save(forecaster.state_dict(), save_path)
        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:02d} | train={train_loss:.4f} val_kl={val_kl:.4f} "
                  f"val_rho={val_rho:.3f}")

    # Test
    forecaster.load_state_dict(torch.load(save_path, map_location=device))
    forecaster.eval()
    test_loader = DataLoader(
        H5ForecastDataset(cfg["dataset_cache"], "test", layer_source, layer_target),
        shuffle=False, **kw)
    rho_f, rho_n = [], []
    overlap_f = {k: [] for k in OVERLAP_KEEP_RATIOS}
    with torch.no_grad():
        for emb, target, _ in test_loader:
            emb, target = emb.to(device), target.to(device)
            logits = forecaster(emb)
            rho_f.append(spearman_rho(logits, target))
            rho_n.append(spearman_rho(emb.norm(dim=-1), target))
            for k in OVERLAP_KEEP_RATIOS:
                overlap_f[k].append(topk_overlap(logits, target, k))
    test_rho_f = torch.cat(rho_f).mean().item()
    test_rho_n = torch.cat(rho_n).mean().item()
    test_overlap = {k: torch.cat(v).mean().item() for k, v in overlap_f.items()}
    results = {
        "model_name": cfg["model_name"],
        "dataset_name": cfg["dataset_name"],
        "layer_source": layer_source,
        "layer_target": layer_target,
        "embed_dim": cfg["embed_dim"],
        "use_adapted_teacher": cfg["use_adapted_teacher"],
        "ckpt_metric": ckpt_metric,
        "best_val_rho": round(best_val_rho, 6),
        "best_val_kl": round(best_val_kl, 6),
        "test_rho_forecaster": round(test_rho_f, 6),
        "test_rho_token_norm": round(test_rho_n, 6),
        "test_delta_vs_norm": round(test_rho_f - test_rho_n, 6),
        **{f"test_overlap@{k}": round(v, 6) for k, v in test_overlap.items()},
    }
    wandb.log({
        "test/rho_forecaster": test_rho_f, "test/rho_token_norm": test_rho_n,
        "test/delta_vs_norm": test_rho_f - test_rho_n,
        "val/best_rho": best_val_rho, "val/best_kl": best_val_kl,
        **{f"test/overlap@{k}": v for k, v in test_overlap.items()},
    })
    print(f"\n  Test rho forecaster: {test_rho_f:.3f}")
    print(f"  Test rho token norm: {test_rho_n:.3f}")
    print(f"  Delta vs baseline:   {test_rho_f - test_rho_n:+.3f}")
    print("  Test overlap@k:      " +
          ", ".join(f"{k}={v:.3f}" for k, v in test_overlap.items()))
    wandb.finish()

    save_results(save_path.parent / f"results_{run_name}.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train attention forecaster")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--layers-source", type=int, nargs="+", default=[2])
    parser.add_argument("--layer-target", type=int, default=None,
                        help="Defaults to last block (n_blocks-1).")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="attention-forecaster")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--forecaster-dir", type=str, default=None,
                        help="Override forecaster checkpoint output dir.")
    parser.add_argument("--phase1-ckpt", type=str, default=None,
                        help="Phase-1 full-backbone LoRA checkpoint (see "
                             "scripts/train_phase1_lora.py). Defaults to "
                             "checkpoints/<dataset>/<model>_lora_phase1/"
                             "best_<model>_<dataset>_phase1.pt if present; falls back to the "
                             "frozen pretrained backbone otherwise.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--ckpt-metric", type=str, default="kl", choices=["kl", "rho"],
                        help="Validation metric for best-checkpoint selection.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Model: {args.model_name} | Dataset: {args.dataset_name}")

    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    print(f"embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}  "
          f"n_patches={adapter.n_patches}  prefix={adapter.num_prefix_tokens}")

    layer_target = args.layer_target if args.layer_target is not None \
        else adapter.n_blocks - 1
    for ls in args.layers_source:
        assert ls < adapter.n_blocks, f"--layers-source {ls} >= n_blocks {adapter.n_blocks}"
    assert layer_target < adapter.n_blocks, \
        f"--layer-target {layer_target} >= n_blocks {adapter.n_blocks}"

    base_ckpt = Path("checkpoints")

    default_phase1_ckpt = (base_ckpt / args.dataset_name / f"{args.model_name}_lora_phase1" /
                            f"best_{args.model_name}_{args.dataset_name}_phase1.pt")
    phase1_ckpt = Path(args.phase1_ckpt) if args.phase1_ckpt else default_phase1_ckpt
    use_adapted = phase1_ckpt.exists()
    if use_adapted:
        teacher_backbone = load_lora_adapted_backbone(
            raw_backbone, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            ckpt_path=phase1_ckpt, map_location=device)
        print(f"Loaded Phase-1 LoRA-adapted backbone: {phase1_ckpt}")
    else:
        teacher_backbone = raw_backbone
        print(f"No Phase-1 LoRA checkpoint found at {phase1_ckpt} -- "
              "using frozen pretrained backbone as teacher.")

    cache_dir = Path(args.cache_dir) if args.cache_dir else base_ckpt / args.dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_lora_adapted" if use_adapted else ""

    cfg = dict(
        model_name=args.model_name, dataset_name=args.dataset_name,
        embed_dim=adapter.embed_dim,
        dataset_cache=cache_dir / f"{args.dataset_name}_{args.model_name}_features{cache_suffix}.h5",
        forecaster_dir=Path(args.forecaster_dir) if args.forecaster_dir else
            base_ckpt / args.dataset_name / f"{args.model_name}_forecaster",
        layers_source=args.layers_source, layer_target=layer_target,
        hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, wandb_project=args.wandb_project,
        ckpt_metric=args.ckpt_metric, use_adapted_teacher=use_adapted,
    )
    cfg["forecaster_dir"].mkdir(parents=True, exist_ok=True)

    if not cfg["dataset_cache"].exists():
        # Teacher = backbone with Phase-1 LoRA adapters applied (or frozen pretrained
        # backbone if no Phase-1 checkpoint was found above).
        train_loader, val_loader, test_loader, _, _ = \
            build_thunder_loaders(args.dataset_name, args.base_data_folder, transform,
                                  args.batch_size, args.num_workers, drop_last_train=False)
        model = FrozenBackbone(teacher_backbone, adapter).to(device).eval()
        collect_and_save_dataset(
            model,
            {"train": train_loader, "val": val_loader, "test": test_loader},
            device,
            layers_source=cfg["layers_source"],
            layers_target=[cfg["layer_target"]],
            save_path=cfg["dataset_cache"],
        )
    else:
        print(f"Feature cache found: {cfg['dataset_cache']}")

    all_results = []
    for ls in cfg["layers_source"]:
        all_results.append(train_forecaster(ls, cfg["layer_target"], cfg, device))

    print(f"\n{'Layer':>8} {'rho Forecaster':>14} {'rho Baseline':>12} {'Delta':>8}")
    print("-" * 46)
    for r in all_results:
        delta = r["test_rho_forecaster"] - r["test_rho_token_norm"]
        print(f"{r['layer_source']:>8} {r['test_rho_forecaster']:>14.3f} "
              f"{r['test_rho_token_norm']:>12.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
