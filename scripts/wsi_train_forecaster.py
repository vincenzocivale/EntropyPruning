"""Phase 1 (new): Train AttentionForecaster on WSI tiles to predict CLS-attention from early embeddings.

Self-supervised: no labels needed. Tiles streamed on-the-fly; coordinates kept in RAM.
Backbone fully frozen. In-loop attention hooks capture source embeddings and target attention maps.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trident.patch_encoder_models import encoder_factory

from src.utils import set_seed, get_device, save_results, EarlyStopping
from src.models import AttentionForecaster
from src.models.backbone_adapter import BackboneAdapter
from src.data.wsi_tile_dataset import WSITileDataset


def spearman_correlation(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Vectorized Spearman rank correlation in PyTorch."""
    r_pred = y_pred.argsort(dim=-1).argsort(dim=-1).float()
    r_true = y_true.argsort(dim=-1).argsort(dim=-1).float()
    r_pred_m = r_pred - r_pred.mean(dim=-1, keepdim=True)
    r_true_m = r_true - r_true.mean(dim=-1, keepdim=True)
    num = (r_pred_m * r_true_m).sum(dim=-1)
    den = torch.sqrt((r_pred_m**2).sum(dim=-1) * (r_true_m**2).sum(dim=-1))
    return num / (den + 1e-8)


def register_hooks(model, backbone, src_layer: int, tgt_layer: int, device):
    """Register attention hooks to capture embeddings and attention maps.

    Returns:
        (hooks_list, cache_dict) where cache_dict will be populated by hooks
    """
    cache = {}
    hooks = []

    def make_src_hook(layer_idx):
        def hook(m, x, y):
            # x: input to block, shape (B, N, D)
            # After norm but before attention
            x_in = x[0] if isinstance(x, tuple) else x
            # Capture spatial patches only (exclude prefix tokens)
            num_prefix = backbone.num_prefix_tokens
            cache[f"emb_{layer_idx}"] = x_in[:, num_prefix:].detach().cpu()
        return hook

    def make_tgt_hook(layer_idx):
        def hook(m, x, y):
            # m: Attention module
            # Replicate attention computation to capture CLS-to-patch scores
            x_in = x[0] if isinstance(x, tuple) else x
            B, N, C = x_in.shape
            num_prefix = backbone.num_prefix_tokens

            # Manual attention computation (same as extract_features.py)
            qkv = m.qkv(x_in).reshape(B, N, 3, m.num_heads, m.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = m.q_norm(q), m.k_norm(k)

            attn = (q @ k.transpose(-2, -1) * m.scale)
            attn = attn.softmax(-1)

            # Extract CLS-to-patch attention (mean over heads)
            cache[f"attn_{layer_idx}"] = (
                attn[:, :, 0, num_prefix:].mean(1).detach().cpu()
            )
        return hook

    # Register source layer hook (before attention, to capture embeddings)
    src_block = backbone.get_blocks()[src_layer]
    h_src = src_block.norm1.register_forward_hook(make_src_hook(src_layer))
    hooks.append(h_src)

    # Register target layer hook (at attention, to capture scores)
    tgt_block = backbone.get_blocks()[tgt_layer]
    attn_module = tgt_block.attn
    h_tgt = attn_module.register_forward_hook(make_tgt_hook(tgt_layer))
    hooks.append(h_tgt)

    return hooks, cache


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 (new): Train AttentionForecaster on WSI tiles (self-supervised)"
    )
    # Model & data
    parser.add_argument("--encoder", type=str, required=True,
                        help="TRIDENT encoder name (e.g. uni_v1, virchow, hoptimus0)")
    parser.add_argument("--wsi-dir", type=str, required=True,
                        help="Directory containing WSI files (*.svs, *.ndpi, etc.)")
    parser.add_argument("--prune-layer", type=int, required=True,
                        help="Source layer L for AttentionForecaster input")
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Target layer T for attention supervision (default: n_blocks - 1)")
    # Dataset
    parser.add_argument("--mag", type=int, default=20,
                        help="Magnification for tile extraction (default: 20)")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size in pixels (default: 256)")
    parser.add_argument("--tiles-per-wsi", type=int, default=64,
                        help="Tiles sampled per WSI per epoch (default: 64)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of WSIs for validation (default: 0.1)")
    # Forecaster hyperparameters
    parser.add_argument("--hidden", type=int, default=256,
                        help="AttentionForecaster hidden dim (default: 256)")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of transformer layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout (default: 0.2)")
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay (default: 0.0)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10, 0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to save forecaster checkpoint and results")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project (default: None = skip W&B)")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Phase 1 (new): Train AttentionForecaster on WSI tiles")
    print(f"  Encoder: {args.encoder}")
    print(f"  WSI dir: {args.wsi_dir}")
    print(f"  Source layer: {args.prune_layer} → Target layer: {args.target_layer or 'n_blocks-1'}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    # Load encoder
    enc = encoder_factory(args.encoder)
    backbone = enc.model
    transform = enc.eval_transforms
    adapter = BackboneAdapter(backbone)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    target_layer = args.target_layer if args.target_layer is not None else adapter.n_blocks - 1
    assert args.prune_layer < adapter.n_blocks, f"Source layer {args.prune_layer} >= n_blocks {adapter.n_blocks}"
    assert target_layer < adapter.n_blocks, f"Target layer {target_layer} >= n_blocks {adapter.n_blocks}"

    print(f"Backbone: {args.encoder}")
    print(f"  embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}")
    print(f"  n_patches={adapter.n_patches}  prefix={adapter.num_prefix_tokens}\n")

    # Load WSIs
    wsi_dir = Path(args.wsi_dir)
    wsi_paths = list(wsi_dir.glob("**/*.svs")) + list(wsi_dir.glob("**/*.ndpi")) + \
                list(wsi_dir.glob("**/*.tif")) + list(wsi_dir.glob("**/*.tiff"))
    wsi_paths = sorted([str(p) for p in wsi_paths])

    if not wsi_paths:
        raise ValueError(f"No WSI files found in {wsi_dir}")

    print(f"Found {len(wsi_paths)} WSI files")

    # Train/val split
    np.random.seed(args.seed)
    split_idx = int(len(wsi_paths) * (1 - args.val_split))
    train_paths = wsi_paths[:split_idx]
    val_paths = wsi_paths[split_idx:]

    print(f"  Train: {len(train_paths)} WSIs")
    print(f"  Val: {len(val_paths)} WSIs\n")

    # Datasets
    print("Building train dataset (Otsu segmentation + indexing)...")
    train_dataset = WSITileDataset(
        train_paths, transform=transform, mag=args.mag, patch_size=args.patch_size,
        tiles_per_wsi=args.tiles_per_wsi, seed=args.seed, verbose=args.verbose
    )
    print("Building val dataset...")
    val_dataset = WSITileDataset(
        val_paths, transform=transform, mag=args.mag, patch_size=args.patch_size,
        tiles_per_wsi=args.tiles_per_wsi, seed=args.seed, verbose=args.verbose
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)

    # Forecaster
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"AttentionForecaster:")
    print(f"  Input: layer {args.prune_layer} embeddings ({adapter.embed_dim}-dim)")
    print(f"  Output: attention scores for layer {target_layer}")
    print(f"  Hidden: {args.hidden}  Heads: {args.n_heads}  Layers: {args.n_layers}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(forecaster.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # W&B
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"phase1_{args.encoder}_src{args.prune_layer}_tgt{target_layer}",
            config=vars(args),
            tags=["phase1", "wsi", "forecaster"]
        )

    # Register hooks once before training (not per-batch)
    print("\nRegistering attention hooks...")
    hooks, cache = register_hooks(backbone, adapter, args.prune_layer, target_layer, device)

    # Training loop
    best_val_loss = float("inf")
    best_val_rho = -1.0
    early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-5) if args.patience > 0 else None

    for epoch in range(args.epochs):
        # Train epoch
        forecaster.train()
        train_loss = 0.0
        train_count = 0

        for tiles in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]",
                         leave=False):
            tiles = tiles.to(device)

            # Backbone forward with hooks already registered
            with torch.no_grad():
                _ = backbone(tiles)

            src_emb = cache.get(f"emb_{args.prune_layer}")
            tgt_attn = cache.get(f"attn_{target_layer}")

            if src_emb is None or tgt_attn is None:
                print(f"Warning: hooks didn't capture data. Skipping batch.")
                continue

            src_emb = src_emb.to(device)  # (B, N, D)
            tgt_attn = tgt_attn.to(device)  # (B, N)

            # Forward
            pred_scores = forecaster(src_emb)  # (B, N)

            # KL divergence loss
            loss = F.kl_div(
                pred_scores.log_softmax(dim=-1),
                tgt_attn.softmax(dim=-1),
                reduction="batchmean"
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * len(tiles)
            train_count += len(tiles)

            # Clear cache for next batch
            cache.clear()

        train_loss /= train_count

        # Validation epoch
        forecaster.eval()
        val_loss = 0.0
        val_rhos = []
        val_count = 0

        with torch.no_grad():
            for tiles in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]",
                             leave=False):
                tiles = tiles.to(device)

                _ = backbone(tiles)

                src_emb = cache.get(f"emb_{args.prune_layer}")
                tgt_attn = cache.get(f"attn_{target_layer}")

                if src_emb is None or tgt_attn is None:
                    continue

                src_emb = src_emb.to(device)
                tgt_attn = tgt_attn.to(device)

                pred_scores = forecaster(src_emb)
                loss = F.kl_div(
                    pred_scores.log_softmax(dim=-1),
                    tgt_attn.softmax(dim=-1),
                    reduction="batchmean"
                )

                val_loss += loss.item() * len(tiles)

                # Spearman correlation
                rho = spearman_correlation(pred_scores, tgt_attn).mean().item()
                val_rhos.append(rho)

                val_count += len(tiles)

                # Clear cache for next batch
                cache.clear()

        val_loss /= max(val_count, 1)
        val_rho = np.mean(val_rhos) if val_rhos else 0.0

        scheduler.step()

        print(f"Epoch {epoch+1:3d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f}  val_rho={val_rho:.4f}")

        if args.wandb_project:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/rho": val_rho,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rho = val_rho
            ckpt_path = output_dir / f"forecaster_{args.encoder}_src{args.prune_layer}_tgt{target_layer}.pt"
            torch.save(forecaster.state_dict(), ckpt_path)
            print(f"  → Saved best checkpoint: {ckpt_path}")

        # Early stopping
        if early_stopper is not None:
            if early_stopper.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Clean up hooks
    for h in hooks:
        h.remove()

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Best val_rho: {best_val_rho:.4f}")
    print(f"  Checkpoint: {output_dir / f'forecaster_{args.encoder}_src{args.prune_layer}_tgt{target_layer}.pt'}")
    print(f"{'='*70}\n")

    if args.wandb_project:
        wandb.finish()

    # Save results JSON
    results = {
        "encoder": args.encoder,
        "source_layer": args.prune_layer,
        "target_layer": target_layer,
        "best_val_loss": best_val_loss,
        "best_val_rho": best_val_rho,
        "train_wsis": len(train_paths),
        "val_wsis": len(val_paths),
    }
    save_results(output_dir / "results.json", results)


if __name__ == "__main__":
    main()
