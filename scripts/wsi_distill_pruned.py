"""Phase 2 (new): Distillation fine-tune of pruned encoder on WSI tiles.

Teacher: frozen non-pruned encoder (CLS embedding)
Student: pruned encoder (GenericLoRAWithForecasterPruning) with frozen forecaster
Loss: 1 - cosine_similarity between CLS embeddings
No labels needed; tiles streamed on-the-fly.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trident.patch_encoder_models import encoder_factory

from src.utils import set_seed, get_device, save_results
from src.models import (AttentionForecaster, GenericLoRAWithForecasterPruning)
from src.models.backbone_adapter import BackboneAdapter
from src.data.wsi_tile_dataset import WSITileDataset


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Cosine distance loss (1 - cosine_similarity)."""
    return 1.0 - F.cosine_similarity(x1, x2, dim=-1).mean()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 (new): Distillation fine-tune of pruned encoder on WSI tiles"
    )
    # Model & data
    parser.add_argument("--encoder", type=str, required=True,
                        help="TRIDENT encoder name (e.g. uni_v1, virchow)")
    parser.add_argument("--wsi-dir", type=str, required=True,
                        help="Directory containing WSI files for training")
    parser.add_argument("--forecaster-ckpt", type=str, required=True,
                        help="Path to Phase 1 forecaster checkpoint")
    parser.add_argument("--prune-layer", type=int, required=True,
                        help="Layer at which to apply pruning")
    parser.add_argument("--keep-ratio", type=float, required=True,
                        help="Fraction of tokens to keep (e.g., 0.5)")
    # Dataset
    parser.add_argument("--mag", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--tiles-per-wsi", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Phase 2 (new): Distillation fine-tune of pruned encoder")
    print(f"  Encoder: {args.encoder}")
    print(f"  Prune layer: {args.prune_layer}  Keep ratio: {args.keep_ratio}")
    print(f"  WSI dir: {args.wsi_dir}")
    print(f"{'='*70}\n")

    # Load encoder and setup
    enc = encoder_factory(args.encoder)
    backbone_teacher = enc.model.to(device).eval()
    for p in backbone_teacher.parameters():
        p.requires_grad_(False)

    backbone_student = enc.model.to(device)
    transform = enc.eval_transforms
    adapter = BackboneAdapter(backbone_student)

    print(f"Backbone: {args.encoder}")
    print(f"  embed_dim={adapter.embed_dim}  n_blocks={adapter.n_blocks}")
    print(f"  n_patches={adapter.n_patches}\n")

    # Load forecaster (frozen for Phase 2)
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=256, n_heads=4, n_layers=2, dropout=0.2,
    ).to(device)
    forecaster.load_state_dict(torch.load(args.forecaster_ckpt, map_location=device))
    forecaster.eval()
    for p in forecaster.parameters():
        p.requires_grad_(False)

    # Build pruned student model (n_classes doesn't matter for distillation)
    student_model = GenericLoRAWithForecasterPruning(
        backbone=backbone_student,
        adapter=adapter,
        n_classes=2,  # dummy; not used in distillation
        forecaster=forecaster,
        prune_layer=args.prune_layer,
        keep_ratio=args.keep_ratio,
    ).to(device)

    print(f"Student: GenericLoRAWithForecasterPruning")
    print(f"  Prune layer: {args.prune_layer}  Keep ratio: {args.keep_ratio}\n")

    # Load WSIs
    wsi_dir = Path(args.wsi_dir)
    wsi_paths = list(wsi_dir.glob("**/*.svs")) + list(wsi_dir.glob("**/*.ndpi")) + \
                list(wsi_dir.glob("**/*.tif")) + list(wsi_dir.glob("**/*.tiff"))
    wsi_paths = sorted([str(p) for p in wsi_paths])

    if not wsi_paths:
        raise ValueError(f"No WSI files found in {wsi_dir}")

    # Train/val split
    np.random.seed(args.seed)
    split_idx = int(len(wsi_paths) * (1 - args.val_split))
    train_paths = wsi_paths[:split_idx]
    val_paths = wsi_paths[split_idx:]

    print(f"Found {len(wsi_paths)} WSI files")
    print(f"  Train: {len(train_paths)}  Val: {len(val_paths)}\n")

    # Datasets
    print("Building datasets...")
    train_dataset = WSITileDataset(
        train_paths, transform=transform, mag=args.mag, patch_size=args.patch_size,
        tiles_per_wsi=args.tiles_per_wsi, seed=args.seed, verbose=args.verbose
    )
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

    # Optimizer (only student LoRA params)
    backbone_params = [p for _, p in student_model.backbone.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(backbone_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"phase2_{args.encoder}_prune{args.prune_layer}_keep{int(args.keep_ratio*100)}",
            config=vars(args),
            tags=["phase2", "wsi", "distillation"]
        )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Train
        student_model.train()
        backbone_teacher.eval()
        train_loss = 0.0
        train_count = 0

        for tiles in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False):
            tiles = tiles.to(device)

            with torch.no_grad():
                # Teacher CLS embedding (no pruning, no head)
                teacher_x = tiles
                for block in backbone_teacher.blocks:
                    teacher_x = block(teacher_x)
                teacher_x = backbone_teacher.norm(teacher_x)
                teacher_cls = teacher_x[:, 0, :]  # (B, D) CLS token

            # Student forward (with pruning)
            student_x = tiles
            for block in student_model.backbone.model.blocks:
                student_x = block(student_x)
            student_x = student_model.backbone.model.norm(student_x)
            student_cls = student_x[:, 0, :]  # (B, D) CLS token

            # Cosine distance loss
            loss = cosine_distance(student_cls, teacher_cls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * len(tiles)
            train_count += len(tiles)

        train_loss /= train_count

        # Validation
        student_model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for tiles in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False):
                tiles = tiles.to(device)

                # Teacher
                teacher_x = tiles
                for block in backbone_teacher.blocks:
                    teacher_x = block(teacher_x)
                teacher_x = backbone_teacher.norm(teacher_x)
                teacher_cls = teacher_x[:, 0, :]

                # Student
                student_x = tiles
                for block in student_model.backbone.model.blocks:
                    student_x = block(student_x)
                student_x = student_model.backbone.model.norm(student_x)
                student_cls = student_x[:, 0, :]

                loss = cosine_distance(student_cls, teacher_cls)
                val_loss += loss.item() * len(tiles)
                val_count += len(tiles)

        val_loss /= max(val_count, 1)
        scheduler.step()

        print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if args.wandb_project:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / \
                f"best_{args.encoder}_prune{args.prune_layer}_keep{int(args.keep_ratio*100)}.pt"
            torch.save(student_model.state_dict(), ckpt_path)
            print(f"  → Saved: {ckpt_path}")

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    if args.wandb_project:
        wandb.finish()

    results = {
        "encoder": args.encoder,
        "prune_layer": args.prune_layer,
        "keep_ratio": args.keep_ratio,
        "best_val_loss": best_val_loss,
    }
    save_results(results, output_dir / "results_phase2.json")


if __name__ == "__main__":
    main()
