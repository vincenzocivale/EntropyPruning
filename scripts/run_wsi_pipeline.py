#!/usr/bin/env python
"""Orchestration script for the 3-phase WSI evaluation pipeline.

Usage:
  python scripts/run_wsi_pipeline.py \
    --encoder uni_v1 --dataset TCGA-BRCA --task subtype \
    --wsi-dir /path/to/wsis --output-dir ./results \
    --prune-layer 4 --keep-ratio 0.5
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run full 3-phase WSI evaluation pipeline"
    )
    # Core args
    parser.add_argument("--encoder", type=str, required=True, help="TRIDENT encoder (e.g., uni_v1)")
    parser.add_argument("--dataset", type=str, required=True, help="Patho-Bench dataset (e.g., TCGA-BRCA)")
    parser.add_argument("--task", type=str, required=True, help="Patho-Bench task (e.g., subtype)")
    parser.add_argument("--wsi-dir", type=str, required=True, help="WSI directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--prune-layer", type=int, required=True, help="Pruning layer")
    parser.add_argument("--keep-ratio", type=float, required=True, help="Keep ratio (0-1)")
    # Optional
    parser.add_argument("--mag", type=int, default=20, help="Magnification")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size")
    parser.add_argument("--tiles-per-wsi", type=int, default=64, help="Tiles per WSI")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs-phase1", type=int, default=30, help="Phase 1 epochs")
    parser.add_argument("--epochs-phase2", type=int, default=20, help="Phase 2 epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip Phase 2")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  WSI Evaluation Pipeline")
    print(f"  Encoder: {args.encoder}  Dataset: {args.dataset}  Task: {args.task}")
    print(f"  Prune layer: {args.prune_layer}  Keep ratio: {args.keep_ratio}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    script_dir = Path(__file__).parent

    # Phase 1
    if args.skip_phase1:
        print("Skipping Phase 1 (--skip-phase1)")
        forecaster_ckpt = None
    else:
        phase1_dir = output_dir / "phase1"
        forecaster_ckpt = phase1_dir / f"forecaster_{args.encoder}_src{args.prune_layer}_tgt{args.prune_layer+2}.pt"

        if forecaster_ckpt.exists():
            print(f"Phase 1 checkpoint exists: {forecaster_ckpt}\n")
        else:
            print("Running Phase 1: Train AttentionForecaster...\n")
            cmd = [
                sys.executable, str(script_dir / "wsi_train_forecaster.py"),
                "--encoder", args.encoder,
                "--wsi-dir", args.wsi_dir,
                "--prune-layer", str(args.prune_layer),
                "--mag", str(args.mag),
                "--patch-size", str(args.patch_size),
                "--tiles-per-wsi", str(args.tiles_per_wsi),
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs_phase1),
                "--output-dir", str(phase1_dir),
                "--seed", str(args.seed),
            ]
            if args.wandb_project:
                cmd += ["--wandb-project", args.wandb_project]
            if subprocess.run(cmd).returncode != 0:
                print("Phase 1 failed!")
                return 1

    # Phase 2
    if args.skip_phase2:
        print("Skipping Phase 2 (--skip-phase2)")
        pruned_ckpt = None
    else:
        if forecaster_ckpt is None:
            raise ValueError("Phase 2 requires Phase 1 forecaster. Use --skip-phase1=False or provide --forecaster-ckpt")

        phase2_dir = output_dir / "phase2"
        pruned_ckpt = phase2_dir / \
            f"best_{args.encoder}_prune{args.prune_layer}_keep{int(args.keep_ratio*100)}.pt"

        if pruned_ckpt.exists():
            print(f"Phase 2 checkpoint exists: {pruned_ckpt}\n")
        else:
            print("Running Phase 2: Distillation fine-tuning...\n")
            cmd = [
                sys.executable, str(script_dir / "wsi_distill_pruned.py"),
                "--encoder", args.encoder,
                "--wsi-dir", args.wsi_dir,
                "--forecaster-ckpt", str(forecaster_ckpt),
                "--prune-layer", str(args.prune_layer),
                "--keep-ratio", str(args.keep_ratio),
                "--mag", str(args.mag),
                "--patch-size", str(args.patch_size),
                "--tiles-per-wsi", str(args.tiles_per_wsi),
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs_phase2),
                "--output-dir", str(phase2_dir),
                "--seed", str(args.seed),
            ]
            if args.wandb_project:
                cmd += ["--wandb-project", args.wandb_project]
            if subprocess.run(cmd).returncode != 0:
                print("Phase 2 failed!")
                return 1

    # Phase 3
    print("Running Phase 3: In-memory evaluation...\n")
    phase3_dir = output_dir / "phase3"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    if forecaster_ckpt is None or pruned_ckpt is None:
        raise ValueError("Phase 3 requires Phase 1 and 2 checkpoints")

    cmd = [
        sys.executable, str(script_dir / "wsi_evaluate.py"),
        "--encoder", args.encoder,
        "--dataset", args.dataset,
        "--task", args.task,
        "--wsi-dir", args.wsi_dir,
        "--prune-layer", str(args.prune_layer),
        "--keep-ratio", str(args.keep_ratio),
        "--forecaster-ckpt", str(forecaster_ckpt),
        "--pruned-ckpt", str(pruned_ckpt),
        "--mag", str(args.mag),
        "--patch-size", str(args.patch_size),
        "--batch-size", str(args.batch_size),
        "--output-dir", str(phase3_dir),
        "--seed", str(args.seed),
    ]

    if subprocess.run(cmd).returncode != 0:
        print("Phase 3 failed!")
        return 1

    print(f"\n{'='*70}")
    print(f"  Pipeline complete!")
    print(f"  Results: {output_dir}")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
