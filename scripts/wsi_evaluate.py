"""Phase 3 (new): In-memory WSI-level classification evaluation.

Compares pruned vs non-pruned encoder on a Patho-Bench task.
Extracts patch embeddings, mean-pools to slide level, runs logistic regression.
Zero HDF5 files; fully in-memory.
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trident import load_wsi
from trident.patch_encoder_models import encoder_factory
from trident.wsi_objects.WSIPatcher import WSIPatcher
from trident.segmentation_models import segmentation_model_factory

from src.models import AttentionForecaster, GenericLoRAWithForecasterPruning
from src.models.backbone_adapter import BackboneAdapter
from src.utils import set_seed, get_device


def extract_slide_embeddings(
    wsi_path: str,
    encoder: torch.nn.Module,
    mag: int,
    patch_size: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Extract mean-pooled slide embedding from a WSI.

    Args:
        wsi_path: Path to WSI file
        encoder: loaded encoder (backbone or pruned model)
        mag: magnification
        patch_size: patch size
        batch_size: batch size for inference
        device: cuda/cpu

    Returns:
        Slide embedding tensor (D,)
    """
    try:
        wsi = load_wsi(wsi_path)
        otsu = segmentation_model_factory("otsu")
        mask_gdf = wsi.segment_tissue(otsu, target_mag=1.25, job_dir=None)

        patcher = WSIPatcher(
            wsi, patch_size=patch_size, dst_mag=mag,
            mask=mask_gdf, pil=True,
        )

        # Extract embeddings for all patches
        embeddings = []
        patch_count = len(patcher)

        if patch_count == 0:
            return torch.ones(1024, device=device) * -1.0  # dummy invalid embedding

        # Batch processing
        for i in range(0, patch_count, batch_size):
            batch_idx = list(range(i, min(i + batch_size, patch_count)))
            tiles = []
            for idx in batch_idx:
                tile, _, _ = patcher[idx]
                # Transform happens inside the encoder typically, but let's handle both
                if hasattr(encoder, 'eval_transforms'):
                    tile_tensor = encoder.eval_transforms(tile).unsqueeze(0)
                else:
                    tile_tensor = torch.from_numpy(np.array(tile)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                tiles.append(tile_tensor)

            tiles_batch = torch.cat(tiles, dim=0).to(device)

            with torch.no_grad():
                if isinstance(encoder, GenericLoRAWithForecasterPruning):
                    # Pruned model: extract CLS before head
                    x = tiles_batch
                    for block in encoder.backbone.model.blocks:
                        x = block(x)
                    x = encoder.backbone.model.norm(x)
                    batch_emb = x[:, 0, :]  # CLS token
                else:
                    # Raw backbone encoder
                    x = tiles_batch
                    for block in encoder.blocks:
                        x = block(x)
                    x = encoder.norm(x)
                    batch_emb = x[:, 0, :]  # CLS token

                embeddings.append(batch_emb.cpu())

        all_emb = torch.cat(embeddings, dim=0)  # (N_patches, D)

        # Mean pool
        return all_emb.mean(dim=0)

    except Exception as e:
        print(f"Error processing {wsi_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 (new): In-memory WSI-level evaluation with Patho-Bench"
    )
    # Model
    parser.add_argument("--encoder", type=str, required=True,
                        help="TRIDENT encoder name")
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="Patho-Bench dataset (e.g., TCGA-BRCA)")
    parser.add_argument("--task", type=str, required=True,
                        help="Patho-Bench task (e.g., subtype)")
    parser.add_argument("--wsi-dir", type=str, required=True,
                        help="Directory containing WSI files")
    parser.add_argument("--splits-dir", type=str, default="./patho_bench_splits",
                        help="Where to download Patho-Bench splits")
    # Checkpoints
    parser.add_argument("--prune-layer", type=int, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--forecaster-ckpt", type=str, required=True)
    parser.add_argument("--pruned-ckpt", type=str, required=True)
    # Inference
    parser.add_argument("--mag", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Phase 3: In-memory WSI classification comparison")
    print(f"  Dataset: {args.dataset} Task: {args.task}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Pruned: layer={args.prune_layer} keep_ratio={args.keep_ratio}")
    print(f"{'='*70}\n")

    # Download Patho-Bench split
    print("Downloading Patho-Bench split...")
    try:
        from patho_bench.SplitFactory import SplitFactory
        splits_dir = Path(args.splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)
        split_path, task_config_path = SplitFactory.from_hf(
            saveto=str(splits_dir),
            source=args.dataset,
            task=args.task
        )
    except Exception as e:
        print(f"Error downloading split: {e}")
        print("Trying to load from local path...")
        split_path = Path(args.splits_dir) / f"{args.dataset}_{args.task}.tsv"
        task_config_path = Path(args.splits_dir) / f"{args.dataset}_{args.task}_config.yaml"

    # Load split
    split_df = pd.read_csv(split_path, sep='\t')
    print(f"Loaded split with {len(split_df)} samples")
    print(f"Columns: {split_df.columns.tolist()}\n")

    # Map slide_id to WSI path
    wsi_dir = Path(args.wsi_dir)
    wsi_files = list(wsi_dir.glob("*.svs")) + list(wsi_dir.glob("*.ndpi")) + \
                list(wsi_dir.glob("*.tif")) + list(wsi_dir.glob("*.tiff"))
    slide_id_to_path = {p.stem: str(p) for p in wsi_files}

    print(f"Found {len(wsi_files)} WSI files in {wsi_dir}\n")

    # Load models
    print("Loading encoders...")
    enc = encoder_factory(args.encoder)
    backbone = enc.model.to(device).eval()
    transform = enc.eval_transforms
    for p in backbone.parameters():
        p.requires_grad_(False)

    adapter = BackboneAdapter(backbone)

    # Load forecaster for pruned model
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=256, n_heads=4, n_layers=2, dropout=0.2,
    ).to(device)
    forecaster.load_state_dict(torch.load(args.forecaster_ckpt, map_location=device))
    forecaster.eval()
    for p in forecaster.parameters():
        p.requires_grad_(False)

    # Load pruned model
    pruned_model = GenericLoRAWithForecasterPruning(
        backbone=enc.model.to(device),
        adapter=adapter,
        n_classes=2,  # dummy
        forecaster=forecaster,
        prune_layer=args.prune_layer,
        keep_ratio=args.keep_ratio,
    ).to(device)
    pruned_model.load_state_dict(torch.load(args.pruned_ckpt, map_location=device))
    pruned_model.eval()

    # Extract embeddings
    print("Extracting embeddings...\n")

    embeddings_unpruned = {}
    embeddings_pruned = {}

    for fold_name in split_df['fold'].unique():
        fold_df = split_df[split_df['fold'] == fold_name]
        print(f"Fold: {fold_name} ({len(fold_df)} samples)")

        for idx, row in tqdm(fold_df.iterrows(), total=len(fold_df), leave=False):
            slide_id = row.get('slide_id', row.get('sample_id'))
            label = row.get('label', row.get(split_df.columns[2]))

            if slide_id not in slide_id_to_path:
                print(f"  Warning: No WSI file found for {slide_id}")
                continue

            wsi_path = slide_id_to_path[slide_id]

            # Extract embeddings
            emb_unpruned = extract_slide_embeddings(
                wsi_path, backbone, args.mag, args.patch_size,
                args.batch_size, str(device)
            )
            emb_pruned = extract_slide_embeddings(
                wsi_path, pruned_model, args.mag, args.patch_size,
                args.batch_size, str(device)
            )

            if emb_unpruned is not None:
                embeddings_unpruned[slide_id] = (emb_unpruned.numpy(), label)
            if emb_pruned is not None:
                embeddings_pruned[slide_id] = (emb_pruned.numpy(), label)

        print(f"  Extracted: unpruned={len(embeddings_unpruned)} pruned={len(embeddings_pruned)}\n")

    # Prepare train/test data
    if 'fold' in split_df.columns:
        train_mask = split_df['fold'] != 'test'
        test_mask = split_df['fold'] == 'test'
    else:
        train_mask = split_df['fold'] == 'train'
        test_mask = split_df['fold'] == 'test'

    train_df = split_df[train_mask]
    test_df = split_df[test_mask]

    # Extract features for sklearn
    def prepare_for_sklearn(embeddings_dict, df):
        X = []
        y = []
        valid_ids = []
        for idx, row in df.iterrows():
            slide_id = row.get('slide_id', row.get('sample_id'))
            if slide_id in embeddings_dict:
                emb, label = embeddings_dict[slide_id]
                X.append(emb)
                y.append(label)
                valid_ids.append(slide_id)
        if len(X) == 0:
            return None, None, None
        return np.array(X), np.array(y), valid_ids

    X_train_unpruned, y_train, _ = prepare_for_sklearn(embeddings_unpruned, train_df)
    X_test_unpruned, y_test, _ = prepare_for_sklearn(embeddings_unpruned, test_df)
    X_train_pruned, _, _ = prepare_for_sklearn(embeddings_pruned, train_df)
    X_test_pruned, _, _ = prepare_for_sklearn(embeddings_pruned, test_df)

    if X_train_unpruned is None or X_train_pruned is None:
        print("Error: No valid embeddings extracted!")
        return

    print(f"Train: {len(X_train_unpruned)} samples")
    print(f"Test: {len(X_test_unpruned)} samples\n")

    # Run logistic regression
    print("Running logistic regression...\n")

    results = {}

    for model_name, X_train, X_test in [
        ("unpruned", X_train_unpruned, X_test_unpruned),
        ("pruned", X_train_pruned, X_test_pruned),
    ]:
        print(f"  {model_name.upper()}:")
        clf = LogisticRegression(max_iter=1000, random_state=args.seed)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        try:
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr", zero_division=0)
        except:
            auc = 0.0

        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1-macro: {f1:.4f}")
        print(f"    AUC: {auc:.4f}\n")

        results[model_name] = {"accuracy": acc, "f1": f1, "auc": auc}

    # Compare
    print(f"{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"Metric          | Unpruned    | Pruned      | Diff")
    print(f"-" * 70)
    for metric in ["accuracy", "f1", "auc"]:
        unpruned_val = results["unpruned"][metric]
        pruned_val = results["pruned"][metric]
        diff = unpruned_val - pruned_val
        print(f"{metric:15} | {unpruned_val:11.4f} | {pruned_val:11.4f} | {diff:+.4f}")

    print(f"{'='*70}\n")

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "evaluation_results.csv")
    print(f"Saved: {output_dir / 'evaluation_results.csv'}\n")


if __name__ == "__main__":
    main()
