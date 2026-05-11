"""Phase 3 (new): In-memory WSI-level classification evaluation.

Compares pruned vs non-pruned encoder on a Patho-Bench task.
Extracts ALL tile embeddings per WSI; aggregates them per slide with a
Gated-Attention MIL classifier (Ilse et al., 2018) — no mean pooling.

Precision policy:
  - Unpruned encoder forward: bf16 autocast (fast, accuracy unaffected)
  - Pruned encoder forward:   fp32 (no autocast) — user-requested
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trident.patch_encoder_models import encoder_factory

from src.models import (AttentionForecaster, GenericLoRAWithForecasterPruning,
                        GatedAttentionMIL)
from src.models.backbone_adapter import BackboneAdapter
from src.utils import set_seed, get_device


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
    parser.add_argument("--wsi-list-csv", type=str, default=None,
                        help="Optional CSV with `wsi` and `mpp` columns. Defaults to "
                             "<wsi-dir>/wsi_list.csv if present.")
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
    parser.add_argument("--num-prep-workers", type=int, default=8,
                        help="Parallel subprocesses for one-shot WSI segmentation/indexing")
    parser.add_argument("--eval-tiles-per-wsi", type=int, default=0,
                        help="Tiles per WSI for inference. 0 (default) = ALL valid tiles.")
    # MIL hyperparams
    parser.add_argument("--mil-hidden", type=int, default=256)
    parser.add_argument("--mil-dropout", type=float, default=0.25)
    parser.add_argument("--mil-epochs", type=int, default=50)
    parser.add_argument("--mil-lr", type=float, default=1e-3)
    parser.add_argument("--mil-weight-decay", type=float, default=1e-4)
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

    # Map slide_id to WSI path (recursive: data/LUAD/uuid/*.svs etc.)
    from src.data.wsi_tile_dataset import load_mpp_map
    wsi_dir = Path(args.wsi_dir)
    wsi_files = list(wsi_dir.glob("**/*.svs")) + list(wsi_dir.glob("**/*.ndpi")) + \
                list(wsi_dir.glob("**/*.tif")) + list(wsi_dir.glob("**/*.tiff"))
    slide_id_to_path = {p.stem: str(p) for p in wsi_files}

    mpp_csv = args.wsi_list_csv or str(wsi_dir / "wsi_list.csv")
    mpp_map = load_mpp_map(mpp_csv)

    print(f"Found {len(wsi_files)} WSI files in {wsi_dir}")
    print(f"MPP map: {len(mpp_map)} entries from {mpp_csv if mpp_map else '(none)'}\n")

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

    # Build a single global DataLoader over ALL valid tiles of ALL WSIs.
    # WSITileDataset segments every slide in parallel up-front, then streams
    # tiles via worker processes so the GPU is kept fed continuously.
    from src.data.wsi_tile_dataset import WSITileDataset

    rows_with_files = []
    for _, row in split_df.iterrows():
        slide_id = row.get('slide_id', row.get('sample_id'))
        if slide_id in slide_id_to_path:
            rows_with_files.append((
                slide_id, slide_id_to_path[slide_id],
                row.get('label', row.get(split_df.columns[2])),
                row.get('fold'),
            ))
        else:
            print(f"  Warning: No WSI file found for {slide_id}")

    aligned_paths = [r[1] for r in rows_with_files]
    aligned_slide_ids = [r[0] for r in rows_with_files]
    aligned_labels_raw = [r[2] for r in rows_with_files]
    aligned_folds = [r[3] for r in rows_with_files]
    path_to_slide_idx = {p: i for i, p in enumerate(aligned_paths)}
    n_slides = len(aligned_paths)

    # Encode labels to integers
    unique_labels = sorted(set(aligned_labels_raw))
    label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
    aligned_labels = np.array([label_to_int[l] for l in aligned_labels_raw],
                              dtype=np.int64)
    n_classes = len(unique_labels)
    print(f"Classes: {label_to_int}")

    tiles_arg = None if args.eval_tiles_per_wsi <= 0 else args.eval_tiles_per_wsi
    print(f"Indexing {n_slides} WSIs (tiles_per_wsi={'ALL' if tiles_arg is None else tiles_arg})...")
    eval_dataset = WSITileDataset(
        aligned_paths, transform=transform,
        mag=args.mag, patch_size=args.patch_size,
        tiles_per_wsi=tiles_arg, seed=args.seed,
        verbose=True, mpp_map=mpp_map,
        num_prep_workers=args.num_prep_workers,
    )

    tile_slide_idx = torch.tensor(
        [path_to_slide_idx[rec[0]] for rec in eval_dataset.tile_records],
        dtype=torch.long,
    )

    class _IndexedTileDataset(torch.utils.data.Dataset):
        def __init__(self, base, slide_idx):
            self.base = base
            self.slide_idx = slide_idx
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            return self.base[i], int(self.slide_idx[i])

    indexed = _IndexedTileDataset(eval_dataset, tile_slide_idx)
    loader = DataLoader(
        indexed, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(str(device) == "cuda"),
        drop_last=False,
    )

    # Preallocate per-slide tile-embedding bags (CPU fp32). Counts come from
    # the indexed tile_records so each bag has exact size.
    D = adapter.embed_dim
    slide_n_tiles = [0] * n_slides
    for rec in eval_dataset.tile_records:
        slide_n_tiles[path_to_slide_idx[rec[0]]] += 1

    total_tiles = sum(slide_n_tiles)
    cpu_gb = total_tiles * D * 4 * 2 / (1024**3)  # fp32, two models
    print(f"Total tiles: {total_tiles} | bag storage ≈ {cpu_gb:.2f} GB CPU RAM\n")

    bags_un = [torch.empty(n, D, dtype=torch.float32) for n in slide_n_tiles]
    bags_pr = [torch.empty(n, D, dtype=torch.float32) for n in slide_n_tiles]
    fill_ptr = [0] * n_slides

    use_amp = (str(device) == "cuda")
    amp_ctx_unpruned = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else torch.cuda.amp.autocast(enabled=False)
    )

    with torch.no_grad():
        for tiles, sidx in tqdm(loader, desc="Forward", total=len(loader)):
            tiles = tiles.to(device, non_blocking=True)
            # Unpruned: bf16 autocast (faster, accuracy unchanged for ViT).
            with amp_ctx_unpruned:
                f_un = backbone.forward_features(tiles)[:, 0, :]
            f_un = f_un.float().cpu()
            # Pruned: fp32, no autocast (user-requested).
            f_pr = pruned_model.raw_backbone.forward_features(tiles)[:, 0, :]
            f_pr = f_pr.float().cpu()

            # Scatter into per-slide bags. shuffle=False means tiles arrive in
            # WSI order, so usually only 1-2 distinct slides per batch.
            unique_sidx, inverse = sidx.unique(return_inverse=True)
            for ui, s in enumerate(unique_sidx.tolist()):
                mask = (inverse == ui)
                n = int(mask.sum())
                bags_un[s][fill_ptr[s]:fill_ptr[s]+n] = f_un[mask]
                bags_pr[s][fill_ptr[s]:fill_ptr[s]+n] = f_pr[mask]
                fill_ptr[s] += n

    # Slides with no tiles (segmentation produced nothing) are dropped.
    valid_slides = [i for i in range(n_slides) if fill_ptr[i] > 0]
    n_dropped = n_slides - len(valid_slides)
    print(f"\n  Extracted bags for {len(valid_slides)} slides "
          f"({n_dropped} dropped: empty/failed segmentation)\n")

    # Free GPU model memory before MIL training — MIL fits on a small fraction.
    del backbone, pruned_model, forecaster
    torch.cuda.empty_cache() if str(device) == "cuda" else None

    # Train/test fold masks
    fold_arr = np.array(aligned_folds)
    train_idx = [i for i in valid_slides if fold_arr[i] != "test"]
    test_idx = [i for i in valid_slides if fold_arr[i] == "test"]
    print(f"Train slides: {len(train_idx)} | Test slides: {len(test_idx)}\n")

    def train_eval_mil(bags, name):
        set_seed(args.seed)
        model = GatedAttentionMIL(
            in_dim=D, hidden=args.mil_hidden, n_classes=n_classes,
            dropout=args.mil_dropout,
        ).to(device)
        opt = torch.optim.Adam(
            model.parameters(), lr=args.mil_lr,
            weight_decay=args.mil_weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        train_order = list(train_idx)
        rng = np.random.RandomState(args.seed)
        print(f"  Training MIL on {name}...")
        for epoch in range(args.mil_epochs):
            model.train()
            rng.shuffle(train_order)
            ep_loss, ep_correct = 0.0, 0
            for i in train_order:
                bag = bags[i].to(device, non_blocking=True)
                y = torch.tensor([aligned_labels[i]], device=device, dtype=torch.long)
                logits = model(bag).unsqueeze(0)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                ep_correct += int(logits.argmax(-1).item() == int(y.item()))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{args.mil_epochs}: "
                      f"loss={ep_loss/len(train_order):.4f} "
                      f"acc={ep_correct/len(train_order):.4f}")

        model.eval()
        y_true, y_pred, y_proba = [], [], []
        with torch.no_grad():
            for i in test_idx:
                bag = bags[i].to(device, non_blocking=True)
                logits = model(bag)
                proba = logits.softmax(-1).cpu().numpy()
                y_true.append(int(aligned_labels[i]))
                y_pred.append(int(proba.argmax()))
                y_proba.append(proba)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            auc = (roc_auc_score(y_true, y_proba[:, 1]) if n_classes == 2
                   else roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            auc = 0.0
        print(f"  {name.upper()}: acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}\n")
        return {"accuracy": acc, "f1": f1, "auc": auc}

    results = {
        "unpruned": train_eval_mil(bags_un, "unpruned"),
        "pruned":   train_eval_mil(bags_pr, "pruned"),
    }

    print(f"{'='*70}")
    print(f"COMPARISON (Gated-Attention MIL)")
    print(f"{'='*70}")
    print(f"Metric          | Unpruned    | Pruned      | Diff")
    print(f"-" * 70)
    for metric in ["accuracy", "f1", "auc"]:
        u, p = results["unpruned"][metric], results["pruned"][metric]
        print(f"{metric:15} | {u:11.4f} | {p:11.4f} | {u - p:+.4f}")
    print(f"{'='*70}\n")

    pd.DataFrame(results).T.to_csv(output_dir / "evaluation_results.csv")
    print(f"Saved: {output_dir / 'evaluation_results.csv'}\n")


if __name__ == "__main__":
    main()
