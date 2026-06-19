"""Build per-dataset Phase 3 (Approach 3) distillation feature caches.

For each dataset, extracts the raw output of ``blocks[prune_layer]`` plus
the model's own final CLS/patch tokens from a single frozen pretrained
foundation model -- no Phase 1/2 checkpoints needed, just ``--prune-layer``.
The resulting per-dataset HDF5 caches let ``distill_pruned.py`` train
without ever touching raw images or the frozen pre-prune_layer blocks again.

``distill_pruned.py`` builds/reuses these caches automatically, so running
this script ahead of time is optional -- useful mainly to pre-build the
cache once before launching several distillation runs (e.g. a keep-ratio or
loss-weight sweep) that share the same ``--model-name``/``--prune-layer``.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
from pathlib import Path

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.collection import build_distill_cache, distill_cache_is_valid
from src.models import ThunderBackboneAdapter
from src.utils import set_seed, get_device

DEFAULT_DATASETS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
    "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
    "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds",
]


def main():
    parser = argparse.ArgumentParser(description="Build Phase 3 distillation feature caches")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-split", type=int, default=None,
                        help="Debug cap on samples per split (default: full datasets).")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Output dir for caches (default: checkpoints/unsupervised).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    raw_teacher, transform, _ = get_model_from_name(args.model_name, str(device))
    raw_teacher = raw_teacher.to(device).eval()
    for p in raw_teacher.parameters():
        p.requires_grad_(False)
    adapter = ThunderBackboneAdapter(raw_teacher)
    print(f"Model: {args.model_name} | embed_dim={adapter.embed_dim} n_blocks={adapter.n_blocks} "
          f"n_patches={adapter.n_patches} | prune_layer={args.prune_layer}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("checkpoints") / "unsupervised"
    cache_dir.mkdir(parents=True, exist_ok=True)

    built, skipped, missing = [], [], []
    for dataset_name in args.datasets:
        split_path = Path(args.base_data_folder) / "data_splits" / f"{dataset_name}.json"
        if not split_path.exists():
            print(f"[{dataset_name}] MISSING data split: {split_path}")
            print(f"  -> thunder download {dataset_name} --make-splits "
                  f"--base-data-folder {args.base_data_folder}")
            missing.append(dataset_name)
            continue

        save_path = cache_dir / f"{dataset_name}_{args.model_name}_distill_prune{args.prune_layer}.h5"
        if distill_cache_is_valid(save_path, adapter.n_patches, adapter.embed_dim, adapter.num_prefix_tokens):
            print(f"[{dataset_name}] cache already valid: {save_path}")
            skipped.append(dataset_name)
            continue

        print(f"[{dataset_name}] building cache -> {save_path}")
        build_distill_cache(
            raw_teacher, adapter, transform, dataset_name, args.base_data_folder, save_path, device,
            prune_layer=args.prune_layer, batch_size=args.batch_size, num_workers=args.num_workers,
            max_samples_per_split=args.max_samples_per_split,
        )
        built.append(dataset_name)

    print(f"\nBuilt: {built}\nSkipped (already cached): {skipped}\nMissing data splits: {missing}")


if __name__ == "__main__":
    main()
