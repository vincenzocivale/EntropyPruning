"""Build per-dataset attention caches for the unsupervised multi-dataset EAF corpus.

For each dataset, extracts ``emb_layer{layer_source}`` (patch embeddings) and
``attn_layer{layer_target}`` (CLS->patch attention, last block by default) from
a single FROZEN, pretrained foundation model -- no Phase-1 fine-tuning. The
resulting per-dataset HDF5 caches are later merged by
``MultiH5ForecastDataset`` to train one universal AttentionForecaster.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.collection import build_attention_cache, build_frozen_model, cache_is_valid
from src.utils import set_seed, get_device

DEFAULT_DATASETS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
    "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
    "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds",
]


def main():
    parser = argparse.ArgumentParser(description="Build the unsupervised multi-dataset attention cache")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--layer-source", type=int, default=2)
    parser.add_argument("--layer-target", type=int, default=None,
                        help="Defaults to last block (n_blocks-1).")
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

    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    model, adapter = build_frozen_model(args.model_name, raw_backbone, device)
    layer_target = args.layer_target if args.layer_target is not None else adapter.n_blocks - 1
    print(f"Model: {args.model_name} | embed_dim={adapter.embed_dim} n_blocks={adapter.n_blocks} "
          f"n_patches={adapter.n_patches} | layer_source={args.layer_source} layer_target={layer_target}")

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

        save_path = cache_dir / f"{dataset_name}_{args.model_name}_attn_features.h5"
        if cache_is_valid(save_path, args.layer_source, layer_target, adapter.n_patches, adapter.embed_dim):
            print(f"[{dataset_name}] cache already valid: {save_path}")
            skipped.append(dataset_name)
            continue

        print(f"[{dataset_name}] building cache -> {save_path}")
        build_attention_cache(
            model, adapter, transform, dataset_name, args.base_data_folder, save_path, device,
            layer_source=args.layer_source, layer_target=layer_target,
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_samples_per_split=args.max_samples_per_split,
        )
        built.append(dataset_name)

    print(f"\nBuilt: {built}\nSkipped (already cached): {skipped}\nMissing data splits: {missing}")


if __name__ == "__main__":
    main()
