#!/usr/bin/env python3
"""Cache frozen-teacher (source embedding, target attention) pairs from THUNDER tiles.

Runs each requested THUNDER tile exactly once through a frozen tile-encoder
(via ``FrozenTimmAttentionTeacher``, the same extractor used by the online
trainer) and persists the source-layer patch tokens and the target-layer
CLS-to-patch attention to HDF5. The cache uses the pre-existing on-disk
layout (``labels``, ``emb_layer{source}``, ``attn_layer{target}`` per split)
so it can be read directly by ``src/data/h5_dataset.py::H5ForecastDataset``.

The point is to pay the backbone forward pass once per tile instead of once
per tile *per epoch*: with the cache, many forecaster epochs cost only a
cheap read + small-model forward/backward, matching the economics of the
historical offline pipeline instead of the fully-online one.

Usage:
    python scripts/build_thunder_online_forecaster_cache.py \\
        --model-name titan \\
        --base-data-folder /path/to/thunder/datasets \\
        --cache-dir results/thunder_forecaster_cache \\
        --source-layer 2 \\
        --dry-run          # print the planned tile counts / disk footprint first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.data.thunder_multi import build_multi_thunder_split_loader
from src.models import ThunderBackboneAdapter
from src.training.online_attention_distillation import FrozenTimmAttentionTeacher
from src.utils import get_device, set_seed

SPLITS = ("train", "val", "test")


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _discover_datasets(base_data_folder: str) -> list[str]:
    splits_dir = Path(base_data_folder) / "data_splits"
    names = sorted(path.stem for path in splits_dir.glob("*.json"))
    if not names:
        raise FileNotFoundError(f"No THUNDER manifests found in {splits_dir}")
    return names


def _bytes_to_human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PiB"


def _capped_loader(
    dataset_name: str,
    base_data_folder: str,
    transform,
    split: str,
    cap: int | None,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, int]:
    """Single-dataset loader; randomly subsamples to ``cap`` tiles if given.

    Reuses ``build_multi_thunder_split_loader`` for dataset construction, then
    replaces the sampling policy so a cap yields a uniform random subsample
    regardless of split (the shared loader only shuffles ``train``).
    """
    loader, info = build_multi_thunder_split_loader(
        [dataset_name], base_data_folder, transform, split,
        batch_size=batch_size, num_workers=0,
        sampler_mode="proportional", seed=seed,
    )
    n_available = info[0]["n_samples"]
    dataset = loader.dataset
    if cap is not None and cap < n_available:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_available)[:cap].tolist()
        dataset = Subset(dataset, indices)
    n_target = len(dataset)
    real_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    return real_loader, n_target


def _probe_token_geometry(
    teacher: FrozenTimmAttentionTeacher, loader: DataLoader, device: torch.device, amp: bool
) -> tuple[int, int]:
    images, _labels, _dataset_idx = next(iter(loader))
    images = images.to(device, non_blocking=True)
    with torch.inference_mode(), _autocast(device, amp):
        source, target = teacher(images)
    return source.shape[1], target.shape[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-data-folder", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=None, help="Defaults to every downloaded THUNDER dataset.")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))

    parser.add_argument("--source-layer", type=int, default=2)
    parser.add_argument("--target-layer", type=int, default=None, help="Defaults to the last transformer block.")
    parser.add_argument("--target-normalization", choices=["patch", "none"], default="patch")

    parser.add_argument(
        "--max-tiles-per-split", type=int, default=None,
        help="Cap per (dataset, split); default caches the full split. Capped splits are a seeded uniform random subsample.",
    )

    parser.add_argument("--batch-size", type=int, default=256, help="Forward-only extraction: no backward pass, so this can be large.")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="On-disk storage dtype.")
    parser.add_argument(
        "--compression", choices=["none", "lzf", "gzip"], default="none",
        help=(
            "Benchmarked on real ViT patch tokens: lzf gives ~1.00x (no reduction) and "
            "gzip only ~1.08x, both at 5-15x slower read/write -- these activations are "
            "high-entropy floats and don't compress. Default is 'none'."
        ),
    )
    parser.add_argument("--gzip-level", type=int, default=4)

    parser.add_argument("--overwrite", action="store_true", help="Recompute splits that are already cached.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and estimated disk footprint; extract nothing.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device()

    available = _discover_datasets(args.base_data_folder)
    dataset_names = list(args.datasets or available)
    unknown = sorted(set(dataset_names) - set(available))
    if unknown:
        raise ValueError(f"Unknown THUNDER datasets: {unknown}")

    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    target_layer = args.target_layer if args.target_layer is not None else adapter.n_blocks - 1
    teacher = FrozenTimmAttentionTeacher(
        raw_backbone, adapter,
        source_layer=args.source_layer, target_layer=target_layer,
        target_normalization=args.target_normalization,
    ).to(device)

    target_tag = "last" if args.target_layer is None else f"{args.target_layer:02d}"
    run_dir = Path(args.cache_dir) / f"{args.model_name}_src{args.source_layer:02d}_tgt{target_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if args.dtype == "fp16" else np.float32
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    itemsize = 2 if args.dtype == "fp16" else 4
    compression = None if args.compression == "none" else args.compression
    compression_opts = args.gzip_level if args.compression == "gzip" else None

    # --- Plan: sample counts per (dataset, split) + one real forward to learn token geometry ---
    plan: list[dict[str, Any]] = []
    n_patches_source = n_patches_target = None
    for name in dataset_names:
        for split in args.splits:
            _loader, info = build_multi_thunder_split_loader(
                [name], args.base_data_folder, transform, split,
                batch_size=args.batch_size, num_workers=0,
                sampler_mode="proportional", seed=args.seed,
            )
            n_available = info[0]["n_samples"]
            n_target = n_available if args.max_tiles_per_split is None else min(n_available, args.max_tiles_per_split)
            if n_patches_source is None:
                n_patches_source, n_patches_target = _probe_token_geometry(teacher, _loader, device, args.amp)
                print(
                    f"[cache] runtime token geometry: source={n_patches_source} patches, "
                    f"target={n_patches_target} patches, embed_dim={adapter.embed_dim}"
                )
            plan.append({"dataset": name, "split": split, "n_available": n_available, "n_target": n_target})

    bytes_per_tile_source = n_patches_source * adapter.embed_dim * itemsize
    bytes_per_tile_target = n_patches_target * itemsize
    total_tiles = sum(p["n_target"] for p in plan)
    total_bytes = total_tiles * (bytes_per_tile_source + bytes_per_tile_target)

    print(f"[cache] plan: {len(plan)} (dataset, split) shards, {total_tiles:,} tiles total")
    print(
        f"[cache] per-tile: source {_bytes_to_human(bytes_per_tile_source)} "
        f"+ target {_bytes_to_human(bytes_per_tile_target)} (dtype={args.dtype}, compression={args.compression})"
    )
    print(f"[cache] estimated on-disk size: {_bytes_to_human(total_bytes)} (pre-compression) under {run_dir}")
    for p in plan:
        cap_note = "" if p["n_target"] == p["n_available"] else f" (capped from {p['n_available']:,})"
        print(f"    {p['dataset']:>20s}/{p['split']:<5s}: {p['n_target']:>9,} tiles{cap_note}")

    if args.dry_run:
        print("[cache] --dry-run: not extracting anything.")
        teacher.close()
        return

    meta = {
        "model_name": args.model_name,
        "source_layer": args.source_layer,
        "target_layer": target_layer,
        "target_normalization": args.target_normalization,
        "n_patches_source": n_patches_source,
        "n_patches_target": n_patches_target,
        "embed_dim": adapter.embed_dim,
        "dtype": args.dtype,
        "datasets": dataset_names,
        "splits": args.splits,
        "max_tiles_per_split": args.max_tiles_per_split,
        "seed": args.seed,
    }
    (run_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))

    # --- Extract ---
    t_start = time.time()
    for name in dataset_names:
        h5_path = run_dir / f"{name}.h5"
        mode = "a" if h5_path.exists() else "w"
        with h5py.File(h5_path, mode) as f:
            for split in args.splits:
                if split in f:
                    if not args.overwrite:
                        print(f"[cache] skip {name}/{split}: already cached")
                        continue
                    del f[split]

                n_target = next(p["n_target"] for p in plan if p["dataset"] == name and p["split"] == split)
                loader, n_target = _capped_loader(
                    name, args.base_data_folder, transform, split,
                    args.max_tiles_per_split, args.batch_size, args.num_workers, args.seed,
                )

                grp = f.create_group(split)
                chunk_tiles = max(1, min(128, n_target))
                ds_emb = grp.create_dataset(
                    f"emb_layer{args.source_layer}",
                    shape=(n_target, n_patches_source, adapter.embed_dim), dtype=np_dtype,
                    chunks=(chunk_tiles, n_patches_source, adapter.embed_dim),
                    compression=compression, compression_opts=compression_opts,
                )
                ds_attn = grp.create_dataset(
                    f"attn_layer{target_layer}",
                    shape=(n_target, n_patches_target), dtype=np_dtype,
                    chunks=(chunk_tiles, n_patches_target),
                    compression=compression, compression_opts=compression_opts,
                )
                ds_labels = grp.create_dataset("labels", shape=(n_target,), dtype="i4")

                ptr = 0
                pbar = tqdm(total=n_target, desc=f"{name}/{split}", unit="tile", unit_scale=True)
                with torch.inference_mode():
                    for images, labels, _dataset_idx in loader:
                        images = images.to(device, non_blocking=True)
                        with _autocast(device, args.amp):
                            source, target = teacher(images)
                        take = source.shape[0]
                        ds_emb[ptr:ptr + take] = source.to(torch_dtype).cpu().numpy()
                        ds_attn[ptr:ptr + take] = target.to(torch_dtype).cpu().numpy()
                        ds_labels[ptr:ptr + take] = labels.numpy().astype("i4")
                        ptr += take
                        pbar.update(take)
                pbar.close()
                assert ptr == n_target, f"{name}/{split}: wrote {ptr}, expected {n_target}"
        elapsed = time.time() - t_start
        print(f"[cache] {name}: done (elapsed {elapsed / 60:.1f} min total)")

    teacher.close()
    print(f"[cache] finished in {(time.time() - t_start) / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
