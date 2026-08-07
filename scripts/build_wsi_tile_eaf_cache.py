#!/usr/bin/env python
"""Cache frozen-teacher (source embedding, target attention) pairs from WSI tiles.

The online WSI pipeline (``train_wsi_tile_eaf_online.py``) deliberately keeps
no cache: each epoch rotates through a fresh subset of WSI/tiles, and the
backbone forward that produces the source tokens and target CLS attention is
recomputed every time a tile is seen, including every epoch of validation.

This script inverts that trade-off. It draws one large, fixed, epoch-independent
pool of tiles (``--tiles-per-wsi`` per WSI, over every WSI in the requested
split unless capped with ``--max-wsis-per-split``), runs each tile once through
``OnlineAttentionTeacher`` (the same extractor the online trainer uses, so the
cached signal is identical to what online training would compute), and writes
the result to HDF5. Many cheap forecaster epochs can then read from disk
instead of re-running the backbone.

Output layout, one HDF5 file per split under
``{cache-dir}/{model}_src{S:02d}_tgt{T:02d}_wsi/``:

    {split}.h5
    ├── emb_layer{S}   [n_tiles, n_patches, embed_dim]  fp16
    ├── attn_layer{T}  [n_tiles, n_patches]              fp16
    ├── slide_idx      [n_tiles]                         i4  (row into {split}_slides.json)
    └── labels         [n_tiles]                         i4  (unused placeholder, kept so
                                                                src/data/h5_dataset.py::H5ForecastDataset
                                                                can read this cache directly)
    {split}_slides.json  -> [{"slide_id", "case_id", "cohort"}, ...] indexed by slide_idx

Usage:
    python scripts/build_wsi_tile_eaf_cache.py \\
        --model-name titan \\
        --manifest $EAF_WSI_ROOT/datasets/pretraining/tcga_eaf_multicohort_v1/manifests/slides.csv \\
        --data-root $EAF_WSI_ROOT \\
        --cache-dir $EAF_WSI_ROOT/cache/tile_eaf \\
        --source-layer 2 \\
        --dry-run          # print the planned tile counts / disk footprint first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.data.wsi_tile_stream import SlideRecord, WSITileDataset, load_wsi_manifest
from src.models import ThunderBackboneAdapter
from src.models.online_tile_eaf import OnlineAttentionTeacher, load_checkpoint_flexibly
from src.utils import get_device, set_seed

SPLITS = ("train", "val", "test")


def _autocast(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _bytes_to_human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PiB"


def _sample_plan(records: list[SlideRecord], tiles_per_wsi: int, seed: int) -> list[tuple[int, int]]:
    """Fixed, seeded (slide_index, coord_index) pool: up to ``tiles_per_wsi`` per WSI.

    Independent of any training epoch/sampler; sampling without replacement
    whenever a WSI has enough coordinates, matching the online sampler's policy.
    """
    plan: list[tuple[int, int]] = []
    for slide_index, record in enumerate(records):
        n = record.coord_count
        take = min(tiles_per_wsi, n)
        rng = np.random.default_rng(seed + slide_index)
        replace = n < take
        coord_indices = rng.choice(n, size=take, replace=replace)
        plan.extend((slide_index, int(c)) for c in coord_indices)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--source-layer", type=int, default=2)
    parser.add_argument("--target-layer", type=int, default=None, help="Defaults to the last transformer block.")

    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=list(SPLITS))
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--slide-group", nargs="+", default=["diagnostic"])
    parser.add_argument("--exclude-cohort", nargs="*", default=[])
    parser.add_argument("--default-patch-size", type=int, default=512)
    parser.add_argument("--tile-size-at-target-mag", type=int, default=None)

    parser.add_argument(
        "--tiles-per-wsi", type=int, default=32,
        help="Fixed pool size per WSI, sampled once with a seed (not per epoch).",
    )
    parser.add_argument(
        "--max-wsis-per-split", type=int, default=None,
        help="Optional cap on the number of WSI included per split (seeded random subset).",
    )

    parser.add_argument("--batch-size", type=int, default=128, help="Forward-only extraction: no backward pass, so this can be large.")
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--slide-cache-size", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")

    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="On-disk storage dtype.")
    parser.add_argument(
        "--compression", choices=["none", "lzf", "gzip"], default="none",
        help=(
            "Benchmarked on real CONCH1.5 patch tokens: lzf gives ~1.00x (no reduction) "
            "and gzip only ~1.08x, both at 5-15x slower read/write -- ViT activations are "
            "high-entropy floats and don't compress. Default is 'none'; only override if "
            "your encoder/layer combination turns out differently."
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
    if device.type != "cuda":
        raise RuntimeError("WSI tile-EAF caching requires a CUDA device")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        args.amp_dtype = "fp16"

    backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    backbone = backbone.to(device).eval()
    adapter = ThunderBackboneAdapter(backbone)
    target_layer = args.target_layer if args.target_layer is not None else adapter.n_blocks - 1
    if args.teacher_checkpoint:
        missing, unexpected = load_checkpoint_flexibly(backbone, args.teacher_checkpoint)
        print(f"[cache] teacher checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    teacher = OnlineAttentionTeacher(backbone, adapter, args.source_layer, target_layer)

    split_records = load_wsi_manifest(
        args.manifest, args.data_root,
        split_column=args.split_column, val_fraction=args.val_fraction, split_seed=args.split_seed,
        include_slide_groups=args.slide_group, exclude_cohorts=args.exclude_cohort,
        default_patch_size=args.default_patch_size, crop_size_at_target_mag=args.tile_size_at_target_mag,
    )

    # --- Plan: fixed tile pool per requested split ---
    plans: dict[str, tuple[list[SlideRecord], list[tuple[int, int]]]] = {}
    for split in args.splits:
        records = list(split_records.get(split, []))
        if not records:
            print(f"[cache] skip split '{split}': empty")
            continue
        if args.max_wsis_per_split is not None and args.max_wsis_per_split < len(records):
            rng = np.random.default_rng(args.seed)
            keep = sorted(rng.permutation(len(records))[: args.max_wsis_per_split].tolist())
            records = [records[i] for i in keep]
        plan = _sample_plan(records, args.tiles_per_wsi, args.seed)
        plans[split] = (records, plan)

    if not plans:
        raise ValueError(f"None of the requested splits {args.splits} have data")

    # --- Probe runtime token geometry with one real batch ---
    probe_split = next(iter(plans))
    probe_records, probe_plan = plans[probe_split]
    probe_ds = WSITileDataset(probe_records, transform, augment=False, slide_cache_size=args.slide_cache_size)
    probe_loader = DataLoader(probe_ds, sampler=probe_plan[: args.batch_size], batch_size=args.batch_size)
    images, _slide_idx = next(iter(probe_loader))
    images = images.to(device, non_blocking=True)
    with torch.inference_mode(), _autocast(device, args.amp_dtype):
        source, target = teacher.extract(images)
    n_patches_source, n_patches_target = source.shape[1], target.shape[1]
    print(
        f"[cache] runtime token geometry: source={n_patches_source} patches, "
        f"target={n_patches_target} patches, embed_dim={adapter.embed_dim}"
    )

    itemsize = 2 if args.dtype == "fp16" else 4
    bytes_per_tile = n_patches_source * adapter.embed_dim * itemsize + n_patches_target * itemsize
    total_tiles = sum(len(plan) for _records, plan in plans.values())
    print(f"[cache] plan: {len(plans)} split(s), {total_tiles:,} tiles total")
    print(f"[cache] per-tile: {_bytes_to_human(bytes_per_tile)} (dtype={args.dtype}, compression={args.compression})")
    print(f"[cache] estimated on-disk size: {_bytes_to_human(total_tiles * bytes_per_tile)} (pre-compression)")
    for split, (records, plan) in plans.items():
        print(f"    {split:>5s}: {len(records):>5,} WSI x up to {args.tiles_per_wsi} tiles = {len(plan):>9,} tiles")

    if args.dry_run:
        print("[cache] --dry-run: not extracting anything.")
        return

    target_tag = "last" if args.target_layer is None else f"{args.target_layer:02d}"
    run_dir = Path(args.cache_dir) / f"{args.model_name}_src{args.source_layer:02d}_tgt{target_tag}_wsi"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": args.model_name,
        "source_layer": args.source_layer,
        "target_layer": target_layer,
        "n_patches_source": n_patches_source,
        "n_patches_target": n_patches_target,
        "embed_dim": adapter.embed_dim,
        "dtype": args.dtype,
        "tiles_per_wsi": args.tiles_per_wsi,
        "max_wsis_per_split": args.max_wsis_per_split,
        "manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
    }
    (run_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))

    np_dtype = np.float16 if args.dtype == "fp16" else np.float32
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    compression = None if args.compression == "none" else args.compression
    compression_opts = args.gzip_level if args.compression == "gzip" else None

    t_start = time.time()
    for split, (records, plan) in plans.items():
        h5_path = run_dir / f"{split}.h5"
        done_marker = run_dir / f"{split}.done"
        if h5_path.exists() and done_marker.exists() and not args.overwrite:
            print(f"[cache] skip {split}: already cached at {h5_path}")
            continue
        if h5_path.exists() and not done_marker.exists():
            print(
                f"[cache] {split}: found {h5_path} without a completion marker "
                "(likely from an interrupted/killed run, or a different plan) -- rewriting from scratch"
            )

        (run_dir / f"{split}_slides.json").write_text(
            json.dumps(
                [{"slide_id": r.slide_id, "case_id": r.case_id, "cohort": r.cohort} for r in records],
                indent=2,
            )
        )

        base_ds = WSITileDataset(records, transform, augment=False, slide_cache_size=args.slide_cache_size)
        loader = DataLoader(
            base_ds, sampler=plan, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        n_target = len(plan)
        chunk_tiles = max(1, min(128, n_target))
        with h5py.File(h5_path, "w") as f:
            ds_emb = f.create_dataset(
                f"emb_layer{args.source_layer}", shape=(n_target, n_patches_source, adapter.embed_dim),
                dtype=np_dtype, chunks=(chunk_tiles, n_patches_source, adapter.embed_dim),
                compression=compression, compression_opts=compression_opts,
            )
            ds_attn = f.create_dataset(
                f"attn_layer{target_layer}", shape=(n_target, n_patches_target),
                dtype=np_dtype, chunks=(chunk_tiles, n_patches_target),
                compression=compression, compression_opts=compression_opts,
            )
            ds_slide_idx = f.create_dataset("slide_idx", shape=(n_target,), dtype="i4")
            ds_labels = f.create_dataset("labels", shape=(n_target,), dtype="i4")  # unused; kept for H5ForecastDataset compat

            ptr = 0
            pbar = tqdm(total=n_target, desc=f"{split}", unit="tile", unit_scale=True)
            with torch.inference_mode():
                for images, slide_indices in loader:
                    images = images.to(device, non_blocking=True)
                    with _autocast(device, args.amp_dtype):
                        source, target_attn = teacher.extract(images)
                    take = source.shape[0]
                    ds_emb[ptr:ptr + take] = source.to(torch_dtype).cpu().numpy()
                    ds_attn[ptr:ptr + take] = target_attn.to(torch_dtype).cpu().numpy()
                    ds_slide_idx[ptr:ptr + take] = slide_indices.numpy().astype("i4")
                    ptr += take
                    pbar.update(take)
            pbar.close()
            assert ptr == n_target, f"{split}: wrote {ptr}, expected {n_target}"
        done_marker.write_text(f"{ptr}\n")
        elapsed = time.time() - t_start
        print(f"[cache] {split}: done, {ptr:,} tiles -> {h5_path} (elapsed {elapsed / 60:.1f} min total)")

    print(f"[cache] finished in {(time.time() - t_start) / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
