#!/usr/bin/env python3
"""Unified operational CLI for the refactored EAF WSI data/cache pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi.corpora import HISTAI_SUBSETS, download_histai, plan_histai
from src.data.wsi.layout import StoreLayout  # noqa: E402
from src.data.wsi.manifest import read_manifest as read_slide_manifest  # noqa: E402
from src.wsi_pipeline.archive import (  # noqa: E402
    LifecycleStage,
    SlideLifecycleEvidence,
    classify_stage,
    release_preconditions,
    verify_pixel_archive,
)
from src.wsi_pipeline.cache_contracts import TileCacheSpec  # noqa: E402
from src.wsi_pipeline.cache_index import build_tile_cache_index  # noqa: E402
from src.wsi_pipeline.cache_io import validate_cache  # noqa: E402
from src.wsi_pipeline.numpy_store import convert_h5, verify_conversion  # noqa: E402
from src.wsi_pipeline.model_adapters import PrunedLoRATileTeacherAdapter  # noqa: E402
from src.wsi_pipeline.registry import load_slides, write_manifest  # noqa: E402
from src.wsi_pipeline.tile_cache_pipeline import (  # noqa: E402
    TileCacheItem,
    TileCacheRunConfig,
    build_encoder,
    cache_many_slides,
    autotune_loader,
    probe_batch_size,
)


def _root(args: argparse.Namespace) -> Path:
    return StoreLayout.from_root(args.data_root).root


def cmd_layout(args: argparse.Namespace) -> None:
    layout = StoreLayout.from_root(args.data_root)
    layout.ensure_base_dirs()
    print(json.dumps({
        "root": str(layout.root),
        "sources": str(layout.sources),
        "datasets": str(layout.datasets),
        "caches": str(layout.caches),
        "archives": str(layout.archives),
        "checkpoints": str(layout.checkpoints),
        "results": str(layout.results),
        "logs": str(layout.logs),
    }, indent=2))


def cmd_plan_histai(args: argparse.Namespace) -> None:
    path = plan_histai(
        _root(args), token=args.token, subsets=args.subset, force=args.force
    )
    print(path)


def cmd_download_histai(args: argparse.Namespace) -> None:
    path = download_histai(
        _root(args), subsets=args.subset, workers=args.workers, token=args.token
    )
    print(path)


def cmd_validate_cache(args: argparse.Namespace) -> None:
    print(json.dumps(validate_cache(args.path, expected_kind=args.kind), indent=2))


def cmd_convert_numpy(args: argparse.Namespace) -> None:
    """Resume a checked conversion of generated cache HDF5 files."""
    root = _root(args)
    paths = sorted((root / args.scope).rglob("*.h5"))
    if args.max_files is not None:
        paths = paths[:args.max_files]
    counts = {"converted": 0, "verified_existing": 0, "errors": 0}
    failures: list[dict[str, str]] = []
    def record(index: int, result: tuple[str, str, str | None]) -> None:
        status, path, error = result
        counts[status] += 1
        if error:
            failures.append({"path": path, "error": error})
        if index % args.report_every == 0:
            print(json.dumps({"processed": index, "total": len(paths), **counts}), flush=True)

    if args.workers == 1:
        for index, path in enumerate(paths, 1):
            record(index, _convert_one_numpy(path))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, result in enumerate(pool.map(_convert_one_numpy, paths, chunksize=16), 1):
                record(index, result)
    print(json.dumps({"processed": len(paths), "total": len(paths), **counts,
                      "failures": failures[:20]}, indent=2), flush=True)
    if counts["errors"]:
        raise SystemExit(1)


def _convert_one_numpy(source: Path) -> tuple[str, str, str | None]:
    try:
        destination = source.with_suffix(".npyd")
        if destination.exists():
            verify_conversion(source, destination)
            return "verified_existing", str(source), None
        convert_h5(source)
        return "converted", str(source), None
    except Exception as exc:
        return "errors", str(source), repr(exc)


def cmd_index_tile_cache(args: argparse.Namespace) -> None:
    rows = build_tile_cache_index(
        args.slides,
        args.cache_root,
        args.output,
        expected_cache_id=args.cache_id,
        data_root=_root(args),
    )
    print(json.dumps({"output": str(args.output), "slides": len(rows)}, indent=2))


def cmd_cache_tile(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch

    device = torch.device(
        args.device if (torch.cuda.is_available() or not args.device.startswith("cuda")) else "cpu"
    )
    if args.pruned_adapter_ckpt is not None:
        # Pruned-input cache: embeddings come from a tile-EAF Stage-2 pruned+LoRA
        # encoder instead of the frozen --encoder. See PrunedLoRATileTeacherAdapter
        # for why --thunder-model-name/--prune-layer/--keep-ratio/--forecaster-ckpt
        # are all optional (auto-resolved from the checkpoint) and why
        # final_attention in the resulting cache is a placeholder.
        adapter = PrunedLoRATileTeacherAdapter(
            pruned_adapter_ckpt=args.pruned_adapter_ckpt,
            forecaster_ckpt=args.forecaster_ckpt,
            thunder_model_name=args.thunder_model_name,
            prune_layer=args.prune_layer,
            keep_ratio=args.keep_ratio,
            lora_r=args.pruned_lora_r,
            lora_alpha=args.pruned_lora_alpha,
            lora_dropout=args.pruned_lora_dropout,
            device=device,
        )
        print(
            f"[eaf-cache-tile] pruned-input cache: encoder={adapter.name} "
            f"prune_layer={adapter.prune_layer} keep_ratio={adapter.keep_ratio} "
            f"forecaster={adapter.forecaster_checkpoint} -- convention: name "
            "--output-dir '<base-cache-dir>__pruned_<tile-run-name>' so it is never "
            "confused with the base frozen-encoder cache (see docs/pipeline.md)"
        )
    else:
        adapter = build_encoder(args.encoder, token=args.hf_token, device=device)

    items: list[TileCacheItem] = []
    if args.manifest is not None:
        data_root = _root(args)
        for entry in read_slide_manifest(args.manifest):
            if not entry.raw_path or not entry.coords_path:
                continue
            wsi_path = Path(entry.raw_path).expanduser()
            coords_path = Path(entry.coords_path).expanduser()
            if not wsi_path.is_absolute():
                wsi_path = (data_root / wsi_path).resolve()
            if not coords_path.is_absolute():
                coords_path = (data_root / coords_path).resolve()
            items.append(
                TileCacheItem(
                    slide_id=entry.slide_id,
                    case_id=entry.case_id or entry.slide_id,
                    wsi_path=wsi_path,
                    coords_path=coords_path,
                )
            )
    else:
        if args.coords_registry is None:
            raise SystemExit("--coords-registry is required with --slides")
        slides = {entry.slide_id: entry for entry in load_slides(args.slides)}
        coords_df = pd.read_csv(args.coords_registry)
        if not {"slide_id", "path"}.issubset(coords_df.columns):
            raise SystemExit("--coords-registry requires columns: slide_id,path")
        for row in coords_df.to_dict("records"):
            slide_id = str(row["slide_id"])
            entry = slides.get(slide_id)
            if entry is None:
                continue
            coords_path = Path(str(row["path"]))
            if not coords_path.is_absolute():
                coords_path = (args.coords_registry.parent / coords_path).resolve()
            items.append(
                TileCacheItem(
                    slide_id=slide_id,
                    case_id=entry.case_id or slide_id,
                    wsi_path=entry.wsi_path,
                    coords_path=coords_path,
                )
            )
    if not items:
        raise SystemExit("No (slide_id) matched between --slides and --coords-registry")
    if args.max_slides is not None:
        items = items[: args.max_slides]

    tuning_results = None
    if args.autotune:
        batch_size, args.num_workers, args.prefetch_factor, tuning_results = autotune_loader(
            items[0],
            adapter,
            device=device,
            batch_size_candidates=tuple(args.batch_size_candidates),
            worker_candidates=tuple(args.worker_candidates),
            prefetch_candidates=tuple(args.prefetch_candidates),
            benchmark_batches=args.benchmark_batches,
            openslide_cache_bytes=args.openslide_cache_mib * 2**20,
        )
        print(json.dumps({"autotune": tuning_results, "selected": {
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
        }}, indent=2))
    elif args.batch_size == "auto":
        batch_size = probe_batch_size(
            adapter,
            device=device,
            candidates=tuple(args.batch_size_candidates),
        )
        print(
            f"[eaf-cache-tile] auto-probed batch_size={batch_size} "
            f"(candidates={args.batch_size_candidates})"
        )
    else:
        batch_size = int(args.batch_size)

    spec = TileCacheSpec(
        tile_encoder=adapter.name,
        model_revision=adapter.revision,
        early_layer=args.early_layer,
        input_mag=args.input_mag,
        patch_size=args.patch_size,
        stride=args.stride,
        input_mpp=args.input_mpp,
        dtype=args.storage_dtype,
        dataset=args.dataset,
    )
    config = TileCacheRunConfig(
        output_dir=args.output_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        device=str(device),
        compression=None if args.compression == "none" else args.compression,
        overwrite=args.overwrite,
        profile=args.profile_json is not None,
        slide_loader_chunk_size=args.slide_loader_chunk_size,
        persistent_workers=args.persistent_workers,
        openslide_cache_bytes=args.openslide_cache_mib * 2**20,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[eaf-cache-tile] cache_id={spec.cache_id} n_slides={len(items)} "
        f"batch_size={batch_size} slide_loader_chunk_size={args.slide_loader_chunk_size} "
        f"persistent_workers={args.persistent_workers} "
        f"openslide_cache_mib={args.openslide_cache_mib}"
    )

    rows = cache_many_slides(items, adapter=adapter, spec=spec, config=config)
    write_manifest(rows, args.output_dir / "manifest.csv")
    profiled = [row for row in rows if "elapsed_seconds" in row]
    profile_summary = None
    if profiled:
        total_tiles = sum(int(row["n_tiles"]) for row in profiled)
        total_elapsed = sum(float(row["elapsed_seconds"]) for row in profiled)
        profile_summary = {
            "slides": len(profiled),
            "tiles": total_tiles,
            "elapsed_seconds": total_elapsed,
            "tiles_per_second": total_tiles / max(total_elapsed, 1e-9),
            "data_wait_seconds": sum(float(row["data_wait_seconds"]) for row in profiled),
            "transfer_compute_seconds": sum(
                float(row["transfer_compute_seconds"]) for row in profiled
            ),
            "write_seconds": sum(float(row["write_seconds"]) for row in profiled),
            "peak_cuda_memory_gib": max(
                (float(row.get("peak_cuda_memory_gib", 0.0)) for row in profiled),
                default=0.0,
            ),
        }
    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(
            json.dumps(
                {"summary": profile_summary, "slides": rows, "autotune": tuning_results},
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

    built = [r for r in rows if r.get("status") == "built"]
    skipped = [r for r in rows if r.get("status") == "skipped_valid"]
    errors = [r for r in rows if r.get("status") == "error"]
    print(
        json.dumps(
            {
                "total": len(rows),
                "built": len(built),
                "skipped_valid": len(skipped),
                "errors": len(errors),
                "cache_id": spec.cache_id,
                "manifest": str(args.output_dir / "manifest.csv"),
                "profile": profile_summary,
            },
            indent=2,
        )
    )
    if errors:
        for row in errors[:10]:
            print(f"  ERROR {row['slide_id']}: {row['error']}", file=sys.stderr)
        raise SystemExit(1)


def cmd_verify_archive(args: argparse.Namespace) -> None:
    print(json.dumps(verify_pixel_archive(args.path), indent=2))


def cmd_lifecycle(args: argparse.Namespace) -> None:
    evidence = SlideLifecycleEvidence(
        slide_id=args.slide_id,
        raw_path=args.raw_path,
        coords_path=args.coords_path,
        tile_cache_paths=tuple(args.tile_cache or ()),
        wsi_cache_paths=tuple(args.wsi_cache or ()),
        pixel_archive_path=args.archive_path,
        remote_recoverable=args.remote_recoverable,
    )
    stage = classify_stage(evidence)
    checks = release_preconditions(evidence)
    print(
        json.dumps(
            {"slide_id": args.slide_id, "stage": stage.value, "release_preconditions": checks},
            indent=2,
            default=str,
        )
    )
    if stage != LifecycleStage.RAW_RELEASABLE:
        print(
            "[eaf-archive] read-only classification only; there is no raw-deletion "
            "command in this repository.",
            file=sys.stderr,
        )


def add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Canonical $EAF_WSI_ROOT. Defaults to the EAF_WSI_ROOT environment variable.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    layout = sub.add_parser("layout", help="Create/print the canonical runtime layout")
    add_root(layout)
    layout.set_defaults(func=cmd_layout)


    data = sub.add_parser("data", help="Plan/download the unlabeled HISTAI training corpus")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    p = data_sub.add_parser("plan-histai")
    add_root(p)
    p.add_argument("--subset", action="append", choices=HISTAI_SUBSETS)
    p.add_argument("--token")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-list every requested subset even if already present in plan.csv.",
    )
    p.set_defaults(func=cmd_plan_histai)

    p = data_sub.add_parser("download-histai")
    add_root(p)
    p.add_argument("--subset", action="append", choices=HISTAI_SUBSETS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--token")
    p.set_defaults(func=cmd_download_histai)


    cache = sub.add_parser("cache", help="Build/inspect versioned offline teacher caches")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    p = cache_sub.add_parser("validate")
    p.add_argument("path", type=Path)
    p.add_argument("--kind", choices=("tile_eaf", "wsi_eaf"))
    p.set_defaults(func=cmd_validate_cache)

    p = cache_sub.add_parser("convert-numpy", help="Convert generated HDF5 to verified .npyd directories")
    add_root(p)
    p.add_argument("--scope", choices=("caches", "datasets"), default="caches")
    p.add_argument("--max-files", type=int, help="Limit for a trial conversion")
    p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_convert_numpy)

    p = cache_sub.add_parser(
        "index-tile", help="Validate and index compact per-slide caches for training"
    )
    add_root(p)
    p.add_argument("--slides", type=Path, required=True, help="HISTAI slides.csv")
    p.add_argument("--cache-root", type=Path, action="append", required=True)
    p.add_argument("--cache-id")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_index_tile_cache)

    p = cache_sub.add_parser(
        "tile",
        help=(
            "Offline compact cache: final CLS-to-patch attention and pooled tile "
            "embedding. Early-layer patch tokens are recomputed during Tile-EAF training."
        ),
    )
    add_root(p)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest", type=Path,
        help="Canonical SlideRecord CSV containing slide_id,case_id,raw_path,coords_path",
    )
    source.add_argument("--slides", type=Path, help="CSV: slide_id,wsi_path[,case_id]")
    p.add_argument(
        "--coords-registry", type=Path,
        help="CSV: slide_id,path (required with --slides; TRIDENT *_patches.h5)",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--encoder", default="conch_v15",
        help=(
            "Tile encoder to cache. 'conch_v15' uses TITAN's return_conch() accessor; "
            "any other name (e.g. uni2h, virchow2, hoptimus1, provgigapath) is loaded "
            "via THUNDER's model registry (thunder.models.pretrained_models."
            "get_model_from_name), the same path scripts/training/train_tile_eaf.py "
            "and distill_tile_encoder.py use for --model-name."
        ),
    )
    p.add_argument(
        "--early-layer", type=int, default=2,
        help="0-based transformer block index; value = that block's OUTPUT (post-residual). See TileCacheSpec.early_layer_semantics.",
    )
    p.add_argument(
        "--pruned-adapter-ckpt", type=Path, default=None,
        help=(
            "Build a pruned-input cache instead: --encoder is ignored and tile_embeddings "
            "come from this tile-EAF Stage-2 checkpoint's forecaster-guided-pruned + LoRA "
            "forward pass (distill_tile_encoder.py's best_<run>.pt) -- "
            "see PrunedLoRATileTeacherAdapter. --thunder-model-name/--prune-layer/"
            "--keep-ratio/--forecaster-ckpt/--pruned-lora-* below all auto-resolve from "
            "this checkpoint's own recorded config when omitted."
        ),
    )
    p.add_argument(
        "--thunder-model-name", default=None,
        help="THUNDER --model-name backing --pruned-adapter-ckpt (e.g. 'titan' for CONCH v1.5, "
        "'uni2h', ...). Auto-resolved from the checkpoint's config.model_name if omitted.",
    )
    p.add_argument("--forecaster-ckpt", type=Path, default=None, help="Overrides the forecaster_checkpoint recorded in --pruned-adapter-ckpt")
    p.add_argument("--prune-layer", type=int, default=None, help="Overrides the prune_layer recorded in --pruned-adapter-ckpt's config")
    p.add_argument("--keep-ratio", type=float, default=None, help="Overrides the keep_ratio recorded in --pruned-adapter-ckpt's config")
    p.add_argument("--pruned-lora-r", type=int, default=None, help="Overrides lora_r recorded in --pruned-adapter-ckpt's config (default 8)")
    p.add_argument("--pruned-lora-alpha", type=int, default=None, help="Overrides lora_alpha recorded in --pruned-adapter-ckpt's config (default 32)")
    p.add_argument("--pruned-lora-dropout", type=float, default=None, help="Overrides lora_dropout recorded in --pruned-adapter-ckpt's config (default 0.05)")
    p.add_argument("--input-mag", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--input-mpp", type=float, default=0.5)
    p.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--dataset", default="unknown", help="Corpus name recorded in cache metadata, e.g. histai")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", default="64", help="Integer, or 'auto' to auto-probe --batch-size-candidates")
    p.add_argument("--batch-size-candidates", type=int, nargs="+", default=[32, 64, 96])
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument(
        "--slide-loader-chunk-size", type=int, default=32,
        help="WSIs sharing one DataLoader worker pool; 1 restores per-slide loaders",
    )
    p.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse one worker pool across each WSI chunk",
    )
    p.add_argument(
        "--openslide-cache-mib", type=int, default=0,
        help="Decoded OpenSlide tile-cache capacity per DataLoader worker (0 uses the library default)",
    )
    p.add_argument("--autotune", action="store_true", help="Benchmark real WSI reads without publishing cache before extraction")
    p.add_argument("--worker-candidates", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--prefetch-candidates", type=int, nargs="+", default=[2, 4])
    p.add_argument("--benchmark-batches", type=int, default=4)
    p.add_argument("--profile-json", type=Path, help="Write per-slide phase timings and autotune results")
    p.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if a valid, matching cache already exists")
    p.add_argument("--max-slides", type=int, help="Test/smoke-run limit; omit to process every matched slide")
    p.set_defaults(func=cmd_cache_tile)

    archive = sub.add_parser("archive", help="Operate on cold tissue-pixel archives")
    archive_sub = archive.add_subparsers(dest="archive_command", required=True)
    p = archive_sub.add_parser("verify")
    p.add_argument("path", type=Path)
    p.set_defaults(func=cmd_verify_archive)

    p = archive_sub.add_parser(
        "lifecycle",
        help=(
            "Read-only: classify a slide's RAW..RAW_RELEASABLE stage and report "
            "release preconditions. Never deletes anything."
        ),
    )
    p.add_argument("--slide-id", required=True)
    p.add_argument("--raw-path", type=Path)
    p.add_argument("--coords-path", type=Path)
    p.add_argument("--tile-cache", type=Path, action="append")
    p.add_argument("--wsi-cache", type=Path, action="append")
    p.add_argument("--archive-path", type=Path)
    p.add_argument(
        "--remote-recoverable",
        action="store_true",
        help="Mark the source as safely remotely recoverable (e.g. public GTEx/IDC).",
    )
    p.set_defaults(func=cmd_lifecycle)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
