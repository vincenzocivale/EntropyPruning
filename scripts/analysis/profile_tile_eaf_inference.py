#!/usr/bin/env python
"""Phase-separated latency/memory profile of one CONCH v1.5 Tile-EAF run.

Compares the frozen full-backbone forward pass against the forecaster-pruned
LoRA forward pass on real WSI tile batches from a validated manifest, per the
E06 protocol (docs/experimental_protocols.md): warm-up excluded, CUDA
synchronized around every phase, median/p95 over repeated batches.

This does not replace the full E06 sweep (organ/preparation-stratified
sample, raw-WSI-to-prediction, cold-cache timing). It answers the narrower
question required before touching a completed run (docs/continuity.md step
2): is the pruned forward actually cheaper than the full forward, and does
the data-loading phase dominate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thunder.models.pretrained_models import get_model_from_name
from src.data.wsi_tile_stream import build_online_tile_loaders, load_wsi_manifest
from src.models import AttentionForecaster, ThunderBackboneAdapter
from src.models.online_tile_eaf import PrunedLoRAEncoder, unwrap_checkpoint_state
from src.utils import set_seed, tile_encoder_dir_name
from src.wsi_pipeline.experiment_results import result_root


def _autocast(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _summarize(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "n": len(ordered),
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5, cwd=repo_root,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, check=True, timeout=5, cwd=repo_root,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"code_revision": revision, "code_dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-separated full vs tile_eaf latency/memory profile"
    )
    parser.add_argument("--model-name", default="titan", help="THUNDER model name (CONCH v1.5 == titan)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--forecaster-ckpt", required=True)
    parser.add_argument(
        "--lora-checkpoint", default=None,
        help=(
            "Optional trained adapter from finetune_wsi_tile_encoder_pruned_online.py "
            "(e.g. best_conch_v15_src00_pruned10pct_adapter.pt). Latency/memory are "
            "materially unaffected by whether the LoRA weights are trained; omitted "
            "means the profile runs the same PrunedLoRAEncoder architecture with "
            "randomly-initialized LoRA deltas."
        ),
    )
    parser.add_argument("--prune-layer", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--split-column", default="split")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--slide-group", nargs="+", default=["diagnostic"])
    parser.add_argument("--exclude-cohort", nargs="*", default=["HISTAI-mixed", "HISTAI-skin-b2"])
    parser.add_argument("--default-patch-size", type=int, default=512)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--slides-per-batch", type=int, default=16)
    parser.add_argument("--profile-slides", type=int, default=32, help="WSI drawn for the profiling sample")
    parser.add_argument("--tiles-per-slide", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--slide-cache-size", type=int, default=20)
    parser.add_argument("--openslide-cache-mib", type=int, default=256)
    parser.add_argument("--cohort-balance-power", type=float, default=0.5)

    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--measured-batches", type=int, default=10, help="E06 default: 10 repetitions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output", default=None, help="Override the summary JSON path")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This profile requires a CUDA device for CUDA-synchronized timings")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        args.amp_dtype = "fp16"

    backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    backbone = backbone.to(device)
    adapter = ThunderBackboneAdapter(backbone, transform=transform)
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim, hidden=args.hidden, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    forecaster_payload = torch.load(args.forecaster_ckpt, map_location="cpu")
    forecaster.load_state_dict(unwrap_checkpoint_state(forecaster_payload), strict=True)

    student = PrunedLoRAEncoder(
        backbone, adapter, forecaster,
        prune_layer=args.prune_layer, keep_ratio=args.keep_ratio,
    ).to(device)
    student.eval()
    if args.lora_checkpoint:
        payload = torch.load(args.lora_checkpoint, map_location="cpu")
        student.load_trainable_state_dict(unwrap_checkpoint_state(payload))

    split_records = load_wsi_manifest(
        args.manifest, args.data_root, split_column=args.split_column,
        val_fraction=args.val_fraction, split_seed=args.split_seed,
        include_slide_groups=args.slide_group, exclude_cohorts=args.exclude_cohort,
        default_patch_size=args.default_patch_size,
    )
    _, val_loader, _, val_sampler = build_online_tile_loaders(
        split_records, transform,
        batch_size=args.batch_size, slides_per_batch=args.slides_per_batch,
        train_slides_per_epoch=args.slides_per_batch, train_tiles_per_slide=1,
        val_slides_per_epoch=args.profile_slides, val_tiles_per_slide=args.tiles_per_slide,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        slide_cache_size=args.slide_cache_size,
        openslide_cache_bytes=args.openslide_cache_mib * 2**20,
        cohort_balance_power=args.cohort_balance_power, seed=args.seed,
    )
    val_sampler.set_epoch(0)

    phases: dict[str, list[float]] = {"data_load_ms": [], "full_forward_ms": [], "pruned_forward_ms": []}
    tiles_per_batch: list[int] = []
    total_batches = args.warmup_batches + args.measured_batches
    iterator = iter(val_loader)
    torch.cuda.reset_peak_memory_stats(device)
    full_peak_mib = 0.0
    pruned_peak_mib = 0.0

    for batch_index in range(total_batches):
        measure = batch_index >= args.warmup_batches
        t0 = time.perf_counter()
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            images, _ = next(iterator)
        images = images.to(device, non_blocking=True)
        _sync(device)
        data_load_ms = (time.perf_counter() - t0) * 1000.0

        with torch.no_grad(), _autocast(device, args.amp_dtype):
            _sync(device)
            t0 = time.perf_counter()
            _ = student.full_teacher_embedding(images)
            _sync(device)
            full_forward_ms = (time.perf_counter() - t0) * 1000.0
            if measure:
                full_peak_mib = max(full_peak_mib, torch.cuda.max_memory_allocated(device) / 2**20)
            torch.cuda.reset_peak_memory_stats(device)

            _sync(device)
            t0 = time.perf_counter()
            _ = student(images)
            _sync(device)
            pruned_forward_ms = (time.perf_counter() - t0) * 1000.0
            if measure:
                pruned_peak_mib = max(pruned_peak_mib, torch.cuda.max_memory_allocated(device) / 2**20)
            torch.cuda.reset_peak_memory_stats(device)

        if measure:
            phases["data_load_ms"].append(data_load_ms)
            phases["full_forward_ms"].append(full_forward_ms)
            phases["pruned_forward_ms"].append(pruned_forward_ms)
            tiles_per_batch.append(int(images.shape[0]))

    per_phase = {name: _summarize(samples) for name, samples in phases.items()}
    mean_tiles = statistics.mean(tiles_per_batch)
    speedup_median = per_phase["full_forward_ms"]["median_ms"] / max(
        per_phase["pruned_forward_ms"]["median_ms"], 1e-9
    )

    run_name = args.run_name or (
        f"{tile_encoder_dir_name(args.model_name)}_src{args.prune_layer:02d}"
        f"_keep{int(round(args.keep_ratio * 100))}pct_profile"
    )
    repo_root = Path(__file__).resolve().parents[2]
    summary = {
        "schema": "eaf.profile.v1",
        "run_name": run_name,
        "model_name": args.model_name,
        "prune_layer": args.prune_layer,
        "keep_ratio": args.keep_ratio,
        "lora_checkpoint": args.lora_checkpoint,
        "forecaster_checkpoint": args.forecaster_ckpt,
        "manifest": args.manifest,
        "warmup_batches_excluded": args.warmup_batches,
        "measured_batches": args.measured_batches,
        "mean_tiles_per_batch": mean_tiles,
        "phases_ms": per_phase,
        "phases_ms_per_tile": {
            name: {key: value / mean_tiles for key, value in stats.items() if key != "n"}
            for name, stats in per_phase.items()
        },
        "peak_memory_mib": {"full_forward": full_peak_mib, "pruned_forward": pruned_peak_mib},
        "pruned_vs_full_forward_speedup_median": speedup_median,
        **_git_state(repo_root),
    }

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        root = result_root() or (Path(args.data_root).expanduser().resolve() / "results")
        output_path = root / "profiling" / run_name / "profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Profile written to {output_path}")
    print(f"Mean tiles/batch: {mean_tiles:.1f}")
    for name, stats in per_phase.items():
        print(f"  {name:>18s}: median={stats['median_ms']:.2f}ms p95={stats['p95_ms']:.2f}ms")
    print(f"  pruned/full forward median speedup: {speedup_median:.2f}x")


if __name__ == "__main__":
    main()
