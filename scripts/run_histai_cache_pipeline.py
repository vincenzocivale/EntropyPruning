#!/usr/bin/env python3
"""
Sequential HISTAI downloader + single-GPU preprocessing/cache producer.

Design
------
* Download producer: downloads one HISTAI subset at a time, then immediately
  advances to the next subset.
* Completion monitor: also notices subsets being downloaded by other processes
  (e.g. HISTAI-skin-b2 already in progress; HISTAI-breast is excluded, see WATCH_ONLY).
* GPU consumer: exactly ONE subset at a time on the A100:
      unique raw-flat view + HISTAI MPP override
      -> TRIDENT segmentation
      -> TRIDENT 20x/512/stride512 coordinates
      -> `scripts/eaf.py cache tile` (CONCH v1.5 offline tile cache)
* `eaf.py cache tile` saves, for EVERY coordinate/tile (no sampling, no
  --max-slides): the final CLS-to-patch attention target used by Tile-EAF
  (head-mean, L1-renormalized), and the final tile embedding shared with
  WSI-EAF -- both in one frozen forward pass per batch. The layer-2 (0-based,
  post-residual block output) early representation is deliberately NOT
  written to disk (it would be >99.8% of cache bytes at corpus scale) --
  EAF Tile training recomputes it online via a cheap early-exit partial
  forward, see ``src/wsi_pipeline/compact_cache_dataset.py``. See
  ``src/wsi_pipeline/tile_cache_pipeline.py`` and
  ``src/wsi_pipeline/model_adapters.py::HookedViTTileTeacherAdapter`` for the
  exact semantics, and ``docs/offline_eaf_pipeline.md`` for the cache schema.
  Every completed cache file is validated with
  ``src.wsi_pipeline.cache_io.validate_cache`` before the subset is marked done.

Environment
-----------
EAF_WSI_ROOT
    default: /data2/home/vcivale/projects/imaging/data/WSI
EAF_REPO
    default: current working directory
TRIDENT_REPO
    default: /data2/home/vcivale/projects/imaging/tools/TRIDENT-v0.3.0
HISTAI_DOWNLOAD_WORKERS
    default: 4
CONCH_BATCH_SIZE
    default: 128
CONCH_NUM_WORKERS
    default: 4
CONCH_OPENSLIDE_CACHE_MIB
    default: 512 per DataLoader worker; 0 uses the OpenSlide library default
CONCH_COMPRESSION
    default: lzf; set to none after benchmarking storage/throughput on the target disk
CONCH_AUTOTUNE
    default: 0; set to 1 to benchmark real WSI reads before each subset cache run
CONCH_SLIDE_LOADER_CHUNK_SIZE
    default: 32; WSIs that share one persistent DataLoader worker pool
GPU_ID
    default: 0
HISTAI_POLL_SECONDS
    default: 60

Usage
-----
  export EAF_WSI_ROOT=/data2/home/vcivale/projects/imaging/data/WSI
  export EAF_REPO=/data2/home/vcivale/projects/imaging/EAF
  export TRIDENT_REPO=/data2/home/vcivale/projects/imaging/tools/TRIDENT-v0.3.0

  python -u scripts/run_histai_cache_pipeline.py --preflight

  nohup python -u scripts/run_histai_cache_pipeline.py \
    > "$EAF_WSI_ROOT/logs/histai_pipeline_master.log" 2>&1 &

The default sequential download order includes only the newly-approved subsets.
Existing skin-b2/hematologic downloads are MONITORED but not started again by this
process. HISTAI-breast is excluded entirely (manual run in a separate terminal).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make `from src...` imports (run_conch_cache's post-write validate_cache check) work
# regardless of cwd/PYTHONPATH at invocation time -- e.g. `python scripts/run_histai_
# cache_pipeline.py` sets sys.path[0] to this file's own directory (scripts/), NOT the
# repo root, so `import src` is NOT reliably available without this. Same pattern
# scripts/eaf.py already uses. A prior run appeared to work only because whatever shell
# launched it happened to have the repo root on PYTHONPATH already; don't depend on that.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd


ROOT = Path(
    os.environ.get(
        "EAF_WSI_ROOT",
        "/data2/home/vcivale/projects/imaging/data/WSI",
    )
).expanduser().resolve()

REPO = Path(
    os.environ.get("EAF_REPO", os.getcwd())
).expanduser().resolve()

TRIDENT = Path(
    os.environ.get(
        "TRIDENT_REPO",
        "/data2/home/vcivale/projects/imaging/tools/TRIDENT-v0.3.0",
    )
).expanduser().resolve()

PLAN = (
    ROOT
    / "datasets"
    / "pretraining"
    / "histai_eaf_wsi_v1"
    / "manifests"
    / "plan.csv"
)

DATASET_ROOT = (
    ROOT
    / "datasets"
    / "pretraining"
    / "histai_eaf_wsi_v1"
)

RAW_ROOT = ROOT / "sources" / "histai"
STATE_ROOT = ROOT / "catalog" / "histai_pipeline"
LOG_ROOT = ROOT / "logs" / "histai_pipeline"

DOWNLOAD_WORKERS = int(os.environ.get("HISTAI_DOWNLOAD_WORKERS", "4"))
CONCH_BATCH_SIZE = int(os.environ.get("CONCH_BATCH_SIZE", "128"))
CONCH_NUM_WORKERS = int(os.environ.get("CONCH_NUM_WORKERS", "4"))
CONCH_OPENSLIDE_CACHE_MIB = int(os.environ.get("CONCH_OPENSLIDE_CACHE_MIB", "512"))
CONCH_COMPRESSION = os.environ.get("CONCH_COMPRESSION", "lzf")
CONCH_AUTOTUNE = os.environ.get("CONCH_AUTOTUNE", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
CONCH_SLIDE_LOADER_CHUNK_SIZE = int(
    os.environ.get("CONCH_SLIDE_LOADER_CHUNK_SIZE", "32")
)
# Post-run integrity re-check (see run_conch_cache) opens every cache HDF5's
# metadata a second time -- an I/O-bound, per-file-independent operation that
# releases the GIL during h5py reads, so a thread pool gives real wall-clock
# speedup at corpus scale without changing behavior.
CONCH_VALIDATE_WORKERS = int(os.environ.get("CONCH_VALIDATE_WORKERS", "8"))
# TRIDENT's own per-WSI segmentation loop is strictly sequential (one slide at a time)
# and was measured GPU-compute-bound only in bursts -- CPU (320 cores, <35% loadavg even
# at 5-way parallel) and VRAM (~6.8 GiB/process on a 40 GiB A100) both have large headroom.
# Running SEG_PARALLEL_WORKERS separate `--task seg` processes on disjoint slide chunks
# measured ~3.2x combined throughput at 5 processes (1.6 -> 5.2 slides/min); 4 is the
# default here to leave a VRAM safety margin against an unusually large/OOM-prone slide.
SEG_PARALLEL_WORKERS = int(os.environ.get("SEG_PARALLEL_WORKERS", "4"))
# Caps TRIDENT's own per-slide DataLoader worker pool (trident/IO.py:get_num_workers
# defaults to min(0.75 * cpu_count, 2 * batch_size) = up to 128 processes, respawned
# for EVERY slide). Left unset, TRIDENT's auto default applies. Set this to keep our
# footprint contained on a host shared with other users.
SEG_MAX_WORKERS = os.environ.get("SEG_MAX_WORKERS")
GPU_ID = os.environ.get("GPU_ID", "0")
POLL_SECONDS = int(os.environ.get("HISTAI_POLL_SECONDS", "60"))

# Newly-approved subsets: download them sequentially, small -> large.
DOWNLOAD_ORDER = [
    "HISTAI-colorectal-b2",
    "HISTAI-gastrointestinal",
    "HISTAI-thorax",
    "HISTAI-colorectal-b1",
    "HISTAI-skin-b1",
    "HISTAI-mixed",
]

# These may already be downloaded / downloading in separate jobs.
#
# HISTAI-breast is intentionally NOT here: its TRIDENT segmentation is being run
# manually by the operator in a separate terminal against the same job_dir
# (processed/trident/HISTAI-breast). This orchestrator must never also touch it --
# concurrent `run_batch_of_slides.py --task seg` runs against the same job_dir would
# race the same lock/output files. Re-add it only once that manual run is done and the
# operator asks for HISTAI-breast to be picked up here too.
WATCH_ONLY = [
    "HISTAI-hematologic",
    "HISTAI-skin-b2",
]

ALL_PIPELINE_SUBSETS = WATCH_ONLY + DOWNLOAD_ORDER

# Subsets whose GPU processing (seg/coords/cache) is happening on a separate machine
# (raw WSI transferred out to storage this orchestrator can't reach). Download tracking
# still applies (the raw is already here), but GPU work is skipped entirely regardless
# of queue position -- unlike HISTAI-breast (excluded from WATCH_ONLY/ALL_PIPELINE_SUBSETS
# entirely), these ARE downloaded subsets in DOWNLOAD_ORDER, so exclusion has to happen at
# the enqueue check itself, not by omitting them from a source list.
# HISTAI-skin-b1: operator is running seg on another machine (no shared storage access);
# processed/trident/HISTAI-skin-b1 output will be synced back here for the cache stage.
EXTERNALLY_PROCESSED = {
    "HISTAI-skin-b1",
}

STOP = threading.Event()
GPU_QUEUE: queue.Queue[str | None] = queue.Queue()
QUEUED: set[str] = set()
QUEUED_LOCK = threading.Lock()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def run_logged(cmd: list[str], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"RUN -> {log_path.name}")
    log(" ".join(shlex.quote(x) for x in cmd))
    with log_path.open("a", buffering=1) as fh:
        fh.write(f"\n[{now()}] COMMAND\n")
        fh.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with rc={proc.returncode}: {' '.join(cmd)}; "
            f"see {log_path}"
        )


def chunk_bounds(n_rows: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split ``range(n_rows)`` into up to ``n_chunks`` contiguous, near-equal, disjoint
    (start, end) ranges. Deterministic across reruns of the same subset (same input
    order -> same chunk assignment), which keeps per-worker logs/behavior reproducible
    and makes a resumed run's chunking identical to the interrupted one. Never returns
    more chunks than there are rows (an empty custom_list_of_wsis CSV is not a case
    worth exercising in TRIDENT)."""
    n_chunks = max(1, min(n_chunks, n_rows))
    base, rem = divmod(n_rows, n_chunks)
    bounds = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def run_parallel(jobs: list[tuple[list[str], Path]], cwd: Path) -> None:
    """Launch every (cmd, log_path) in ``jobs`` concurrently, wait for all of them, then
    raise once with every failure listed -- never lets one straggler's traceback hide
    another's. Each job's own log file is independent (one process, one file), so a
    crash in worker 2 never interleaves/corrupts worker 0's output.

    Process-group semantics: each child inherits this (the orchestrator's) process
    group, exactly like the existing single-process run_logged path -- a `kill -TERM --
    -<orchestrator_pgid>` still stops every worker, no separate cleanup needed.
    """
    log(f"RUN(parallel x{len(jobs)}) -> " + ", ".join(p.name for _, p in jobs))
    procs = []
    for cmd, log_path in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log(" ".join(shlex.quote(x) for x in cmd))
        fh = log_path.open("a", buffering=1)
        fh.write(f"\n[{now()}] COMMAND\n")
        fh.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, text=True
        )
        procs.append((proc, fh, log_path, cmd))

    failures = []
    for proc, fh, log_path, cmd in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            failures.append((log_path, rc))

    if failures:
        detail = "; ".join(f"{p} rc={rc}" for p, rc in failures)
        raise RuntimeError(
            f"{len(failures)}/{len(procs)} parallel workers failed: {detail}"
        )


def marker(subset: str, stage: str) -> Path:
    safe = subset.replace("/", "_")
    return STATE_ROOT / f"{safe}.{stage}"


def touch_marker(subset: str, stage: str, text: str = "") -> None:
    p = marker(subset, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{now()}\n{text}\n")


def plan_df() -> pd.DataFrame:
    if not PLAN.is_file():
        raise FileNotFoundError(f"Missing HISTAI frozen plan: {PLAN}")
    df = pd.read_csv(PLAN)
    required = {"subset", "repo_id", "repo_path", "case_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"plan.csv missing columns: {sorted(missing)}")
    return df


PLAN_DF: pd.DataFrame | None = None


def rows_for(subset: str) -> pd.DataFrame:
    assert PLAN_DF is not None
    df = PLAN_DF[PLAN_DF["subset"] == subset].copy()
    if df.empty:
        raise RuntimeError(f"No planned rows for {subset}")
    return df


def raw_path(row) -> Path:
    return RAW_ROOT / str(row.subset) / str(row.repo_path)


def expected_count(subset: str) -> int:
    return len(rows_for(subset))


def present_count(subset: str) -> int:
    return sum(
        raw_path(row).is_file()
        for row in rows_for(subset).itertuples(index=False)
    )


def download_complete(subset: str) -> bool:
    return present_count(subset) == expected_count(subset)


def download_subset(subset: str) -> None:
    """Download exactly the frozen one-WSI-per-case rows for one subset."""
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm

    df = rows_for(subset)
    n = len(df)
    log(f"DOWNLOAD {subset}: planned={n:,}, workers={DOWNLOAD_WORKERS}")

    def one(row):
        dst_root = RAW_ROOT / row.subset
        dst_root.mkdir(parents=True, exist_ok=True)
        expected = raw_path(row)
        if expected.is_file():
            return "skip"
        hf_hub_download(
            repo_id=row.repo_id,
            repo_type="dataset",
            filename=row.repo_path,
            local_dir=dst_root,
        )
        if not expected.is_file():
            raise RuntimeError(f"Expected file missing after download: {expected}")
        return "download"

    failures = []
    downloaded = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futs = {ex.submit(one, r): r for r in df.itertuples(index=False)}
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"download {subset}",
            unit="WSI",
            dynamic_ncols=True,
        ):
            row = futs[fut]
            try:
                status = fut.result()
                if status == "skip":
                    skipped += 1
                else:
                    downloaded += 1
            except Exception as exc:
                failures.append((row.case_id, row.repo_path, repr(exc)))

    if failures:
        fail_path = LOG_ROOT / f"{subset}.download_failures.tsv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            failures,
            columns=["case_id", "repo_path", "error"],
        ).to_csv(fail_path, sep="\t", index=False)
        raise RuntimeError(
            f"{subset}: {len(failures)} download failures; see {fail_path}"
        )

    current = present_count(subset)
    log(
        f"DOWNLOAD DONE {subset}: present={current:,}/{n:,}, "
        f"new={downloaded:,}, existing={skipped:,}"
    )
    if current != n:
        raise RuntimeError(f"{subset}: incomplete after download {current}/{n}")
    touch_marker(subset, "download.done", f"{current}/{n}")


def detect_mpp(filename: str) -> float:
    """
    HISTAI known metadata bug: embedded MPP may be 1000.
    Dataset convention: default 20x -> 0.5 um/px; explicit 40x -> 0.25.
    """
    name = filename.lower()
    if re.search(r"(^|[_-])(40x|x40)([_-]|$)", name):
        return 0.25
    if re.search(r"(^|[_-])(20x|x20)([_-]|$)", name):
        return 0.50
    return 0.50


def prepare_flat_view(subset: str) -> tuple[Path, Path, pd.DataFrame]:
    """
    Create unique symlinks and custom TRIDENT wsi,mpp CSV.
    HISTAI reuses slide_H&E_0.tiff across many cases, so unique names are mandatory.
    """
    df = rows_for(subset)

    view = DATASET_ROOT / "views" / "raw_flat" / subset
    manifests = DATASET_ROOT / "manifests" / "subsets" / subset
    view.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    trident_csv = manifests / "trident_wsi_mpp.csv"
    enriched = []

    for row in df.itertuples(index=False):
        src = raw_path(row)
        if not src.is_file():
            raise FileNotFoundError(src)

        unique_filename = f"{subset}__{row.case_id}__{src.name}"
        link = view / unique_filename

        if link.is_symlink():
            if link.resolve() != src.resolve():
                raise RuntimeError(f"Symlink collision: {link}")
        elif link.exists():
            raise RuntimeError(f"Non-symlink collision: {link}")
        else:
            link.symlink_to(src.resolve())

        enriched.append(
            {
                "subset": subset,
                "case_id": row.case_id,
                "repo_path": row.repo_path,
                "slide_id": Path(unique_filename).stem,
                "wsi": unique_filename,
                "wsi_path": str(link),
                "mpp": detect_mpp(src.name),
            }
        )

    edf = pd.DataFrame(enriched)
    edf[["wsi", "mpp"]].to_csv(trident_csv, index=False)
    return view, trident_csv, edf


def trident_job(subset: str) -> Path:
    return DATASET_ROOT / "processed" / "trident" / subset


def run_seg_coords(subset: str) -> tuple[Path, pd.DataFrame]:
    view, mpp_csv, edf = prepare_flat_view(subset)
    job = trident_job(subset)
    job.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        str(TRIDENT / "run_batch_of_slides.py"),
        "--wsi_dir", str(view),
        "--job_dir", str(job),
        "--gpus", str(GPU_ID),
    ]

    # Segmentation: SEG_PARALLEL_WORKERS concurrent `--task seg` processes on disjoint,
    # contiguous slide chunks of the SAME job_dir. Safe to run concurrently against one
    # job_dir: TRIDENT itself lock-guards each slide's contour file
    # (create_lock/is_locked in Processor.run_segmentation_job), so even if chunk
    # boundaries were ever wrong, the worst case is a benign "locked, skipping" log
    # line, never corruption or duplicate work. Measured ~3.2x combined throughput at
    # 5 workers (1.6 -> 5.2 slides/min) on this A100; TRIDENT's per-WSI loop itself
    # stays sequential *within* each worker, so this is the only real lever available.
    mpp_df = pd.read_csv(mpp_csv)
    chunk_dir = DATASET_ROOT / "manifests" / "subsets" / subset / "seg_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    seg_jobs = []
    for i, (start, end) in enumerate(chunk_bounds(len(mpp_df), SEG_PARALLEL_WORKERS)):
        chunk_csv = chunk_dir / f"chunk_{i:02d}.csv"
        mpp_df.iloc[start:end].to_csv(chunk_csv, index=False)
        seg_cmd = common + [
            "--custom_list_of_wsis", str(chunk_csv),
            "--task", "seg",
            "--segmenter", "hest",
        ]
        if SEG_MAX_WORKERS:
            seg_cmd += ["--max_workers", str(SEG_MAX_WORKERS)]
        seg_jobs.append((seg_cmd, LOG_ROOT / f"{subset}.seg.chunk{i:02d}.log"))
    log(f"SEG {subset}: {len(seg_jobs)} parallel workers, {len(mpp_df)} slides")
    run_parallel(seg_jobs, cwd=TRIDENT)

    # Coords: kept single-process. Cheap (no deep-model inference, just tissue-mask ->
    # patch-grid math over already-computed contours) and never benchmarked for a
    # parallel win -- no reason to add the complexity without measured evidence.
    coords_cmd = common + [
        "--custom_list_of_wsis", str(mpp_csv),
        "--task", "coords",
        "--mag", "20",
        "--patch_size", "512",
        "--overlap", "0",
    ]
    run_logged(coords_cmd, LOG_ROOT / f"{subset}.coords.log", cwd=TRIDENT)

    coords_dir = job / "20x_512px_0px_overlap" / "patches"
    if not coords_dir.is_dir():
        raise RuntimeError(f"TRIDENT coords directory missing: {coords_dir}")

    missing = [
        sid for sid in edf["slide_id"]
        if not (coords_dir / f"{sid}_patches.h5").is_file()
    ]

    # A missing HDF5 is expected (not a failure) when TRIDENT itself legitimately
    # found no tissue to patch -- coords.status == "skipped (empty_geodataframe)"
    # in that slide's wsi_states JSON. Those slides have nothing to cache; drop
    # them from `edf` rather than blocking the whole subset. Anything else missing
    # (no state file, or a status that isn't an empty-tissue skip) still hard-fails.
    states_dir = job / "wsi_states"
    empty_tissue: list[str] = []
    genuinely_missing: list[str] = []
    for sid in missing:
        # A slide can have >1 state file (e.g. re-segmented on a different machine/run,
        # each attempt gets its own hash suffix) -- scan all matches, not just the
        # first one glob happens to return, and accept the reason from either the
        # coords task or the segmentation task (an older file may predate the coords
        # attempt entirely and only have the segmentation-side empty_geodataframe).
        reason = None
        for state_file in states_dir.glob(f"{sid}__*.json"):
            try:
                state = json.loads(state_file.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            tasks = state.get("tasks", {})
            found = (
                tasks.get("coords", {}).get("reason")
                or tasks.get("segmentation", {}).get("reason")
            )
            if found == "empty_geodataframe":
                reason = found
                break
        if reason == "empty_geodataframe":
            empty_tissue.append(sid)
        else:
            genuinely_missing.append(sid)

    if genuinely_missing:
        raise RuntimeError(
            f"{subset}: {len(genuinely_missing)} slides missing coordinate HDF5 "
            f"for no recognized reason; first={genuinely_missing[:5]}"
        )
    if empty_tissue:
        log(f"COORDS {subset}: {len(empty_tissue)} slides skipped (empty tissue), "
            f"excluded from cache: {empty_tissue}")
        edf = edf[~edf["slide_id"].isin(empty_tissue)].reset_index(drop=True)

    touch_marker(subset, "coords.done", f"{len(edf)} slides")
    return coords_dir, edf


def build_extraction_registries(
    subset: str,
    coords_dir: Path,
    edf: pd.DataFrame,
) -> tuple[Path, Path]:
    out = DATASET_ROOT / "manifests" / "subsets" / subset
    out.mkdir(parents=True, exist_ok=True)

    slides_csv = out / "slides_for_tile_cache.csv"
    coords_csv = out / "coords_registry.csv"

    edf[["slide_id", "case_id", "wsi_path"]].to_csv(slides_csv, index=False)

    coords_rows = [
        {"slide_id": sid, "path": str((coords_dir / f"{sid}_patches.h5").resolve())}
        for sid in edf["slide_id"]
    ]
    pd.DataFrame(coords_rows).to_csv(coords_csv, index=False)

    return slides_csv, coords_csv


def _validate_cache_files(
    h5_files: list[Path], *, max_workers: int = CONCH_VALIDATE_WORKERS
) -> list[tuple[str, str]]:
    """Validate every cache file in parallel.

    Order-preserving (``executor.map``), so callers see the same "first N bad
    files" as the serial loop this replaces -- only the wall-clock changes.
    """
    from src.wsi_pipeline.cache_io import validate_cache

    def _validate_one(p: Path) -> tuple[str, str] | None:
        try:
            validate_cache(p, expected_kind="tile_eaf")
        except Exception as exc:
            return (str(p), repr(exc))
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return [row for row in ex.map(_validate_one, h5_files) if row is not None]


def run_conch_cache(
    subset: str,
    coords_dir: Path,
    edf: pd.DataFrame,
) -> None:
    """Build the canonical offline Tile-EAF/WSI-EAF cache via ``scripts/eaf.py cache tile``.

    Every coordinate/tile is processed (no ``--max-slides``, no sampling flag). The
    extractor always produces ``final_attention`` (CLS-to-patch, head-mean,
    L1-renormalized -- the Tile-EAF teacher target) and ``tile_embeddings`` (shared
    with WSI-EAF) in one forward pass per batch. ``early_tokens`` is intentionally
    NOT part of this permanent cache (storage cost; see
    ``docs/offline_eaf_pipeline.md``) -- there is no separate "does this exist" flag to detect
    anymore, so a missing/legacy extractor is now a hard import-time failure, not a
    silent partial cache.
    """
    slides_csv, coords_csv = build_extraction_registries(subset, coords_dir, edf)

    # Path includes cache_id (docs/data_layout.md: caches/tile_eaf/<dataset>/<tile_encoder>/<cache_id>/)
    # so a future early_layer/attention/dtype change can never silently overwrite a
    # cache built under different teacher semantics -- it gets its own directory.
    from src.wsi_pipeline.cache_contracts import TileCacheSpec
    from src.wsi_pipeline.tile_cache_pipeline import CONCH_V15_REVISION

    spec_for_path = TileCacheSpec(
        tile_encoder="conch_v15",
        model_revision=CONCH_V15_REVISION,
        early_layer=2,
        input_mag=20,
        patch_size=512,
        stride=512,
        input_mpp=0.5,
        dtype="float16",
        dataset="histai_eaf_wsi_v1",
    )
    out_dir = (
        ROOT / "caches" / "tile_eaf" / "histai_eaf_wsi_v1" / "conch_v15"
        / spec_for_path.cache_id / subset
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "eaf.py"),
        "cache", "tile",
        "--slides", str(slides_csv),
        "--coords-registry", str(coords_csv),
        "--output-dir", str(out_dir),
        "--encoder", "conch_v15",
        "--early-layer", "2",
        "--input-mag", "20",
        "--patch-size", "512",
        "--stride", "512",
        "--input-mpp", "0.5",
        "--dataset", "histai_eaf_wsi_v1",
        "--device", "cuda",
        "--batch-size", str(CONCH_BATCH_SIZE),
        "--num-workers", str(CONCH_NUM_WORKERS),
        "--openslide-cache-mib", str(CONCH_OPENSLIDE_CACHE_MIB),
        "--slide-loader-chunk-size", str(CONCH_SLIDE_LOADER_CHUNK_SIZE),
        "--storage-dtype", "float16",
        "--compression", CONCH_COMPRESSION,
        "--profile-json", str(LOG_ROOT / f"{subset}.conch_v15_profile.json"),
    ]
    if CONCH_AUTOTUNE:
        cmd.extend(
            [
                "--autotune",
                "--batch-size-candidates", "32", "64", "96",
                "--worker-candidates", "4", "8", "16",
                "--prefetch-candidates", "2", "4",
            ]
        )
    # Intentionally NO --max-slides: every coordinate/tile is processed. Reruns are
    # cheap: eaf.py cache tile skips any slide whose cache is already valid+complete
    # and matches this exact spec (encoder/layer/attention/dtype), and only rebuilds
    # slides that are missing, partial, corrupt, or stale.
    run_logged(cmd, LOG_ROOT / f"{subset}.conch_v15_cache.log", cwd=REPO)

    h5_files = list(out_dir.glob("*.h5"))
    if len(h5_files) < len(edf):
        raise RuntimeError(
            f"{subset}: only {len(h5_files)} cache HDF5 for {len(edf)} WSIs"
        )

    bad = _validate_cache_files(h5_files)
    if bad:
        raise RuntimeError(
            f"{subset}: {len(bad)} cache files fail validate_cache(); first={bad[:3]}"
        )

    touch_marker(
        subset,
        "cache.done",
        f"{len(edf)} slides; batch={CONCH_BATCH_SIZE}",
    )


def process_subset_gpu(subset: str) -> None:
    log(f"GPU PIPELINE START {subset}")
    touch_marker(subset, "gpu.running")

    try:
        coords_dir, edf = run_seg_coords(subset)
        run_conch_cache(subset, coords_dir, edf)
        touch_marker(subset, "pipeline.done", f"{len(edf)} WSI")
        marker(subset, "gpu.running").unlink(missing_ok=True)
        marker(subset, "pipeline.failed").unlink(missing_ok=True)
        log(f"GPU PIPELINE DONE {subset}")
    except Exception as exc:
        marker(subset, "gpu.running").unlink(missing_ok=True)
        touch_marker(
            subset,
            "pipeline.failed",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        log(f"GPU PIPELINE FAILED {subset}: {exc}")
        # Keep worker alive so later subsets can still proceed.


def enqueue_if_ready(subset: str) -> None:
    if subset in EXTERNALLY_PROCESSED:
        return
    if not download_complete(subset):
        return
    if marker(subset, "pipeline.done").exists():
        return
    if marker(subset, "gpu.running").exists():
        return

    with QUEUED_LOCK:
        if subset in QUEUED:
            return
        QUEUED.add(subset)
        GPU_QUEUE.put(subset)
        log(f"ENQUEUED GPU {subset}")


def gpu_worker() -> None:
    while not STOP.is_set():
        item = GPU_QUEUE.get()
        if item is None:
            GPU_QUEUE.task_done()
            return
        subset = item
        try:
            process_subset_gpu(subset)
        finally:
            with QUEUED_LOCK:
                QUEUED.discard(subset)
            GPU_QUEUE.task_done()


def monitor_existing_downloads() -> None:
    while not STOP.is_set():
        for subset in ALL_PIPELINE_SUBSETS:
            try:
                enqueue_if_ready(subset)
            except Exception as exc:
                log(f"MONITOR warning {subset}: {exc}")
        STOP.wait(POLL_SECONDS)


def preflight() -> None:
    log(f"EAF_WSI_ROOT={ROOT}")
    log(f"EAF_REPO={REPO}")
    log(f"TRIDENT_REPO={TRIDENT}")
    log(f"PLAN={PLAN}")

    if not REPO.is_dir():
        raise RuntimeError(f"EAF repo missing: {REPO}")
    if not (TRIDENT / "run_batch_of_slides.py").is_file():
        raise RuntimeError(f"TRIDENT invalid: {TRIDENT}")
    if not PLAN.is_file():
        raise RuntimeError(f"HISTAI plan missing: {PLAN}")

    # HF access/auth smoke.
    from huggingface_hub import HfApi
    api = HfApi()
    for subset in DOWNLOAD_ORDER:
        api.repo_info(f"histai/{subset}", repo_type="dataset")
    log("HF access OK for all sequential-download subsets")

    # CONCH v1.5 access + cache-contract check before committing to a huge GPU run:
    # actually load the encoder used by `eaf.py cache tile` (TITAN's return_conch(),
    # NOT a `timm.create_model("hf_hub:...")` shortcut, which is not how this
    # checkpoint is distributed) and confirm its block/token geometry matches what
    # TileCacheSpec's early_layer=2 / final-attention semantics assume.
    import torch

    from src.wsi_pipeline.tile_cache_pipeline import build_encoder

    adapter = build_encoder("conch_v15", token=None, device=torch.device("cpu"))
    n_blocks = len(adapter._blocks)
    if not (0 <= 2 < n_blocks):
        raise RuntimeError(f"CONCH v1.5 has {n_blocks} blocks; early_layer=2 out of range")
    log(
        f"CONCH v1.5 accessible: n_blocks={n_blocks}, input_size={adapter.input_size}, "
        f"num_prefix_tokens={adapter.num_prefix_tokens}"
    )

    log(
        f"GPU cache: batch_size={CONCH_BATCH_SIZE}, "
        f"num_workers={CONCH_NUM_WORKERS}, gpu={GPU_ID}"
    )
    log("PREFLIGHT OK")


def acquire_master_lock():
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_ROOT / "master.lock"
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"Another HISTAI pipeline orchestrator is already running: {lock_path}"
        )
    fh.write(f"pid={os.getpid()}\nstarted={now()}\n")
    fh.flush()
    return fh


def clear_stale_gpu_running_markers() -> None:
    """Remove leftover ``*.gpu.running`` markers from a previous, now-dead process.

    Called right after acquiring the master lock: at that point ``fcntl.flock`` has
    already proven no other orchestrator instance is alive, so any ``gpu.running``
    marker still on disk cannot possibly reflect real, in-progress work -- it is a
    crash/kill -9 leftover (``process_subset_gpu`` only clears it on a clean
    success/failure exit). Left in place, a stale marker would permanently block that
    subset's ``enqueue_if_ready`` check forever, since nothing else ever removes it.
    """
    for stale in STATE_ROOT.glob("*.gpu.running"):
        subset = stale.name[: -len(".gpu.running")]
        log(f"Clearing stale gpu.running marker from a previous run: {subset}")
        stale.unlink(missing_ok=True)


def main() -> int:
    global PLAN_DF

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="Validate access/paths/cache contract and exit.",
    )
    ap.add_argument(
        "--downloads-only",
        action="store_true",
        help="Do not start the GPU consumer.",
    )
    ap.add_argument(
        "--subset",
        choices=ALL_PIPELINE_SUBSETS,
        help=(
            "Process exactly one already-downloaded subset synchronously "
            "(seg -> coords -> eaf.py cache tile) and exit. No downloading, no "
            "daemon threads, no master lock. Every tissue tile in the subset is "
            "processed; use this for a full-subset run once the smoke test has "
            "validated the pipeline, before committing to the whole corpus."
        ),
    )
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    PLAN_DF = plan_df()

    unknown = [
        s for s in ALL_PIPELINE_SUBSETS
        if s not in set(PLAN_DF["subset"].unique())
    ]
    if unknown:
        raise RuntimeError(f"Subsets missing from plan.csv: {unknown}")

    if args.preflight:
        preflight()
        return 0

    if args.subset:
        if not download_complete(args.subset):
            raise SystemExit(
                f"{args.subset}: not fully downloaded "
                f"({present_count(args.subset)}/{expected_count(args.subset)}); "
                "run `python scripts/eaf.py data download-histai --subset "
                f"{args.subset}` first."
            )
        coords_dir, edf = run_seg_coords(args.subset)
        run_conch_cache(args.subset, coords_dir, edf)
        log(f"SUBSET DONE {args.subset}: {len(edf)} WSI cached")
        return 0

    # Keep the descriptor alive for the lifetime of the process so flock remains held.
    master_lock = acquire_master_lock()
    clear_stale_gpu_running_markers()

    worker = None
    monitor = None

    if not args.downloads_only:
        worker = threading.Thread(
            target=gpu_worker,
            name="histai-gpu-worker",
            daemon=True,
        )
        monitor = threading.Thread(
            target=monitor_existing_downloads,
            name="histai-download-monitor",
            daemon=True,
        )
        worker.start()
        monitor.start()

    try:
        # Catch already-completed externally downloaded datasets immediately.
        if not args.downloads_only:
            for subset in ALL_PIPELINE_SUBSETS:
                enqueue_if_ready(subset)

        # Sequential producer: exactly one newly-approved repository at a time.
        for subset in DOWNLOAD_ORDER:
            if download_complete(subset):
                log(f"DOWNLOAD SKIP complete {subset}")
                touch_marker(
                    subset,
                    "download.done",
                    f"{present_count(subset)}/{expected_count(subset)}",
                )
            else:
                download_subset(subset)

            if not args.downloads_only:
                enqueue_if_ready(subset)

        log("ALL SEQUENTIAL DOWNLOADS FINISHED")

        if not args.downloads_only:
            # Wait for queued GPU work; monitor may enqueue skin-b2 when their
            # external downloads finish. Once sequential downloads are done, keep
            # monitoring until every fully-downloaded subset has been processed.
            while True:
                for subset in ALL_PIPELINE_SUBSETS:
                    enqueue_if_ready(subset)

                fully_downloaded = [
                    s for s in ALL_PIPELINE_SUBSETS if download_complete(s)
                ]
                unfinished = [
                    s for s in fully_downloaded
                    if not marker(s, "pipeline.done").exists()
                    and not marker(s, "pipeline.failed").exists()
                ]

                if not unfinished and GPU_QUEUE.unfinished_tasks == 0:
                    break

                time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        log("Interrupted")
        return 130
    finally:
        STOP.set()
        if worker is not None:
            GPU_QUEUE.put(None)
            worker.join(timeout=10)
        if monitor is not None:
            monitor.join(timeout=10)

    done = [
        s for s in ALL_PIPELINE_SUBSETS
        if marker(s, "pipeline.done").exists()
    ]
    failed = [
        s for s in ALL_PIPELINE_SUBSETS
        if marker(s, "pipeline.failed").exists()
    ]
    log(f"PIPELINE SUMMARY done={done}")
    log(f"PIPELINE SUMMARY failed={failed}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
