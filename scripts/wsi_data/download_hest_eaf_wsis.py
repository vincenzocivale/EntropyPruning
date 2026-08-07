#!/usr/bin/env python3
"""Download only HEST pyramidal WSI files selected by the EAF inventory."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_WSI_ROOT = Path("/data2/home/vcivale/projects/imaging/data/WSI")
VALID_EXTENSIONS = {".tif", ".tiff", ".btf", ".bigtiff"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_ids(path: Path) -> list[str]:
    return [line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_wsi(wsi_dir: Path, sample_id: str) -> list[Path]:
    return sorted(
        path
        for path in wsi_dir.glob(f"{sample_id}.*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-root", type=Path, default=DEFAULT_WSI_ROOT)
    parser.add_argument("--dataset-id", default="hest_eaf_thunder_clean_v1")
    parser.add_argument("--release", default="v1.3.0")
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Optional text file containing the exact HEST sample IDs to download.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    dataset_root = args.wsi_root / "datasets/pretraining" / args.dataset_id
    id_path = (
        args.ids_file
        if args.ids_file is not None
        else dataset_root / "manifests/download_ids.txt"
    )
    source_root = args.wsi_root / "sources/huggingface/hest" / args.release
    wsi_dir = source_root / "wsis"
    journal = args.wsi_root / "logs/hest_eaf_download/download_journal.jsonl"

    ids = read_ids(id_path)
    if not ids:
        raise RuntimeError(f"No sample IDs found in {id_path}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    batches = [ids[i : i + args.batch_size] for i in range(0, len(ids), args.batch_size)]
    selected = batches[args.start_batch : args.end_batch]
    print("=== HEST WSI-ONLY DOWNLOAD ===")
    print(f"Eligible IDs:       {len(ids):,}")
    print(f"Total batches:      {len(batches):,}")
    print(f"Selected batches:   {len(selected):,}")
    print(f"Destination:        {source_root}")

    for relative_index, batch_ids in enumerate(selected, start=args.start_batch):
        complete: list[str] = []
        pending: list[str] = []
        for sample_id in batch_ids:
            matches = find_wsi(wsi_dir, sample_id)
            if len(matches) == 1 and matches[0].stat().st_size > 0:
                complete.append(sample_id)
            elif len(matches) > 1:
                raise RuntimeError(f"Multiple WSI files found for {sample_id}: {matches}")
            else:
                pending.append(sample_id)

        print(f"\n=== BATCH {relative_index:03d} ===")
        print(f"Already complete: {len(complete):,}")
        print(f"Pending:          {len(pending):,}")
        if not pending:
            continue

        patterns = [f"wsis/{sample_id}[_.]**" for sample_id in pending]
        append_jsonl(
            journal,
            {"timestamp_utc": utc_now(), "status": "batch_started", "batch": relative_index, "ids": pending},
        )
        snapshot_download(
            repo_id="MahmoodLab/hest",
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=source_root,
            max_workers=args.max_workers,
        )

        missing: list[str] = []
        for sample_id in pending:
            matches = find_wsi(wsi_dir, sample_id)
            if len(matches) != 1 or matches[0].stat().st_size == 0:
                missing.append(sample_id)
        if missing:
            append_jsonl(
                journal,
                {"timestamp_utc": utc_now(), "status": "batch_incomplete", "batch": relative_index, "missing": missing},
            )
            raise RuntimeError(f"Batch {relative_index:03d} incomplete; missing={missing[:20]}")

        append_jsonl(
            journal,
            {"timestamp_utc": utc_now(), "status": "batch_complete", "batch": relative_index, "files": len(batch_ids)},
        )
        print("Batch complete.")

    print("\nDownload pass completed. Re-running the same command is safe and resumable.")


if __name__ == "__main__":
    main()
