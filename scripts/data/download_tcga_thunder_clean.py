from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


WSI_ROOT = Path("/data2/home/vcivale/projects/imaging/data/WSI")

DATASET_ROOT = (
    WSI_ROOT
    / "datasets/pretraining/tcga_eaf_thunder_clean_v1"
)

MANIFEST_ROOT = DATASET_ROOT / "manifests"
BATCH_ROOT = MANIFEST_ROOT / "gdc_batches"

INVENTORY = MANIFEST_ROOT / "eligible_download_missing.csv"

STAGING_ROOT = (
    WSI_ROOT
    / "staging/gdc/tcga_eaf_thunder_clean_v1"
)

LOG_ROOT = WSI_ROOT / "logs/gdc_tcga_thunder_clean"
JOURNAL = LOG_ROOT / "download_journal.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_journal(entry: dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def md5sum(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(32 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_inventory() -> dict[str, dict[str, str]]:
    with INVENTORY.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        rows = list(csv.DictReader(handle))

    by_id = {}

    for row in rows:
        file_id = row["file_id"].strip()

        if file_id in by_id:
            raise RuntimeError(f"file_id duplicato: {file_id}")

        by_id[file_id] = row

    return by_id


def load_batch(path: Path) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def final_destination(row: dict[str, str]) -> Path:
    destination = row["raw_destination"].strip()

    if not destination:
        raise RuntimeError(
            f"raw_destination assente per {row['file_id']}"
        )

    return WSI_ROOT / destination


def validate_file(
    path: Path,
    expected_size: int,
    expected_md5: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_size = path.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            f"Dimensione errata per {path}: "
            f"{actual_size} != {expected_size}"
        )

    actual_md5 = md5sum(path)

    if actual_md5.lower() != expected_md5.lower():
        raise RuntimeError(
            f"MD5 errato per {path}: "
            f"{actual_md5} != {expected_md5}"
        )


def write_pending_manifest(
    batch_path: Path,
    rows: list[dict[str, str]],
) -> Path:
    pending_path = (
        STAGING_ROOT
        / "pending_manifests"
        / batch_path.name
    )
    pending_path.parent.mkdir(parents=True, exist_ok=True)

    temporary = pending_path.with_suffix(
        pending_path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "filename",
                "md5",
                "size",
                "state",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, pending_path)
    return pending_path


def remove_empty_uuid_dir(path: Path) -> None:
    if not path.is_dir():
        return

    # Rimuove la directory soltanto dopo che il file è stato
    # verificato e trasferito nella destinazione canonica.
    shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gdc-client",
        default="gdc-client",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end-batch",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.processes < 1:
        raise ValueError("--processes deve essere >= 1")

    inventory = load_inventory()
    batch_paths = sorted(
        BATCH_ROOT.glob("gdc_clean_batch_*.tsv")
    )

    if len(batch_paths) != 16:
        raise RuntimeError(
            f"Attesi 16 batch, trovati {len(batch_paths)}"
        )

    selected_batches = batch_paths[
        args.start_batch:args.end_batch
    ]

    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    total_reused = 0
    total_downloaded = 0

    print("=== TCGA THUNDER-CLEAN DOWNLOAD ===")
    print("Batch selezionati:", len(selected_batches))
    print("Staging:", STAGING_ROOT)
    print("Processi GDC:", args.processes)

    for batch_index, batch_path in enumerate(
        selected_batches,
        start=args.start_batch,
    ):
        batch_rows = load_batch(batch_path)
        pending_rows = []

        print()
        print(
            f"=== BATCH {batch_index:03d} "
            f"({len(batch_rows)} file) ==="
        )

        for manifest_row in batch_rows:
            file_id = manifest_row["id"].strip()
            inventory_row = inventory.get(file_id)

            if inventory_row is None:
                raise RuntimeError(
                    f"File non presente nell'inventario: {file_id}"
                )

            destination = final_destination(inventory_row)
            expected_size = int(inventory_row["file_size"])
            expected_md5 = inventory_row["md5sum"]

            if destination.is_file():
                validate_file(
                    destination,
                    expected_size,
                    expected_md5,
                )

                total_reused += 1

                append_journal(
                    {
                        "timestamp_utc": utc_now(),
                        "status": "already_complete",
                        "batch": batch_index,
                        "file_id": file_id,
                        "destination": str(destination),
                    }
                )
            else:
                pending_rows.append(manifest_row)

        print("Già complete:", len(batch_rows) - len(pending_rows))
        print("Da scaricare:", len(pending_rows))

        if pending_rows:
            pending_manifest = write_pending_manifest(
                batch_path,
                pending_rows,
            )

            batch_log = (
                LOG_ROOT
                / f"gdc_clean_batch_{batch_index:03d}.client.log"
            )

            command = [
                args.gdc_client,
                "download",
                "-m",
                str(pending_manifest),
                "-d",
                str(STAGING_ROOT),
                "-n",
                str(args.processes),
                "--log-file",
                str(batch_log),
            ]

            append_journal(
                {
                    "timestamp_utc": utc_now(),
                    "status": "batch_started",
                    "batch": batch_index,
                    "pending_files": len(pending_rows),
                    "command": command,
                }
            )

            print("Comando:", " ".join(command))

            result = subprocess.run(
                command,
                check=False,
            )

            if result.returncode != 0:
                append_journal(
                    {
                        "timestamp_utc": utc_now(),
                        "status": "batch_download_failed",
                        "batch": batch_index,
                        "returncode": result.returncode,
                    }
                )

                raise SystemExit(
                    f"gdc-client fallito nel batch "
                    f"{batch_index:03d}. "
                    "Rilancia lo stesso comando per riprendere."
                )

        moved_this_batch = 0

        for manifest_row in batch_rows:
            file_id = manifest_row["id"].strip()
            inventory_row = inventory[file_id]

            file_name = inventory_row["file_name"]
            expected_size = int(inventory_row["file_size"])
            expected_md5 = inventory_row["md5sum"]

            destination = final_destination(inventory_row)

            if destination.is_file():
                continue

            uuid_dir = STAGING_ROOT / file_id
            source = uuid_dir / file_name

            validate_file(
                source,
                expected_size,
                expected_md5,
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if source.stat().st_dev != destination.parent.stat().st_dev:
                raise RuntimeError(
                    "Staging e destinazione sono su filesystem "
                    f"differenti: {source} -> {destination}"
                )

            os.replace(source, destination)

            validate_file(
                destination,
                expected_size,
                expected_md5,
            )

            append_journal(
                {
                    "timestamp_utc": utc_now(),
                    "status": "downloaded_and_relocated",
                    "batch": batch_index,
                    "file_id": file_id,
                    "file_name": file_name,
                    "destination": str(destination),
                    "size_bytes": expected_size,
                    "md5": expected_md5,
                }
            )

            remove_empty_uuid_dir(uuid_dir)

            moved_this_batch += 1
            total_downloaded += 1

        append_journal(
            {
                "timestamp_utc": utc_now(),
                "status": "batch_complete",
                "batch": batch_index,
                "files": len(batch_rows),
                "moved": moved_this_batch,
            }
        )

        print("Batch completato.")
        print("File spostati:", moved_this_batch)

    print()
    print("=== DOWNLOAD COMPLETATO ===")
    print("Già presenti/verificati:", total_reused)
    print("Scaricati in questa esecuzione:", total_downloaded)
    print("Journal:", JOURNAL)


if __name__ == "__main__":
    main()
