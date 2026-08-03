#!/usr/bin/env python3
import argparse
import csv
import hashlib
import time
from pathlib import Path

import requests


def read_manifest(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise SystemExit(
                f"Manifest {path} must contain an 'id' column. "
                f"Found: {reader.fieldnames}"
            )
        return list(reader)


def md5sum(path: Path, chunk_size: int = 1024 * 1024 * 16) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(row, out_dir: Path, retries: int, verify_md5: bool):
    file_id = row["id"]
    filename = row.get("filename") or row.get("file_name") or f"{file_id}.svs"
    expected_md5 = row.get("md5") or row.get("md5sum") or ""
    expected_size = row.get("size") or row.get("file_size") or ""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    part_path = out_dir / f"{filename}.part"

    if out_path.exists() and out_path.stat().st_size > 0:
        if expected_size and out_path.stat().st_size != int(expected_size):
            print(f"[redo] size mismatch for {out_path.name}")
        elif verify_md5 and expected_md5:
            observed = md5sum(out_path)
            if observed.lower() == expected_md5.lower():
                print(f"[skip] {out_path.name} md5 OK")
                return
            print(f"[redo] md5 mismatch for {out_path.name}: {observed} != {expected_md5}")
        else:
            print(f"[skip] {out_path.name}")
            return

    url = f"https://api.gdc.cancer.gov/data/{file_id}"

    for attempt in range(1, retries + 1):
        try:
            print(f"[download] {filename} ({file_id}) attempt {attempt}/{retries}")
            with requests.get(url, stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                with part_path.open("wb") as f:
                    downloaded = 0
                    last_print = time.time()
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 4):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_print > 30:
                            print(f"  {filename}: {downloaded / 1e9:.2f} GB")
                            last_print = now

            part_path.replace(out_path)

            if expected_size and out_path.stat().st_size != int(expected_size):
                raise RuntimeError(
                    f"Downloaded size mismatch for {filename}: "
                    f"{out_path.stat().st_size} != {expected_size}"
                )

            if verify_md5 and expected_md5:
                observed = md5sum(out_path)
                if observed.lower() != expected_md5.lower():
                    raise RuntimeError(
                        f"MD5 mismatch for {filename}: {observed} != {expected_md5}"
                    )

            print(f"[done] {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
            return

        except Exception as e:
            print(f"[error] {filename}: {e}")
            if attempt == retries:
                raise
            sleep_s = 10 * attempt
            print(f"[retry] sleeping {sleep_s}s")
            time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--verify-md5", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = Path(args.output_dir)
    total_size = 0
    for row in rows:
        size = row.get("size") or row.get("file_size") or 0
        try:
            total_size += int(size)
        except Exception:
            pass

    print(f"Files to download: {len(rows)}")
    if total_size:
        print(f"Expected total size: {total_size / 1e9:.2f} GB")
    print(f"Output dir: {out_dir}")

    for i, row in enumerate(rows, start=1):
        print(f"\n[{i}/{len(rows)}]")
        download_file(row, out_dir, retries=args.retries, verify_md5=args.verify_md5)


if __name__ == "__main__":
    main()
