"""Rewrite per-dataset attention-cache HDF5 files with row-level chunking.

The caches built by ``extract_features.py`` use ``chunks=(128, n_patches,
embed_dim)`` for the embedding datasets (~51MB/chunk for UNI). Training
reads one row at a time under ``shuffle=True`` (``H5ForecastDataset``), and
h5py's default chunk cache (8MB) is smaller than a single chunk, so every
row access forces a full chunk re-read from disk -- measured ~5x slower
than sequential access, and the dominant cost behind ~1h/epoch training.

Rewriting with chunks=(1, ...) makes each row read fetch only its own
bytes, eliminating that amplification.

Processes one cache file at a time (smallest first), copying split-by-split
into a ``.rechunk_tmp`` sibling file, then atomically replacing the
original via ``os.replace``. This is safe even while another process holds
the original file open for reading: it keeps reading the old (now
unlinked) inode until it reopens the path.

Idempotent: files whose datasets are already row-chunked are skipped, so
the script can be re-run safely after an interruption.
"""

import argparse
import os
import time
from pathlib import Path

import h5py

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def _already_rechunked(path: Path) -> bool:
    with h5py.File(path, "r") as f:
        for split in f.keys():
            for key in f[split].keys():
                ds = f[split][key]
                if ds.chunks is not None and len(ds.shape) > 1 and ds.chunks[0] != 1:
                    return False
    return True


def rechunk_file(src_path: Path, block_rows: int):
    tmp_path = src_path.with_suffix(src_path.suffix + ".rechunk_tmp")
    t0 = time.time()
    with h5py.File(src_path, "r") as fsrc, h5py.File(tmp_path, "w") as fdst:
        for split in fsrc.keys():
            gsrc = fsrc[split]
            gdst = fdst.create_group(split)
            for key in gsrc.keys():
                dsrc = gsrc[key]
                n = dsrc.shape[0]
                if dsrc.chunks is None or len(dsrc.shape) == 1:
                    gdst.create_dataset(key, data=dsrc[...], dtype=dsrc.dtype)
                    continue
                new_chunks = (1,) + dsrc.shape[1:]
                ddst = gdst.create_dataset(
                    key, shape=dsrc.shape, dtype=dsrc.dtype, chunks=new_chunks,
                )
                for start in range(0, n, block_rows):
                    end = min(start + block_rows, n)
                    ddst[start:end] = dsrc[start:end]
                print(f"    {split}/{key}: {n} rows rechunked", flush=True)

    with h5py.File(src_path, "r") as fsrc, h5py.File(tmp_path, "r") as fdst:
        for split in fsrc.keys():
            for key in fsrc[split].keys():
                assert fsrc[split][key].shape == fdst[split][key].shape, \
                    f"shape mismatch {split}/{key}"

    os.replace(tmp_path, src_path)
    dt = time.time() - t0
    print(f"[{src_path.name}] rechunked in {dt/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="checkpoints/unsupervised")
    ap.add_argument("--pattern", default="*_uni_attn_features.h5")
    ap.add_argument("--block-rows", type=int, default=1024,
                     help="Rows per sequential read/write block during copy.")
    args = ap.parse_args()

    paths = sorted(Path(args.cache_dir).glob(args.pattern), key=lambda p: p.stat().st_size)
    if not paths:
        print(f"No files matching {args.pattern} in {args.cache_dir}")
        return
    print(f"Found {len(paths)} cache files (smallest first):")
    for p in paths:
        print(f"  {p.name}: {p.stat().st_size/1e9:.2f} GB")

    for p in paths:
        print(f"\n=== {p.name} ===", flush=True)
        if _already_rechunked(p):
            print("  already row-chunked, skipping.")
            continue
        rechunk_file(p, block_rows=args.block_rows)

    print("\nAll done.")


if __name__ == "__main__":
    main()
