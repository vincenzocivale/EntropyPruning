#!/usr/bin/env python
"""Run a pretrained WSI-level foundation model on top of an existing Tile-EAF
CONCH v1.5 offline cache (`scripts/eaf.py cache tile`).

Bridges two schemas that currently don't talk to each other in this repo:

- Input: the current Tile-EAF compact cache (`coords`, `tile_embeddings`,
  `final_attention` flat datasets, ``kind="tile_eaf"``; see
  ``src/wsi_pipeline/cache_contracts.py`` / ``cache_io.TileCacheWriter``),
  produced by ``python scripts/eaf.py cache tile``.
- Output: ``eaf.wsi.fm_output.v1`` (``src/wsi_pipeline/io.py``), the schema
  ``scripts/wsi_extract_fm_outputs.py`` already writes.

``scripts/wsi_extract_fm_outputs.py`` itself reads a *different*, now-removed
input schema (``eaf.wsi.tile_features.v2``, written by the deleted
``wsi_extract_tile_embeddings.py``) -- see
``docs/wsi_preprocessing_attention_pipeline.md``'s superseded note. This
script is the missing bridge: read tile embeddings straight from the current
Tile-EAF cache, run them through the same ``WSIModelAdapter`` implementations
(``src/wsi_pipeline/wsi_models/``), and write output with the exact same
writer (`write_wsi_output_record`) so downstream consumers of
``eaf.wsi.fm_output.v1`` don't care which pipeline produced it.

Only TITAN and FEATHER are wired up here: both consume CONCH v1.5's native
768-d tile embeddings directly (``required_feature_dim=768``), matching what
the Tile-EAF cache already stores. Prov-GigaPath needs its own 1536-d tile
encoder's features and is therefore not CONCH v1.5-compatible -- it is
intentionally not offered by this script's ``--model`` choices.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wsi_pipeline.cache_io import validate_cache  # noqa: E402
from src.wsi_pipeline.io import (  # noqa: E402
    WSIOutputRecord,
    output_is_complete,
    write_wsi_output_record,
)
from src.wsi_pipeline.registry import write_manifest  # noqa: E402
from src.wsi_pipeline.wsi_models import create_wsi_model  # noqa: E402
from src.wsi_pipeline.wsi_models.base import WSIModelAdapter  # noqa: E402


def _read_tile_cache_features(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Read (slide_id, coords, tile_embeddings) straight from a Tile-EAF cache file.

    ``validate_cache`` raises (rather than returning ok/reason) on an invalid,
    incomplete, or wrong-kind cache; the caller's per-slide try/except turns that
    into an `error` row instead of aborting the whole run.
    """
    validate_cache(path, expected_kind="tile_eaf")
    with h5py.File(path, "r") as handle:
        slide_id = str(handle.attrs["slide_id"])
        coords = np.asarray(handle["coords"][:])
        embeddings = np.asarray(handle["tile_embeddings"][:])
    return slide_id, coords, embeddings


def extract_one(
    path: Path,
    *,
    model: WSIModelAdapter,
    output_dir: Path,
    device: torch.device,
    storage_dtype: str,
    compression: str | None,
    patch_size_level0: int,
    overwrite: bool,
    max_full_attention_tiles: int,
) -> dict:
    slide_id, coords, embeddings = _read_tile_cache_features(path)
    output_path = output_dir / f"{slide_id}.h5"
    if not overwrite and output_is_complete(output_path, "eaf.wsi.fm_output.v1"):
        return {"slide_id": slide_id, "path": str(output_path), "status": "skipped"}

    if embeddings.shape[1] != model.required_feature_dim:
        raise ValueError(
            f"{model.name} expects D={model.required_feature_dim}, "
            f"tile cache {path} has D={embeddings.shape[1]}"
        )

    features = torch.from_numpy(embeddings).float().to(device)
    coords_t = torch.from_numpy(coords).long().to(device)
    started = time.time()
    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        output = model.encode(features, coords_t, patch_size_level0)

    attention = {name: t.detach().float().cpu().numpy() for name, t in output.attention.items()}
    auxiliary = {name: t.detach().cpu().numpy() for name, t in output.auxiliary.items()}
    for name, array in attention.items():
        if array.ndim >= 2 and array.shape[-1] == array.shape[-2] and array.shape[-1] > max_full_attention_tiles:
            raise RuntimeError(
                f"Refusing to save full {array.shape[-1]}x{array.shape[-1]} attention matrix {name!r}. "
                "Store CLS-to-tile attention or raise --titan-max-full-attention-tokens explicitly."
            )
    slide_embedding = output.slide_embedding.detach().float().cpu().numpy()
    write_wsi_output_record(
        output_path,
        WSIOutputRecord(
            slide_id=slide_id,
            slide_embedding=slide_embedding,
            coords=coords,
            attention=attention,
            auxiliary=auxiliary,
            metadata={
                "wsi_model": model.name,
                "feature_key": "tile_embeddings",
                "source_tile_cache": str(path),
                **dict(output.metadata),
            },
        ),
        storage_dtype=storage_dtype,
        compression=compression,
    )
    return {
        "slide_id": slide_id,
        "path": str(output_path),
        "status": "complete",
        "n_tiles": int(embeddings.shape[0]),
        "elapsed_s": round(time.time() - started, 2),
        "attention_available": bool(attention),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tile-cache-dir", type=Path, required=True,
        help="Per-subset directory of *.h5 files from `eaf.py cache tile` (e.g. "
        "$EAF_WSI_ROOT/caches/tile_eaf/<dataset>/conch_v15/<cache_id>/<subset>)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("feather", "titan"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--patch-size-level0", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--feather-model", default="MahmoodLab/abmil.base.conch_v15.pc108-24k")
    parser.add_argument(
        "--titan-attention-mode", action="append",
        choices=("global_to_tokens", "received", "rollout", "full"),
        help=(
            "Repeat to select TITAN outputs. Default: global_to_tokens, received "
            "(both O(T)). 'rollout' is O(T^2) compute+memory per layer and is gated by "
            "--titan-max-rollout-tokens (default 4096) -- with HISTAI slides running up "
            "to ~20k tiles, leaving it on by default would error out on the largest "
            "slides in the corpus, so it's opt-in here rather than in wsi_extract_fm_outputs.py's default."
        ),
    )
    parser.add_argument("--titan-global-token-index", type=int, default=0)
    parser.add_argument(
        "--titan-full-layer", action="append", type=int,
        help="Layer index to save as a full HxTxT matrix; negative indices are supported.",
    )
    parser.add_argument("--titan-max-full-attention-tokens", type=int, default=2048)
    parser.add_argument("--titan-max-rollout-tokens", type=int, default=4096)
    parser.add_argument("--titan-revision")
    parser.add_argument(
        "--titan-hidden-layer", action="append", type=int,
        help=(
            "Repeat to also capture the residual-stream output of TITAN's own vision-encoder "
            "block(s) at this 0-based index (negative indices supported, same convention as "
            "--titan-full-layer). Written as auxiliary/hidden_layer_{layer:03d}, already mapped "
            "to input-tile order -- the input for a WSI-EAF forecaster that predicts final-layer "
            "attention from TITAN's own intermediate representation instead of from the tile "
            "encoder's context-free output. See src/models/wsi/dense_forecaster.py."
        ),
    )
    args = parser.parse_args()

    paths = sorted(p for p in args.tile_cache_dir.glob("*.h5"))
    if args.max_slides is not None:
        paths = paths[: args.max_slides]
    if not paths:
        raise SystemExit(f"No .h5 tile-cache files found in {args.tile_cache_dir}")

    if args.model == "feather":
        model = create_wsi_model("feather", model_id=args.feather_model, token=args.hf_token)
    else:
        modes = tuple(args.titan_attention_mode or ("global_to_tokens", "received"))
        full_layers = tuple(args.titan_full_layer or (-1,))
        model = create_wsi_model(
            "titan",
            token=args.hf_token,
            attention_modes=modes,
            global_token_index=args.titan_global_token_index,
            full_attention_layers=full_layers,
            max_full_attention_tokens=args.titan_max_full_attention_tokens,
            max_rollout_tokens=args.titan_max_rollout_tokens,
            hidden_layers=tuple(args.titan_hidden_layer or ()),
            revision=args.titan_revision,
        )

    device = torch.device(
        args.device if (torch.cuda.is_available() or not args.device.startswith("cuda")) else "cpu"
    )
    model.to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    rows: list[dict] = []
    for path in tqdm(paths, desc=f"{model.name}: WSI extraction", unit="slide"):
        try:
            rows.append(
                extract_one(
                    path,
                    model=model,
                    output_dir=args.output_dir,
                    device=device,
                    storage_dtype=args.storage_dtype,
                    compression=compression,
                    patch_size_level0=args.patch_size_level0,
                    overwrite=args.overwrite,
                    max_full_attention_tiles=args.titan_max_full_attention_tokens,
                )
            )
        except Exception as exc:  # noqa: BLE001 - record and continue, same as wsi_extract_fm_outputs.py
            rows.append({"slide_id": path.stem, "status": "error", "error": repr(exc)})

    write_manifest(rows, args.output_dir / "manifest.csv")
    errors = [r for r in rows if r.get("status") == "error"]
    missing_attention = sum(
        not r.get("attention_available", False) for r in rows if r.get("status") == "complete"
    )
    print(
        f"complete={len(rows) - len(errors)} errors={len(errors)} "
        f"without_native_attention={missing_attention} manifest={args.output_dir / 'manifest.csv'}"
    )
    if errors:
        for row in errors[:10]:
            print(f"  ERROR {row['slide_id']}: {row['error']}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
