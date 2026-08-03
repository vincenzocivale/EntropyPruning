from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .io import TileFeatureRecord, output_is_complete, write_tile_feature_record
from .patch_dataset import OpenSlideCoordinateDataset
from .registry import SlideEntry
from .tile_encoders.base import TileEncoderAdapter
from .utils import seed_everything


@dataclass(frozen=True)
class TileExtractionConfig:
    output_dir: Path
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    storage_dtype: str = "float16"
    compression: str | None = "lzf"
    device: str = "cuda"
    seed: int = 17
    overwrite: bool = False


def extract_slide_tile_embeddings(
    *,
    slide: SlideEntry,
    coords_path: Path,
    encoder: TileEncoderAdapter,
    config: TileExtractionConfig,
    patch_size_level0: int | None = None,
) -> dict:
    output_path = Path(config.output_dir) / f"{slide.slide_id}.h5"
    if not config.overwrite and output_is_complete(output_path, "eaf.wsi.tile_features.v2"):
        return {"slide_id": slide.slide_id, "path": str(output_path), "status": "skipped"}

    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or not config.device.startswith("cuda") else "cpu")
    dataset = OpenSlideCoordinateDataset(
        slide.wsi_path,
        coords_path,
        encoder.transform,
        output_size=encoder.input_size,
        patch_size_level0=patch_size_level0,
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
    loader = DataLoader(**loader_kwargs)

    encoder.to(device).eval()
    buffers: dict[str, list[np.ndarray]] = {}
    coords_parts: list[np.ndarray] = []
    started = time.time()
    autocast_enabled = device.type == "cuda"
    for images, coords in tqdm(loader, desc=f"{slide.slide_id}: tiles", leave=False, unit="batch"):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            output = encoder.encode(images)
        for name, tensor in output.embeddings.items():
            buffers.setdefault(name, []).append(tensor.detach().float().cpu().numpy())
        coords_parts.append(coords.numpy())

    embeddings = {name: np.concatenate(parts, axis=0) for name, parts in buffers.items()}
    coords = np.concatenate(coords_parts, axis=0)
    write_tile_feature_record(
        output_path,
        TileFeatureRecord(
            slide_id=slide.slide_id,
            coords=coords,
            embeddings=embeddings,
            metadata={
                "tile_encoder": encoder.name,
                "intermediate_layer": 2,
                "input_size": encoder.input_size,
                "source_wsi": str(slide.wsi_path),
                "source_coords": str(coords_path),
            },
        ),
        storage_dtype=config.storage_dtype,
        compression=config.compression,
    )
    return {
        "slide_id": slide.slide_id,
        "path": str(output_path),
        "status": "complete",
        "n_tiles": int(coords.shape[0]),
        "elapsed_s": round(time.time() - started, 2),
    }


def extract_many_slides(
    items: Iterable[tuple[SlideEntry, Path]],
    *,
    encoder: TileEncoderAdapter,
    config: TileExtractionConfig,
) -> list[dict]:
    rows: list[dict] = []
    items = list(items)
    for slide, coords_path in tqdm(items, desc="Tile embedding extraction", unit="slide"):
        try:
            rows.append(
                extract_slide_tile_embeddings(
                    slide=slide,
                    coords_path=coords_path,
                    encoder=encoder,
                    config=config,
                )
            )
        except Exception as exc:
            rows.append({"slide_id": slide.slide_id, "status": "error", "error": repr(exc)})
    return rows
