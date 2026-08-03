from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from .io import (
    WSIOutputRecord,
    output_is_complete,
    read_tile_feature_record,
    write_wsi_output_record,
)
from .wsi_models.base import WSIModelAdapter


@dataclass(frozen=True)
class WSIExtractionConfig:
    output_dir: Path
    device: str = "cuda"
    storage_dtype: str = "float16"
    compression: str | None = "lzf"
    patch_size_level0: int = 512
    overwrite: bool = False
    max_full_attention_tiles: int = 2048


def extract_one_wsi(
    feature_path: Path,
    *,
    model: WSIModelAdapter,
    config: WSIExtractionConfig,
    feature_key: str | None = None,
) -> dict:
    record = read_tile_feature_record(feature_path)
    output_path = Path(config.output_dir) / f"{record.slide_id}.h5"
    if not config.overwrite and output_is_complete(output_path, "eaf.wsi.fm_output.v1"):
        return {"slide_id": record.slide_id, "path": str(output_path), "status": "skipped"}

    key = feature_key or model.required_feature_key
    if key not in record.embeddings:
        if len(record.embeddings) == 1:
            key = next(iter(record.embeddings))
        else:
            raise KeyError(f"Feature key {key!r} not found in {feature_path}; keys={list(record.embeddings)}")
    values = np.asarray(record.embeddings[key])
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    if values.ndim != 2:
        raise ValueError(f"Expected [N,D] features, got {values.shape}")

    device = torch.device(config.device if torch.cuda.is_available() or not config.device.startswith("cuda") else "cpu")
    features = torch.from_numpy(values).float().to(device)
    coords = torch.from_numpy(np.asarray(record.coords)).long().to(device)
    model.to(device).eval()
    started = time.time()
    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        output = model.encode(features, coords, config.patch_size_level0)

    attention = {name: tensor.detach().float().cpu().numpy() for name, tensor in output.attention.items()}
    auxiliary = {name: tensor.detach().cpu().numpy() for name, tensor in output.auxiliary.items()}
    for name, array in attention.items():
        if array.ndim >= 2 and array.shape[-1] == array.shape[-2] and array.shape[-1] > config.max_full_attention_tiles:
            raise RuntimeError(
                f"Refusing to save full {array.shape[-1]}x{array.shape[-1]} attention matrix {name!r}. "
                "Store CLS-to-tile attention or raise --max-full-attention-tiles explicitly."
            )
    slide_embedding = output.slide_embedding.detach().float().cpu().numpy()
    write_wsi_output_record(
        output_path,
        WSIOutputRecord(
            slide_id=record.slide_id,
            slide_embedding=slide_embedding,
            coords=record.coords,
            attention=attention,
            auxiliary=auxiliary,
            metadata={
                "wsi_model": model.name,
                "feature_key": key,
                "source_features": str(feature_path),
                **dict(output.metadata),
            },
        ),
        storage_dtype=config.storage_dtype,
        compression=config.compression,
    )
    return {
        "slide_id": record.slide_id,
        "path": str(output_path),
        "status": "complete",
        "n_tiles": int(values.shape[0]),
        "elapsed_s": round(time.time() - started, 2),
        "attention_available": bool(attention),
    }


def extract_many_wsi_outputs(
    feature_paths: Iterable[Path],
    *,
    model: WSIModelAdapter,
    config: WSIExtractionConfig,
    feature_key: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    paths = list(feature_paths)
    for path in tqdm(paths, desc=f"{model.name}: WSI extraction", unit="slide"):
        try:
            rows.append(extract_one_wsi(path, model=model, config=config, feature_key=feature_key))
        except Exception as exc:
            rows.append({"slide_id": Path(path).stem, "status": "error", "error": repr(exc)})
    return rows
