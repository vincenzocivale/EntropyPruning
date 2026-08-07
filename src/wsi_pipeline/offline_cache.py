"""Generic orchestration for generating frozen teacher caches offline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .cache_contracts import TileCacheSpec, WSICacheSpec
from .cache_io import TileCacheWriter, write_wsi_cache
from .model_adapters import TileTeacherAdapter, WSITeacherAdapter


def cache_tile_slide(
    *,
    adapter: TileTeacherAdapter,
    batches: Iterable[tuple[Any, Any]],
    output_path: str | Path,
    slide_id: str,
    case_id: str,
    spec: TileCacheSpec,
) -> Path:
    """Execute a frozen tile teacher once and persist all supervision needed by EAF.

    ``batches`` yields ``(images, coords)``. The adapter owns model-specific hooks for
    the requested early layer and final attention. Training later consumes only the
    resulting HDF5 cache.
    """

    output_path = Path(output_path)
    with TileCacheWriter(output_path, spec, slide_id=slide_id, case_id=case_id) as writer:
        for images, coords in batches:
            output = adapter.extract(images, early_layer=spec.early_layer)
            writer.append(
                coords=coords,
                early_tokens=output.early_tokens,
                final_attention=output.final_attention,
                tile_embeddings=output.tile_embeddings,
            )
    return output_path


def cache_wsi_slide(
    *,
    adapter: WSITeacherAdapter,
    tile_embeddings: Any,
    coords: Any,
    output_path: str | Path,
    slide_id: str,
    case_id: str,
    spec: WSICacheSpec,
) -> Path:
    """Execute a frozen WSI teacher once and persist tile scores/full embedding."""

    output = adapter.extract(tile_embeddings, coords)
    return write_wsi_cache(
        output_path,
        spec,
        slide_id=slide_id,
        case_id=case_id,
        coords=coords,
        tile_embeddings=tile_embeddings,
        tile_scores=output.tile_scores,
        wsi_embedding=output.wsi_embedding,
        raw_attention=output.raw_attention,
    )
