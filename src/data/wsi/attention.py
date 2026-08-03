"""Attention sources and strict alignment for WSI embedding analysis."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from src.data.wsi.attention_file import (
    read_attention_coords,
    read_attention_tensor,
    reduce_attention_tensor,
)
from src.data.wsi.bag import WSIBag
from src.data.wsi.coord_alignment import align_by_coords
from src.data.wsi.feature_store import WSIFeatureStore


@dataclass(frozen=True)
class WSIAttention:
    """One attention signal aligned to the tiles of a slide."""

    slide_id: str
    values: torch.Tensor
    coords: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.slide_id:
            raise ValueError("slide_id must be non-empty.")
        if self.values.ndim != 1 or self.values.shape[0] == 0:
            raise ValueError(
                f"values must have shape [n_tiles] with n_tiles > 0; got {tuple(self.values.shape)}."
            )
        if not torch.is_floating_point(self.values):
            raise TypeError("values must be floating point.")
        if not torch.isfinite(self.values).all():
            raise ValueError("values contain NaN or Inf.")
        if self.coords is not None:
            if self.coords.ndim != 2 or self.coords.shape[1] not in (2, 4):
                raise ValueError(
                    "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
                    f"got {tuple(self.coords.shape)}."
                )
            if self.coords.shape[0] != self.values.shape[0]:
                raise ValueError("coords and values must contain the same number of tiles.")


class WSIAttentionSource(ABC):
    """Abstract source of one tile-attention vector per slide."""

    @abstractmethod
    def read(self, slide_id: str, *, n_tiles: int) -> WSIAttention:
        """Read and reduce the attention signal for ``slide_id``."""


class EmbeddedAttentionSource(WSIAttentionSource):
    """Read the legacy ``attention`` dataset embedded in a feature store."""

    def __init__(self, store: WSIFeatureStore) -> None:
        self.store = store

    def read(self, slide_id: str, *, n_tiles: int) -> WSIAttention:
        bag = self.store.read(slide_id)
        if bag.attention is None:
            raise ValueError(
                f"slide {slide_id} has no embedded attention dataset in the feature store."
            )
        if bag.attention.shape[0] != n_tiles:
            raise ValueError(
                f"slide {slide_id}: embedded attention has {bag.attention.shape[0]} tiles, "
                f"feature bag has {n_tiles}."
            )
        return WSIAttention(
            slide_id=slide_id,
            values=bag.attention.to(torch.float32),
            coords=bag.coords,
            metadata={"source": "embedded_h5_attention"},
        )


class FeatureStoreAttentionSource(WSIAttentionSource):
    """Read an existing HDF5 target store without converting it.

    Old WSI target stores normally put the signal in ``WSIBag.attention``.  A
    one-column ``tile_features`` fallback is supported for locally generated
    stores that encoded the signal as a feature vector.
    """

    def __init__(
        self,
        store: WSIFeatureStore,
        *,
        allow_single_feature_column: bool = True,
    ) -> None:
        self.store = store
        self.allow_single_feature_column = allow_single_feature_column

    def read(self, slide_id: str, *, n_tiles: int) -> WSIAttention:
        bag = self.store.read(slide_id)
        if bag.attention is not None:
            values = bag.attention.to(torch.float32)
            field = "attention"
        elif self.allow_single_feature_column and bag.tile_features.shape[1] == 1:
            values = bag.tile_features[:, 0].to(torch.float32)
            field = "tile_features[:,0]"
        else:
            raise ValueError(
                f"slide {slide_id}: target store has neither an attention field nor "
                "a one-column tile_features tensor."
            )
        if values.shape[0] != n_tiles and bag.coords is None:
            raise ValueError(
                f"slide {slide_id}: target store has {values.shape[0]} values and no coords; "
                f"feature bag has {n_tiles} tiles."
            )
        return WSIAttention(
            slide_id=slide_id,
            values=values,
            coords=bag.coords,
            metadata={"source": "feature_store", "field": field},
        )


@dataclass(frozen=True)
class _ManifestRow:
    slide_id: str
    attention_path: Path
    coords_path: Path | None
    attention_key: str | None
    coords_key: str | None
    tile_axis: int | None
    tile_slice_start: int | None
    reduction: str | None


class ManifestAttentionSource(WSIAttentionSource):
    """Read per-slide attention tensors listed in a CSV manifest.

    Both the new ``attention_path`` column and the legacy ``target_path``
    column are accepted.  Relative paths are resolved against the manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        attention_key: str | None = None,
        coords_key: str | None = None,
        tile_axis: int | None = None,
        tile_slice_start: int | None = None,
        reduction: str = "mean",
        selections: Mapping[int, int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.attention_key = attention_key
        self.coords_key = coords_key
        self.tile_axis = tile_axis
        self.tile_slice_start = tile_slice_start
        self.reduction = reduction
        self.selections = dict(selections or {})
        self._rows = self._read_manifest()

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.manifest_path.parent / path).resolve()

    def _read_manifest(self) -> dict[str, _ManifestRow]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        rows: dict[str, _ManifestRow] = {}
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                slide_id = (raw.get("slide_id") or "").strip()
                path_value = (raw.get("attention_path") or raw.get("target_path") or "").strip()
                if not slide_id or not path_value:
                    raise ValueError(
                        "attention manifest requires slide_id and attention_path/target_path."
                    )
                if slide_id in rows:
                    raise ValueError(f"duplicate slide_id in attention manifest: {slide_id}")
                coords_value = (raw.get("coords_path") or "").strip()
                row_axis = (raw.get("tile_axis") or "").strip()
                row_slice_start = (raw.get("tile_slice_start") or "").strip()
                rows[slide_id] = _ManifestRow(
                    slide_id=slide_id,
                    attention_path=self._resolve(path_value),
                    coords_path=self._resolve(coords_value) if coords_value else None,
                    attention_key=(raw.get("attention_key") or raw.get("target_key") or "").strip()
                    or None,
                    coords_key=(raw.get("coords_key") or "").strip() or None,
                    tile_axis=int(row_axis) if row_axis else None,
                    tile_slice_start=int(row_slice_start) if row_slice_start else None,
                    reduction=(raw.get("reduction") or "").strip() or None,
                )
        if not rows:
            raise ValueError(f"attention manifest is empty: {self.manifest_path}")
        return rows

    def read(self, slide_id: str, *, n_tiles: int) -> WSIAttention:
        try:
            row = self._rows[slide_id]
        except KeyError as exc:
            raise KeyError(f"slide {slide_id} is absent from {self.manifest_path}") from exc
        tensor = read_attention_tensor(
            row.attention_path,
            key=row.attention_key or self.attention_key,
        )
        values = reduce_attention_tensor(
            tensor,
            n_tiles=n_tiles,
            tile_axis=row.tile_axis if row.tile_axis is not None else self.tile_axis,
            tile_slice_start=(
                row.tile_slice_start
                if row.tile_slice_start is not None
                else self.tile_slice_start
            ),
            reduction=row.reduction or self.reduction,
            selections=self.selections,
        )
        coords = None
        if row.coords_path is not None:
            coords = read_attention_coords(
                row.coords_path,
                key=row.coords_key or self.coords_key,
            )
            if coords.shape[0] != values.shape[0]:
                raise ValueError(
                    f"slide {slide_id}: attention coords and values have different lengths."
                )
        return WSIAttention(
            slide_id=slide_id,
            values=values,
            coords=coords,
            metadata={
                "source": "manifest",
                "attention_path": str(row.attention_path),
                "tile_slice_start": (
                    row.tile_slice_start
                    if row.tile_slice_start is not None
                    else self.tile_slice_start
                ),
                "reduction": row.reduction or self.reduction,
            },
        )


def align_attention_to_bag(
    bag: WSIBag,
    attention: WSIAttention,
    *,
    mode: str = "auto",
) -> tuple[WSIBag, WSIAttention]:
    """Return feature and attention records in identical tile order."""

    if bag.slide_id != attention.slide_id:
        raise ValueError(
            f"slide_id mismatch: feature={bag.slide_id}, attention={attention.slide_id}."
        )
    if mode not in {"auto", "index", "coords"}:
        raise ValueError("alignment mode must be one of: auto, index, coords.")

    resolved_mode = mode
    if mode == "auto":
        resolved_mode = "coords" if bag.coords is not None and attention.coords is not None else "index"

    if resolved_mode == "coords":
        if bag.coords is None or attention.coords is None:
            raise ValueError("coordinate alignment requires coords in both feature and attention data.")
        input_indices, attention_indices = align_by_coords(bag.coords, attention.coords)
        aligned_bag = WSIBag(
            slide_id=bag.slide_id,
            tile_features=bag.tile_features[input_indices],
            coords=bag.coords[input_indices],
            label=bag.label,
            attention=None,
            metadata=bag.metadata,
        )
        aligned_attention = WSIAttention(
            slide_id=attention.slide_id,
            values=attention.values[attention_indices],
            coords=attention.coords[attention_indices],
            metadata=attention.metadata,
        )
        return aligned_bag, aligned_attention

    if bag.n_tiles != attention.values.shape[0]:
        raise ValueError(
            f"slide {bag.slide_id}: index alignment requires equal tile counts; "
            f"got {bag.n_tiles} and {attention.values.shape[0]}."
        )
    if bag.coords is not None and attention.coords is not None and not torch.equal(
        bag.coords, attention.coords
    ):
        raise ValueError(
            f"slide {bag.slide_id}: coords differ under index alignment; use mode='coords'."
        )
    return bag, attention
