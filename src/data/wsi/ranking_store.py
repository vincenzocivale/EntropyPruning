"""Per-slide ranking store for WSI tile scoring outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import numpy as np
import torch

try:  # pragma: no cover - optional dependency path
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only when pandas is missing
    pd = None


_NPZ_FORMAT = "npz"
_PARQUET_FORMAT = "parquet"
_SUPPORTED_FORMATS = (_NPZ_FORMAT, _PARQUET_FORMAT)


def _ratio_key(value: float) -> str:
    return format(float(value), ".12g")


def _normalize_index_map(
    value: dict[str | float, torch.Tensor] | None,
    *,
    name: str,
    n_tiles: int,
) -> dict[str, torch.Tensor]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict; got {type(value).__name__}.")

    normalized: dict[str, torch.Tensor] = {}
    for raw_key, raw_tensor in value.items():
        key = _ratio_key(raw_key) if isinstance(raw_key, float) else str(raw_key)
        if not isinstance(raw_tensor, torch.Tensor):
            raise TypeError(
                f"{name}[{key!r}] must be a torch.Tensor; got {type(raw_tensor).__name__}."
            )
        if raw_tensor.ndim != 1:
            raise ValueError(
                f"{name}[{key!r}] must be 1D; got shape {tuple(raw_tensor.shape)}."
            )
        if raw_tensor.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"{name}[{key!r}] must contain integer indices.")
        tensor = raw_tensor.to(dtype=torch.int64)
        if tensor.numel() > n_tiles:
            raise ValueError(
                f"{name}[{key!r}] has {tensor.numel()} items but n_tiles is {n_tiles}."
            )
        if tensor.numel() > 0:
            if int(tensor.min()) < 0 or int(tensor.max()) >= n_tiles:
                raise ValueError(f"{name}[{key!r}] contains out-of-range indices.")
            if torch.unique(tensor).numel() != tensor.numel():
                raise ValueError(f"{name}[{key!r}] contains duplicate indices.")
        normalized[key] = tensor.cpu()
    return normalized


def _validate_json_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict; got {type(metadata).__name__}.")
    try:
        json.dumps(metadata)
    except TypeError as exc:
        raise TypeError("metadata must be JSON-serializable.") from exc
    return metadata


@dataclass(frozen=True)
class WSITileRanking:
    """Ranking output for one WSI slide."""

    slide_id: str
    scores: torch.Tensor
    ranks: torch.Tensor
    coords: torch.Tensor | None = None
    selected_indices: dict[str | float, torch.Tensor] | None = None
    original_order_indices: dict[str | float, torch.Tensor] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slide_id, str) or not self.slide_id:
            raise ValueError("slide_id must be a non-empty string.")

        if not isinstance(self.scores, torch.Tensor):
            raise TypeError("scores must be a torch.Tensor.")
        if self.scores.ndim != 1:
            raise ValueError(f"scores must be 1D; got shape {tuple(self.scores.shape)}.")
        if self.scores.numel() == 0:
            raise ValueError("scores must contain at least one item.")
        if not torch.is_floating_point(self.scores):
            raise TypeError("scores must be floating-point.")
        if not torch.isfinite(self.scores).all():
            raise ValueError("scores must contain only finite values.")

        if not isinstance(self.ranks, torch.Tensor):
            raise TypeError("ranks must be a torch.Tensor.")
        if self.ranks.ndim != 1:
            raise ValueError(f"ranks must be 1D; got shape {tuple(self.ranks.shape)}.")
        if self.ranks.shape != self.scores.shape:
            raise ValueError("ranks must have the same shape as scores.")
        if self.ranks.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("ranks must be an integer tensor.")

        n_tiles = int(self.scores.numel())
        expected_ranks = torch.arange(1, n_tiles + 1, dtype=torch.int64)
        if not torch.equal(torch.sort(self.ranks.to(dtype=torch.int64)).values, expected_ranks):
            raise ValueError("ranks must be a permutation of 1..N.")

        if self.coords is not None:
            if not isinstance(self.coords, torch.Tensor):
                raise TypeError("coords must be a torch.Tensor when provided.")
            if self.coords.ndim != 2:
                raise ValueError(
                    "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
                    f"got shape {tuple(self.coords.shape)}."
                )
            if self.coords.shape[0] != n_tiles:
                raise ValueError("coords must have the same number of rows as scores.")
            if self.coords.shape[1] not in (2, 4):
                raise ValueError("coords second dimension must be 2 or 4.")

        normalized_selected = _normalize_index_map(
            self.selected_indices,
            name="selected_indices",
            n_tiles=n_tiles,
        )
        normalized_original = _normalize_index_map(
            self.original_order_indices,
            name="original_order_indices",
            n_tiles=n_tiles,
        )

        if set(normalized_selected) != set(normalized_original):
            if normalized_original:
                raise ValueError(
                    "selected_indices and original_order_indices must define the same keep ratios."
                )

        for key, selected in normalized_selected.items():
            original = normalized_original.get(key)
            if selected.numel() > 1:
                order = torch.argsort(self.scores[selected], descending=True, stable=True)
                if not torch.equal(selected, selected[order]):
                    raise ValueError(
                        f"selected_indices[{key!r}] must be ordered by descending score."
                    )
            if original is not None:
                if not torch.equal(torch.sort(selected).values, torch.sort(original).values):
                    raise ValueError(
                        f"selected_indices[{key!r}] and original_order_indices[{key!r}] "
                        "must reference the same tiles."
                    )
                if original.numel() > 1 and not torch.equal(original, torch.sort(original).values):
                    raise ValueError(
                        f"original_order_indices[{key!r}] must be sorted by original order."
                    )

        _validate_json_metadata(self.metadata)

        object.__setattr__(self, "scores", self.scores.detach().cpu())
        object.__setattr__(self, "ranks", self.ranks.to(dtype=torch.int64).detach().cpu())
        object.__setattr__(self, "coords", None if self.coords is None else self.coords.detach().cpu())
        object.__setattr__(self, "selected_indices", normalized_selected)
        object.__setattr__(self, "original_order_indices", normalized_original)


class WSIRankingStore:
    """Directory-backed ranking store with one file per slide."""

    def __init__(self, root_dir: str | Path, *, file_format: str = _NPZ_FORMAT) -> None:
        file_format = str(file_format).lower()
        if file_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"file_format must be one of {_SUPPORTED_FORMATS}; got {file_format!r}."
            )

        self.root_dir = Path(root_dir)
        self.file_format = file_format
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def slide_ids(self) -> tuple[str, ...]:
        slide_ids = [
            unquote(path.stem)
            for path in sorted(self.root_dir.glob(f"*.{self.file_format}"))
        ]
        return tuple(slide_ids)

    def exists(self, slide_id: str) -> bool:
        return self.path_for_slide(slide_id).exists()

    def path_for_slide(self, slide_id: str) -> Path:
        if not isinstance(slide_id, str) or not slide_id:
            raise ValueError("slide_id must be a non-empty string.")
        return self.root_dir / f"{quote(slide_id, safe='')}.{self.file_format}"

    def write(self, ranking: WSITileRanking) -> Path:
        if not isinstance(ranking, WSITileRanking):
            raise TypeError(
                f"write expects WSITileRanking; got {type(ranking).__name__}."
            )

        path = self.path_for_slide(ranking.slide_id)
        if self.file_format == _NPZ_FORMAT:
            self._write_npz(path, ranking)
        else:
            self._write_parquet(path, ranking)
        return path

    def read(self, slide_id: str) -> WSITileRanking:
        path = self.path_for_slide(slide_id)
        if not path.exists():
            raise KeyError(f"slide_id not found in ranking store: {slide_id}")

        if self.file_format == _NPZ_FORMAT:
            return self._read_npz(path)
        return self._read_parquet(path)

    @staticmethod
    def _selection_json(
        selected_indices: dict[str, torch.Tensor],
        original_order_indices: dict[str, torch.Tensor],
    ) -> tuple[str, str]:
        selected_payload = {
            key: tensor.to(dtype=torch.int64).tolist()
            for key, tensor in selected_indices.items()
        }
        original_payload = {
            key: tensor.to(dtype=torch.int64).tolist()
            for key, tensor in original_order_indices.items()
        }
        return json.dumps(selected_payload), json.dumps(original_payload)

    def _write_npz(self, path: Path, ranking: WSITileRanking) -> None:
        metadata_json = json.dumps(ranking.metadata or {})
        selected_json, original_json = self._selection_json(
            ranking.selected_indices or {},
            ranking.original_order_indices or {},
        )

        payload: dict[str, Any] = {
            "slide_id": np.array(ranking.slide_id),
            "scores": ranking.scores.numpy(),
            "ranks": ranking.ranks.numpy(),
            "metadata_json": np.array(metadata_json),
            "selected_indices_json": np.array(selected_json),
            "original_order_indices_json": np.array(original_json),
        }
        if ranking.coords is not None:
            payload["coords"] = ranking.coords.numpy()

        np.savez_compressed(path, **payload)

    def _read_npz(self, path: Path) -> WSITileRanking:
        with np.load(path, allow_pickle=False) as data:
            coords = torch.from_numpy(data["coords"]) if "coords" in data else None
            return WSITileRanking(
                slide_id=str(data["slide_id"].item()),
                scores=torch.from_numpy(data["scores"]),
                ranks=torch.from_numpy(data["ranks"]),
                coords=coords,
                selected_indices={
                    key: torch.tensor(value, dtype=torch.int64)
                    for key, value in json.loads(str(data["selected_indices_json"].item())).items()
                },
                original_order_indices={
                    key: torch.tensor(value, dtype=torch.int64)
                    for key, value in json.loads(
                        str(data["original_order_indices_json"].item())
                    ).items()
                },
                metadata=json.loads(str(data["metadata_json"].item())),
            )

    def _write_parquet(self, path: Path, ranking: WSITileRanking) -> None:
        if pd is None:
            raise ImportError("Parquet output requires pandas to be installed.")

        metadata_json = json.dumps(ranking.metadata or {})
        selected_json, original_json = self._selection_json(
            ranking.selected_indices or {},
            ranking.original_order_indices or {},
        )
        coords_payload = None
        if ranking.coords is not None:
            coords_payload = ranking.coords.tolist()

        frame = pd.DataFrame(
            [
                {
                    "slide_id": ranking.slide_id,
                    "coords": coords_payload,
                    "scores": ranking.scores.tolist(),
                    "ranks": ranking.ranks.to(dtype=torch.int64).tolist(),
                    "selected_indices_json": selected_json,
                    "original_order_indices_json": original_json,
                    "metadata_json": metadata_json,
                }
            ]
        )
        frame.to_parquet(path, index=False)

    def _read_parquet(self, path: Path) -> WSITileRanking:
        if pd is None:
            raise ImportError("Parquet input requires pandas to be installed.")

        frame = pd.read_parquet(path)
        if len(frame) != 1:
            raise ValueError(f"expected one row in ranking parquet file: {path}")
        row = frame.iloc[0]
        coords = None
        if row["coords"] is not None:
            coords = torch.tensor(row["coords"])

        return WSITileRanking(
            slide_id=str(row["slide_id"]),
            scores=torch.tensor(row["scores"], dtype=torch.float32),
            ranks=torch.tensor(row["ranks"], dtype=torch.int64),
            coords=coords,
            selected_indices={
                key: torch.tensor(value, dtype=torch.int64)
                for key, value in json.loads(str(row["selected_indices_json"])).items()
            },
            original_order_indices={
                key: torch.tensor(value, dtype=torch.int64)
                for key, value in json.loads(str(row["original_order_indices_json"])).items()
            },
            metadata=json.loads(str(row["metadata_json"])),
        )
