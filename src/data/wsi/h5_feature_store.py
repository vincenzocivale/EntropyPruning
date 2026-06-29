"""HDF5-backed feature store for WSI-level bags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.wsi.bag import WSIBag
from src.data.wsi.feature_store import WSIFeatureStore

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only when h5py is missing
    h5py = None


_SCHEMA_VERSION = 1
_SLIDES_GROUP = "slides"


class H5WSIFeatureStore(WSIFeatureStore):
    """HDF5-backed storage for :class:`WSIBag` objects.

    This store is intentionally conservative: it serializes tensors and simple
    scalar labels, while keeping the training code dependent only on the
    ``WSIFeatureStore`` interface.
    """

    def __init__(self, path: str | Path) -> None:
        if h5py is None:
            raise ImportError("H5WSIFeatureStore requires h5py to be installed.")

        self.path = Path(path)

        with h5py.File(self.path, "a") as handle:
            handle.attrs.setdefault("schema_version", _SCHEMA_VERSION)
            handle.require_group(_SLIDES_GROUP)

    def slide_ids(self) -> tuple[str, ...]:
        with h5py.File(self.path, "r") as handle:
            slides = handle[_SLIDES_GROUP]
            ordered = sorted(
                slides.keys(),
                key=lambda key: int(slides[key].attrs.get("order", int(key))),
            )
            return tuple(str(slides[key].attrs["slide_id"]) for key in ordered)

    def exists(self, slide_id: str) -> bool:
        if not isinstance(slide_id, str):
            raise TypeError(f"slide_id must be a str; got {type(slide_id).__name__}.")
        return self._find_group_key(slide_id) is not None

    def read(self, slide_id: str) -> WSIBag:
        if not isinstance(slide_id, str):
            raise TypeError(f"slide_id must be a str; got {type(slide_id).__name__}.")

        with h5py.File(self.path, "r") as handle:
            key = self._find_group_key(slide_id, handle=handle)
            if key is None:
                raise KeyError(f"slide_id not found in HDF5 feature store: {slide_id}")

            group = handle[_SLIDES_GROUP][key]

            tile_features = torch.from_numpy(group["tile_features"][...])
            coords = torch.from_numpy(group["coords"][...]) if "coords" in group else None
            attention = (
                torch.from_numpy(group["attention"][...])
                if "attention" in group
                else None
            )
            label = self._read_label(group)
            metadata = self._read_metadata(group)

            return WSIBag(
                slide_id=str(group.attrs["slide_id"]),
                tile_features=tile_features,
                coords=coords,
                label=label,
                attention=attention,
                metadata=metadata,
            )

    def write(self, bag: WSIBag) -> None:
        if not isinstance(bag, WSIBag):
            raise TypeError(
                "H5WSIFeatureStore.write expects a WSIBag; "
                f"got {type(bag).__name__}."
            )

        with h5py.File(self.path, "a") as handle:
            slides = handle[_SLIDES_GROUP]
            key = self._find_group_key(bag.slide_id, handle=handle)

            if key is None:
                key = self._next_group_key(slides)
                group = slides.create_group(key)
                order = len(slides) - 1
            else:
                group = slides[key]
                order = int(group.attrs["order"])
                self._clear_group(group)

            group.attrs["slide_id"] = bag.slide_id
            group.attrs["order"] = order

            group.create_dataset(
                "tile_features",
                data=bag.tile_features.detach().cpu().numpy(),
                compression="gzip",
                compression_opts=4,
            )

            if bag.coords is not None:
                group.create_dataset(
                    "coords",
                    data=bag.coords.detach().cpu().numpy(),
                    compression="gzip",
                    compression_opts=4,
                )

            if bag.attention is not None:
                group.create_dataset(
                    "attention",
                    data=bag.attention.detach().cpu().numpy(),
                    compression="gzip",
                    compression_opts=4,
                )

            self._write_label(group, bag.label)
            self._write_metadata(group, bag.metadata)

    def _find_group_key(self, slide_id: str, *, handle: Any | None = None) -> str | None:
        if handle is None:
            with h5py.File(self.path, "r") as opened:
                return self._find_group_key(slide_id, handle=opened)

        slides = handle[_SLIDES_GROUP]
        for key in slides.keys():
            if str(slides[key].attrs.get("slide_id")) == slide_id:
                return str(key)
        return None

    @staticmethod
    def _next_group_key(slides: Any) -> str:
        numeric_keys = [int(key) for key in slides.keys() if str(key).isdigit()]
        next_index = max(numeric_keys, default=-1) + 1
        return f"{next_index:08d}"

    @staticmethod
    def _clear_group(group: Any) -> None:
        for name in list(group.keys()):
            del group[name]
        for name in list(group.attrs.keys()):
            del group.attrs[name]

    @staticmethod
    def _write_label(group: Any, label: int | float | torch.Tensor | None) -> None:
        if label is None:
            group.attrs["label_kind"] = "none"
            return

        if isinstance(label, bool):
            group.attrs["label_kind"] = "int"
            group.attrs["label_value"] = int(label)
            return

        if isinstance(label, int):
            group.attrs["label_kind"] = "int"
            group.attrs["label_value"] = label
            return

        if isinstance(label, float):
            group.attrs["label_kind"] = "float"
            group.attrs["label_value"] = label
            return

        if isinstance(label, torch.Tensor):
            group.attrs["label_kind"] = "tensor"
            group.create_dataset("label", data=label.detach().cpu().numpy())
            return

        raise TypeError(
            "H5WSIFeatureStore supports labels of type int, float, torch.Tensor, "
            f"or None; got {type(label).__name__}."
        )

    @staticmethod
    def _read_label(group: Any) -> int | float | torch.Tensor | None:
        label_kind = str(group.attrs.get("label_kind", "none"))

        if label_kind == "none":
            return None
        if label_kind == "int":
            return int(group.attrs["label_value"])
        if label_kind == "float":
            return float(group.attrs["label_value"])
        if label_kind == "tensor":
            return torch.from_numpy(group["label"][...])

        raise ValueError(f"Unsupported label_kind in HDF5 feature store: {label_kind}")

    @staticmethod
    def _write_metadata(group: Any, metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            return

        try:
            group.attrs["metadata_json"] = json.dumps(metadata)
        except TypeError as exc:
            raise TypeError("metadata must be JSON-serializable for HDF5 storage.") from exc

    @staticmethod
    def _read_metadata(group: Any) -> dict[str, Any] | None:
        metadata_json = group.attrs.get("metadata_json")
        if metadata_json is None:
            return None
        return json.loads(str(metadata_json))
