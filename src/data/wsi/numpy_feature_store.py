"""NumPy-backed WSI bag store with one memory-mappable directory per slide."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.data.wsi.bag import WSIBag
from src.data.wsi.feature_store import WSIFeatureStore
from src.wsi_pipeline.numpy_store import NumpyStoreWriter, array_names, read_array, read_metadata


class NumpyWSIFeatureStore(WSIFeatureStore):
    """Store each bag separately so appending slides never rewrites earlier bags."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if self.path.suffix != ".npyd":
            raise ValueError("NumPy feature store path must end in .npyd")
        if read_only and not (self.path / "index.json").is_file():
            raise FileNotFoundError(self.path / "index.json")
        if not read_only:
            self.path.mkdir(parents=True, exist_ok=True)
            index = self.path / "index.json"
            if not index.exists():
                index.write_text("[]\n", encoding="utf-8")

    def _ids(self) -> list[str]:
        with (self.path / "index.json").open() as handle:
            ids = json.load(handle)
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"Invalid feature-store index: {self.path}")
        return ids

    def _slide_path(self, slide_id: str) -> Path:
        digest = hashlib.sha256(slide_id.encode("utf-8")).hexdigest()
        return self.path / "slides" / f"{digest}.npyd"

    def slide_ids(self) -> tuple[str, ...]:
        return tuple(self._ids())

    def exists(self, slide_id: str) -> bool:
        if not isinstance(slide_id, str):
            raise TypeError("slide_id must be a str")
        return slide_id in self._ids() and self._slide_path(slide_id).is_dir()

    def read(self, slide_id: str) -> WSIBag:
        if not self.exists(slide_id):
            raise KeyError(f"slide_id not found in NumPy feature store: {slide_id}")
        path = self._slide_path(slide_id)
        metadata = read_metadata(path)
        names = array_names(path)
        kind = metadata.get("label_kind", "none")
        label = (None if kind == "none" else
                 int(metadata["label_value"]) if kind == "int" else
                 float(metadata["label_value"]) if kind == "float" else
                 torch.from_numpy(read_array(path, "label")) if kind == "tensor" else
                 None)
        return WSIBag(slide_id=slide_id,
                      tile_features=torch.from_numpy(read_array(path, "tile_features")),
                      coords=torch.from_numpy(read_array(path, "coords")) if "coords" in names else None,
                      attention=torch.from_numpy(read_array(path, "attention")) if "attention" in names else None,
                      label=label, metadata=metadata.get("bag_metadata"))

    def write(self, bag: WSIBag) -> None:
        if self.read_only:
            raise PermissionError(f"feature store is read-only: {self.path}")
        if not isinstance(bag, WSIBag):
            raise TypeError("write expects a WSIBag")
        ids = self._ids()
        metadata: dict[str, object] = {"schema": "eaf.wsi.feature_store.slide.v1",
                                       "slide_id": bag.slide_id, "bag_metadata": bag.metadata}
        label = bag.label
        if label is None:
            metadata["label_kind"] = "none"
        elif isinstance(label, (bool, int)):
            metadata.update(label_kind="int", label_value=int(label))
        elif isinstance(label, float):
            metadata.update(label_kind="float", label_value=label)
        elif isinstance(label, torch.Tensor):
            metadata["label_kind"] = "tensor"
        else:
            raise TypeError(f"Unsupported label type: {type(label).__name__}")
        with NumpyStoreWriter(self._slide_path(bag.slide_id), metadata, replace=True) as store:
            store.write("tile_features", bag.tile_features.detach().cpu().numpy())
            if bag.coords is not None:
                store.write("coords", bag.coords.detach().cpu().numpy())
            if bag.attention is not None:
                store.write("attention", bag.attention.detach().cpu().numpy())
            if isinstance(label, torch.Tensor):
                store.write("label", label.detach().cpu().numpy())
        if bag.slide_id not in ids:
            ids.append(bag.slide_id)
            temporary = self.path / "index.json.tmp"
            temporary.write_text(json.dumps(ids) + "\n", encoding="utf-8")
            os.replace(temporary, self.path / "index.json")
