"""Atomic, memory-mappable NumPy directories for generated WSI arrays.

Each ``*.npyd`` directory contains one ``.npy`` file per dataset and a
``metadata.json`` file. Dataset names retain HDF5-style group paths.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SUFFIX = ".npyd"


def numpy_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == SUFFIX else path.with_suffix(SUFFIX)


def preferred_path(path: str | Path) -> Path:
    """Prefer a completed NumPy sibling while historical HDF5 remains readable."""
    path = Path(path)
    candidate = numpy_path(path)
    return candidate if candidate.is_dir() and (candidate / "metadata.json").is_file() else path


def read_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == SUFFIX:
        with (path / "metadata.json").open() as handle:
            return json.load(handle)
    import h5py

    with h5py.File(path, "r") as handle:
        return {key: _json_value(value) for key, value in handle.attrs.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_array(path: str | Path, key: str, *, mmap: bool = False) -> np.ndarray:
    path = Path(path)
    if path.suffix == SUFFIX:
        return np.load(path / f"{key}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle[key][...])


def array_names(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix == SUFFIX:
        return sorted(str(item.relative_to(path).with_suffix("")) for item in path.rglob("*.npy"))
    import h5py

    names: list[str] = []
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
    return sorted(names)


class NumpyStoreWriter:
    """Publish a directory only after all arrays and metadata are complete."""

    def __init__(self, path: str | Path, metadata: Mapping[str, Any], *, replace: bool = False) -> None:
        self.path = numpy_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix=f".{self.path.name}.", dir=self.path.parent))
        self.metadata = {str(k): _json_value(v) for k, v in metadata.items()}
        self._closed = False
        self.replace = replace

    def memmap(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.memmap:
        target = self.tmp / f"{key}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(target, mode="w+", dtype=dtype, shape=shape)

    def write(self, key: str, value: Any) -> None:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Object arrays cannot be saved safely: {key}")
        target = self.tmp / f"{key}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, array, allow_pickle=False)

    def close(self, *, publish: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if not publish:
            shutil.rmtree(self.tmp)
            return
        self.metadata["complete"] = True
        with (self.tmp / "metadata.json").open("w") as handle:
            json.dump(self.metadata, handle, sort_keys=True, indent=2, allow_nan=False)
        if self.path.exists():
            if not self.replace:
                shutil.rmtree(self.tmp)
                raise FileExistsError(self.path)
            backup = self.path.with_name(f".{self.path.name}.previous.{os.getpid()}")
            if backup.exists():
                shutil.rmtree(self.tmp)
                raise FileExistsError(backup)
            os.rename(self.path, backup)
            try:
                os.rename(self.tmp, self.path)
            except BaseException:
                os.rename(backup, self.path)
                raise
            shutil.rmtree(backup)
        else:
            os.rename(self.tmp, self.path)

    def __enter__(self) -> "NumpyStoreWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(publish=exc_type is None)


def convert_h5(path: str | Path, *, chunk_bytes: int = 16 << 20) -> Path:
    """Copy one HDF5 artifact losslessly, using bounded memory, and verify it."""
    import h5py

    source = Path(path)
    destination = numpy_path(source)
    if destination.exists():
        verify_conversion(source, destination, chunk_bytes=chunk_bytes)
        return destination
    with h5py.File(source, "r") as handle:
        metadata = {key: _json_value(value) for key, value in handle.attrs.items()}
        object_attrs: dict[str, dict[str, Any]] = {}
        handle.visititems(lambda name, obj: object_attrs.update({name: {
            key: _json_value(value) for key, value in obj.attrs.items()
        }}) if obj.attrs else None)
        if object_attrs:
            metadata["_hdf5_object_attrs"] = object_attrs
        with NumpyStoreWriter(destination, metadata) as writer:
            for key in array_names(source):
                dataset = handle[key]
                if dataset.dtype.hasobject:
                    raise ValueError(f"Object dtype not supported: {source}:{key}")
                target = writer.memmap(key, dataset.shape, dataset.dtype)
                if dataset.ndim == 0:
                    target[...] = dataset[()]
                else:
                    rows = max(1, chunk_bytes // max(1, dataset.dtype.itemsize * int(np.prod(dataset.shape[1:]))))
                    for start in range(0, dataset.shape[0], rows):
                        target[start:start + rows] = dataset[start:start + rows]
                target.flush()
                del target
    verify_conversion(source, destination, chunk_bytes=chunk_bytes)
    return destination


def verify_conversion(source: str | Path, destination: str | Path, *, chunk_bytes: int = 16 << 20) -> None:
    """Compare every element and root attribute against the source HDF5 file."""
    import h5py

    source, destination = Path(source), Path(destination)
    with h5py.File(source, "r") as handle:
        expected = {key: _json_value(value) for key, value in handle.attrs.items()}
        observed = read_metadata(destination)
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Metadata mismatch: {source}")
        object_attrs: dict[str, dict[str, Any]] = {}
        handle.visititems(lambda name, obj: object_attrs.update({name: {
            key: _json_value(value) for key, value in obj.attrs.items()
        }}) if obj.attrs else None)
        if observed.get("_hdf5_object_attrs", {}) != object_attrs:
            raise ValueError(f"Dataset/group attribute mismatch: {source}")
        if sorted(array_names(source)) != sorted(array_names(destination)):
            raise ValueError(f"Dataset list mismatch: {source}")
        for key in array_names(source):
            old = handle[key]
            new = read_array(destination, key, mmap=True)
            if old.shape != new.shape or old.dtype != new.dtype:
                raise ValueError(f"Shape/dtype mismatch: {source}:{key}")
            if old.ndim == 0:
                equal = np.array_equal(old[()], new[()], equal_nan=True)
                if not equal:
                    raise ValueError(f"Value mismatch: {source}:{key}")
            else:
                rows = max(1, chunk_bytes // max(1, old.dtype.itemsize * int(np.prod(old.shape[1:]))))
                for start in range(0, old.shape[0], rows):
                    if not np.array_equal(old[start:start + rows], new[start:start + rows], equal_nan=True):
                        raise ValueError(f"Value mismatch: {source}:{key} row {start}")
