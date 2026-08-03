"""Top-level data utilities with lazy imports.

The WSI attention audit must not import optional tile-level dependencies such as
Hugging Face ``datasets`` or torchvision. Existing public imports are retained
and resolved only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "HistologicalImageDataset",
    "H5ForecastDataset",
    "get_train_transform",
    "get_eval_transform",
    "build_loaders",
]

_LAZY_IMPORTS = {
    "HistologicalImageDataset": ("src.data.dataset", "HistologicalImageDataset"),
    "H5ForecastDataset": ("src.data.h5_dataset", "H5ForecastDataset"),
    "get_train_transform": ("src.data.transforms", "get_train_transform"),
    "get_eval_transform": ("src.data.transforms", "get_eval_transform"),
    "build_loaders": ("src.data.loaders", "build_loaders"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
