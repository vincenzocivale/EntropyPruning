from __future__ import annotations

from .base import WSIModelAdapter
from .feather import FeatherABMILAdapter
from .gigapath import ProvGigaPathAdapter
from .titan import TitanAdapter


def create_wsi_model(name: str, **kwargs) -> WSIModelAdapter:
    normalized = name.lower()
    if normalized == "feather":
        return FeatherABMILAdapter(**kwargs)
    if normalized == "titan":
        return TitanAdapter(**kwargs)
    if normalized in {"gigapath", "prov-gigapath"}:
        return ProvGigaPathAdapter(**kwargs)
    raise ValueError(f"Unsupported WSI model: {name}")
