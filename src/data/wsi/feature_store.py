"""Feature-store contracts for WSI-level bags."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from torch.utils.data import Dataset

from src.data.wsi.bag import WSIBag


class WSIFeatureStore(ABC):
    """Abstract storage backend for WSI bags.

    Implementations may use HDF5, Parquet, Zarr, Trident outputs,
    per-slide manifests, or another physical representation. Preprocessing and
    analysis code should depend on this interface rather than a storage format.
    """

    @abstractmethod
    def slide_ids(self) -> tuple[str, ...]:
        """Return all available slide ids."""

    @abstractmethod
    def exists(self, slide_id: str) -> bool:
        """Return whether ``slide_id`` is available in the store."""

    @abstractmethod
    def read(self, slide_id: str) -> WSIBag:
        """Read one WSI bag by slide id."""

    @abstractmethod
    def write(self, bag: WSIBag) -> None:
        """Write or overwrite one WSI bag."""

    def __contains__(self, slide_id: object) -> bool:
        if not isinstance(slide_id, str):
            return False
        return self.exists(slide_id)

    def __len__(self) -> int:
        return len(self.slide_ids())


class InMemoryWSIFeatureStore(WSIFeatureStore):
    """In-memory feature store for tests and small debugging runs."""

    def __init__(self, bags: Iterable[WSIBag] | None = None) -> None:
        self._bags: dict[str, WSIBag] = {}

        if bags is not None:
            for bag in bags:
                self.write(bag)

    def slide_ids(self) -> tuple[str, ...]:
        return tuple(self._bags.keys())

    def exists(self, slide_id: str) -> bool:
        if not isinstance(slide_id, str):
            raise TypeError(f"slide_id must be a str; got {type(slide_id).__name__}.")
        return slide_id in self._bags

    def read(self, slide_id: str) -> WSIBag:
        if not isinstance(slide_id, str):
            raise TypeError(f"slide_id must be a str; got {type(slide_id).__name__}.")
        try:
            return self._bags[slide_id]
        except KeyError as exc:
            raise KeyError(f"slide_id not found in feature store: {slide_id}") from exc

    def write(self, bag: WSIBag) -> None:
        if not isinstance(bag, WSIBag):
            raise TypeError(
                "InMemoryWSIFeatureStore.write expects a WSIBag; "
                f"got {type(bag).__name__}."
            )
        self._bags[bag.slide_id] = bag


class FeatureStoreWSIBagDataset(Dataset):
    """Expose selected bags from any :class:`WSIFeatureStore` as a dataset."""

    def __init__(self, store: WSIFeatureStore, *, slide_ids: Iterable[str] | None = None) -> None:
        self.store = store
        self.slide_ids = tuple(slide_ids if slide_ids is not None else store.slide_ids())
        missing = [slide_id for slide_id in self.slide_ids if not store.exists(slide_id)]
        if missing:
            raise KeyError(f"Feature store is missing slide ids: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, index: int) -> WSIBag:
        return self.store.read(self.slide_ids[index])
