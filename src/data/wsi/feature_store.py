"""Feature-store contracts for WSI-level bags."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from src.data.wsi.bag import WSIBag


class WSIFeatureStore(ABC):
    """Abstract storage backend for WSI bags.

    Implementations may use HDF5, Parquet, Zarr, Trident outputs,
    Patho-Bench manifests, or any other physical representation. Training code
    should depend on this interface rather than on a specific storage format.
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
