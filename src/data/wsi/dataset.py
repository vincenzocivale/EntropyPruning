"""Dataset contracts for WSI-level bags."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from torch.utils.data import Dataset

from src.data.wsi.bag import WSIBag


class WSIBagDataset(Dataset, ABC):
    """Abstract dataset whose samples are :class:`WSIBag` objects.

    This class intentionally defines only the minimal contract needed by
    WSI-level training and evaluation code. Concrete implementations may read
    from HDF5, Parquet, Patho-Bench manifests, Trident outputs, or in-memory
    fixtures, but they should all expose the same sample type.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of WSI bags."""

    @abstractmethod
    def __getitem__(self, index: int) -> WSIBag:
        """Return one WSI bag."""


class InMemoryWSIBagDataset(WSIBagDataset):
    """Simple immutable WSI bag dataset backed by an in-memory sequence.

    This is useful for tests, synthetic fixtures, and small debugging runs.
    Production-scale datasets should implement ``WSIBagDataset`` directly
    without materialising all bags in memory.
    """

    def __init__(self, bags: Iterable[WSIBag]) -> None:
        self._bags = tuple(bags)

        for index, bag in enumerate(self._bags):
            if not isinstance(bag, WSIBag):
                raise TypeError(
                    "InMemoryWSIBagDataset expects WSIBag instances; "
                    f"item {index} has type {type(bag).__name__}."
                )

    def __len__(self) -> int:
        return len(self._bags)

    def __getitem__(self, index: int) -> WSIBag:
        if not isinstance(index, int):
            raise TypeError(f"index must be an int; got {type(index).__name__}.")
        return self._bags[index]

    def __iter__(self) -> Iterator[WSIBag]:
        return iter(self._bags)
