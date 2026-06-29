"""Dataset contracts for WSI-level bags."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence

from torch.utils.data import Dataset

from src.data.wsi.bag import WSIBag
from src.data.wsi.feature_store import WSIFeatureStore


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


class FeatureStoreWSIBagDataset(WSIBagDataset):
    """WSI bag dataset backed by a :class:`WSIFeatureStore`.

    The dataset stores only slide ids. Actual bags are read lazily from the
    feature store in ``__getitem__``. This keeps the dataset contract stable
    while allowing different physical storage backends.
    """

    def __init__(
        self,
        store: WSIFeatureStore,
        slide_ids: Sequence[str] | None = None,
        *,
        validate_slide_ids: bool = True,
    ) -> None:
        if not isinstance(store, WSIFeatureStore):
            raise TypeError(
                "store must implement WSIFeatureStore; "
                f"got {type(store).__name__}."
            )

        if slide_ids is None:
            resolved_slide_ids = store.slide_ids()
        else:
            resolved_slide_ids = tuple(slide_ids)

        for index, slide_id in enumerate(resolved_slide_ids):
            if not isinstance(slide_id, str) or not slide_id:
                raise ValueError(
                    "slide_ids must contain non-empty strings; "
                    f"item {index} is {slide_id!r}."
                )

        if validate_slide_ids:
            missing = [slide_id for slide_id in resolved_slide_ids if not store.exists(slide_id)]
            if missing:
                raise KeyError(
                    "slide_ids not found in feature store: "
                    + ", ".join(missing[:10])
                    + (" ..." if len(missing) > 10 else "")
                )

        self.store = store
        self.slide_ids = resolved_slide_ids

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, index: int) -> WSIBag:
        if not isinstance(index, int):
            raise TypeError(f"index must be an int; got {type(index).__name__}.")
        return self.store.read(self.slide_ids[index])

    def __iter__(self) -> Iterator[WSIBag]:
        for slide_id in self.slide_ids:
            yield self.store.read(slide_id)
