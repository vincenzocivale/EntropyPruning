"""Dataset contracts for WSI-level bags."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence

from torch.utils.data import Dataset

from src.data.wsi.bag import WSIBag
from src.data.wsi.feature_store import WSIFeatureStore
from src.data.wsi.paired_feature_store import load_paired_wsi_bag


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


class PairedFeatureStoreWSIBagDataset(WSIBagDataset):
    """WSI bag dataset backed by an input store and a separate target store.

    Each sample is assembled by ``load_paired_wsi_bag`` (input tile features
    joined with a target-store importance value by slide id, optionally
    aligned by coordinates) and returned as an ordinary ``WSIBag`` with
    ``attention`` set to the target importance. This lets ``pad_wsi_bags``,
    ``collate_padded_wsi_bags``, and any training loop that consumes
    ``WSIBag``/``PaddedWSIBatch`` work unchanged on paired stores. The
    legacy single-store case is supported by passing the same store object
    as both ``input_store`` and ``target_store``.
    """

    def __init__(
        self,
        input_store: WSIFeatureStore,
        target_store: WSIFeatureStore,
        slide_ids: Sequence[str] | None = None,
        *,
        alignment_mode: str = "index",
        require_coords: bool = False,
        validate_slide_ids: bool = True,
    ) -> None:
        if not isinstance(input_store, WSIFeatureStore):
            raise TypeError(
                "input_store must implement WSIFeatureStore; "
                f"got {type(input_store).__name__}."
            )
        if not isinstance(target_store, WSIFeatureStore):
            raise TypeError(
                "target_store must implement WSIFeatureStore; "
                f"got {type(target_store).__name__}."
            )

        if slide_ids is None:
            resolved_slide_ids = input_store.slide_ids()
        else:
            resolved_slide_ids = tuple(slide_ids)

        for index, slide_id in enumerate(resolved_slide_ids):
            if not isinstance(slide_id, str) or not slide_id:
                raise ValueError(
                    "slide_ids must contain non-empty strings; "
                    f"item {index} is {slide_id!r}."
                )

        if validate_slide_ids:
            missing_input = [
                slide_id for slide_id in resolved_slide_ids if not input_store.exists(slide_id)
            ]
            if missing_input:
                raise KeyError(
                    "slide_ids not found in input feature store: "
                    + ", ".join(missing_input[:10])
                    + (" ..." if len(missing_input) > 10 else "")
                )

            missing_target = [
                slide_id for slide_id in resolved_slide_ids if not target_store.exists(slide_id)
            ]
            if missing_target:
                raise KeyError(
                    "slide_ids not found in target feature store: "
                    + ", ".join(missing_target[:10])
                    + (" ..." if len(missing_target) > 10 else "")
                )

        self.input_store = input_store
        self.target_store = target_store
        self.slide_ids = resolved_slide_ids
        self.alignment_mode = alignment_mode
        self.require_coords = require_coords

    def __len__(self) -> int:
        return len(self.slide_ids)

    def _load(self, slide_id: str) -> WSIBag:
        paired = load_paired_wsi_bag(
            self.input_store,
            self.target_store,
            slide_id,
            alignment_mode=self.alignment_mode,
            require_coords=self.require_coords,
        )
        return paired.to_wsi_bag()

    def __getitem__(self, index: int) -> WSIBag:
        if not isinstance(index, int):
            raise TypeError(f"index must be an int; got {type(index).__name__}.")
        return self._load(self.slide_ids[index])

    def __iter__(self) -> Iterator[WSIBag]:
        for slide_id in self.slide_ids:
            yield self._load(slide_id)
