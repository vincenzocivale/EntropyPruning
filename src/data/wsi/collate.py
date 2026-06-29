"""Collation utilities for WSI-level bags."""

from __future__ import annotations

from collections.abc import Sequence

from src.data.wsi.bag import WSIBag


def collate_wsi_bags(batch: Sequence[WSIBag]) -> list[WSIBag]:
    """Return a list of WSI bags without tensor stacking.

    WSI bags usually contain a variable number of tiles, so default PyTorch
    collation is not appropriate: it would try to stack tensors with different
    first dimensions. This collate function keeps each slide intact.

    Args:
        batch: Sequence of ``WSIBag`` objects.

    Returns:
        A list of ``WSIBag`` objects, one per slide.

    Raises:
        ValueError: If the batch is empty.
        TypeError: If any item is not a ``WSIBag``.
    """

    if len(batch) == 0:
        raise ValueError("batch must contain at least one WSIBag.")

    bags = list(batch)
    for index, bag in enumerate(bags):
        if not isinstance(bag, WSIBag):
            raise TypeError(
                "collate_wsi_bags expects WSIBag instances; "
                f"item {index} has type {type(bag).__name__}."
            )

    return bags
