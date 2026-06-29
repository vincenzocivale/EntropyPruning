"""Collation utilities for WSI-level bags."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.data.wsi.bag import WSIBag
from src.data.wsi.batch import PaddedWSIBatch, pad_wsi_bags


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


def collate_padded_wsi_bags(
    batch: Sequence[WSIBag],
    *,
    pad_value: float = 0.0,
    attention_pad_value: float = 0.0,
    coord_pad_value: int = -1,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PaddedWSIBatch:
    """Collate variable-length WSI bags into a padded tensor batch.

    This is the preferred collate function for training and evaluation loops
    that feed WSI bags directly into ``WSITileAttentionForecaster``.

    ``mask`` in the returned batch follows the EAF WSI convention:
    ``True`` means valid tile and ``False`` means padding.
    """

    return pad_wsi_bags(
        collate_wsi_bags(batch),
        pad_value=pad_value,
        attention_pad_value=attention_pad_value,
        coord_pad_value=coord_pad_value,
        device=device,
        dtype=dtype,
    )
