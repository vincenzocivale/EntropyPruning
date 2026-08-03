"""Coordinate-based alignment utilities for pairing WSI feature stores.

Used when an input feature store (e.g. early-layer tile features) and a
target feature store (e.g. tile importance targets) were produced
independently and cannot be trusted to share tile order or tile count.
Alignment is done by exact coordinate match; any duplicate, mismatched, or
incomplete coordinate coverage fails loudly instead of silently dropping or
misaligning tiles.
"""

from __future__ import annotations

import torch


def align_by_coords(
    input_coords: torch.Tensor,
    target_coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two tile-coordinate tensors by exact coordinate match.

    Args:
        input_coords: Coordinates for the input store, shape ``[N, C]``.
        target_coords: Coordinates for the target store, shape ``[M, C]``.

    Returns:
        ``(input_indices, target_indices)`` index tensors such that
        ``input_coords[input_indices]`` and ``target_coords[target_indices]``
        refer to the same physical tiles, in the order tiles appear in
        ``input_coords``.

    Raises:
        ValueError: If either tensor is not 2D, if the coordinate
            dimensionality differs, if either tensor contains duplicate
            coordinates, or if coordinate coverage between the two tensors is
            incomplete in either direction.
    """

    if input_coords.ndim != 2 or target_coords.ndim != 2:
        raise ValueError(
            "coords must be 2D tensors; got input "
            f"{tuple(input_coords.shape)} and target {tuple(target_coords.shape)}."
        )

    if input_coords.shape[1] != target_coords.shape[1]:
        raise ValueError(
            "input and target coords must have the same coordinate dimension; "
            f"got {input_coords.shape[1]} and {target_coords.shape[1]}."
        )

    input_keys = [tuple(row.tolist()) for row in input_coords]
    target_keys = [tuple(row.tolist()) for row in target_coords]

    input_duplicates = _find_duplicates(input_keys)
    if input_duplicates:
        raise ValueError(
            "input coords contain duplicate tile coordinates: "
            f"{sorted(input_duplicates)[:10]}"
        )

    target_duplicates = _find_duplicates(target_keys)
    if target_duplicates:
        raise ValueError(
            "target coords contain duplicate tile coordinates: "
            f"{sorted(target_duplicates)[:10]}"
        )

    target_index_by_key = {key: index for index, key in enumerate(target_keys)}

    missing_in_target = [key for key in input_keys if key not in target_index_by_key]
    if missing_in_target:
        raise ValueError(
            f"coords coverage incomplete: {len(missing_in_target)} input tile(s) "
            f"have no matching target coordinate, e.g. {sorted(missing_in_target)[:10]}"
        )

    input_key_set = set(input_keys)
    missing_in_input = [key for key in target_keys if key not in input_key_set]
    if missing_in_input:
        raise ValueError(
            f"coords coverage incomplete: {len(missing_in_input)} target tile(s) "
            f"have no matching input coordinate, e.g. {sorted(missing_in_input)[:10]}"
        )

    input_indices = torch.arange(len(input_keys), dtype=torch.long)
    target_indices = torch.tensor(
        [target_index_by_key[key] for key in input_keys],
        dtype=torch.long,
    )

    return input_indices, target_indices


def _find_duplicates(keys: list[tuple[int, ...]]) -> set[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    duplicates: set[tuple[int, ...]] = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates
