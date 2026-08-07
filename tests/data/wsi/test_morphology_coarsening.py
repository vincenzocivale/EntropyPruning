from __future__ import annotations

import math

import torch

from src.data.wsi.morphology_coarsening import (
    MorphologyCoarseningConfig,
    build_spatial_edges,
    coarsen_wsi_tiles,
)


def _grid(width: int, height: int) -> torch.Tensor:
    coords = []
    for y in range(height):
        for x in range(width):
            coords.append((x * 512, y * 512))
    return torch.tensor(coords, dtype=torch.long)


def test_random_selection_is_deterministic_and_exact_budget() -> None:
    features = torch.randn(20, 8, generator=torch.Generator().manual_seed(3))
    config = MorphologyCoarseningConfig(
        keep_ratio=0.25,
        strategy="random",
        seed=19,
    )

    first = coarsen_wsi_tiles(features, None, config)
    second = coarsen_wsi_tiles(features, None, config)

    assert torch.equal(first.selected_indices, second.selected_indices)
    assert first.selected_indices.numel() == math.ceil(20 * 0.25)
    assert torch.equal(first.selected_indices, torch.sort(first.selected_indices).values)
    assert torch.unique(first.selected_indices).numel() == first.selected_indices.numel()


def test_grid_edges_recover_four_and_eight_neighborhoods() -> None:
    coords = _grid(3, 3)

    edges4, diagnostics4 = build_spatial_edges(coords, mode="grid", connectivity=4)
    edges8, diagnostics8 = build_spatial_edges(coords, mode="grid", connectivity=8)

    assert edges4.shape == (12, 2)
    assert edges8.shape == (20, 2)
    assert diagnostics4["spatial_edge_mode"] == "grid"
    assert diagnostics8["grid_quantization_error"] == 0.0


def test_topology_selector_rescues_small_rare_region() -> None:
    coords = _grid(5, 2)
    # Three spatially connected morphologies: a large common region, a second
    # common region, and a one-tile rare focus in the bottom-right corner.
    common_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    common_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    rare = torch.tensor([0.0, 0.0, 1.0, 0.0])
    features = torch.stack(
        [
            common_a,
            common_a,
            common_a,
            common_b,
            common_b,
            common_a,
            common_a,
            common_a,
            common_b,
            rare,
        ]
    )
    config = MorphologyCoarseningConfig(
        keep_ratio=0.4,
        strategy="morphology_topology",
        merge_similarity=0.95,
        connectivity=4,
        rarity_weight=4.0,
        projection_dim=4,
    )

    result = coarsen_wsi_tiles(features, coords, config)

    assert result.selected_indices.numel() == 4
    assert 9 in result.selected_indices.tolist()
    assert result.diagnostics["n_regions"] == 3
    assert result.diagnostics["region_coverage"] == 1.0
    assert result.diagnostics["rare_region_coverage"] == 1.0


def test_boundary_tiles_are_prioritized_at_morphology_interface() -> None:
    coords = _grid(6, 1)
    left = torch.tensor([1.0, 0.0])
    right = torch.tensor([0.0, 1.0])
    features = torch.stack([left, left, left, right, right, right])
    config = MorphologyCoarseningConfig(
        keep_ratio=0.5,
        strategy="morphology_topology",
        merge_similarity=0.9,
        connectivity=4,
        boundary_fraction=1.0,
        projection_dim=2,
    )

    result = coarsen_wsi_tiles(features, coords, config)

    assert result.boundary_scores is not None
    highest = torch.topk(result.boundary_scores, k=2).indices.tolist()
    assert set(highest) == {2, 3}
    assert {2, 3}.intersection(result.selected_indices.tolist())


def test_attention_is_diagnostic_only() -> None:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(16, 6, generator=generator)
    coords = _grid(4, 4)
    config = MorphologyCoarseningConfig(
        keep_ratio=0.25,
        strategy="morphology_topology",
        merge_quantile=0.25,
        seed=5,
    )
    attention_a = torch.softmax(torch.arange(16, dtype=torch.float32), dim=0)
    attention_b = torch.flip(attention_a, dims=(0,))

    result_a = coarsen_wsi_tiles(features, coords, config, attention=attention_a)
    result_b = coarsen_wsi_tiles(features, coords, config, attention=attention_b)

    assert torch.equal(result_a.selected_indices, result_b.selected_indices)
    assert result_a.diagnostics["attention_mass_retained"] != result_b.diagnostics[
        "attention_mass_retained"
    ]
