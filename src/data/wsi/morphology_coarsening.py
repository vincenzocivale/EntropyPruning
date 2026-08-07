"""Morphology- and topology-preserving tile selection for WSI bags.

The selectors in this module operate *after* tile feature extraction and before a
WSI encoder. They never alter EAF or the tile encoder. The proposed
``morphology_topology`` strategy removes inter-tile redundancy while attempting
to preserve spatially connected morphology, region boundaries, heterogeneous
regions, and rare regions.

Only PyTorch is required. The implementation is intentionally training-free so
that the biological prior can be validated before introducing a learned policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


COARSENING_STRATEGIES = (
    "random",
    "spatial_fps",
    "morphology_fps",
    "morphology_topology",
)


@dataclass(frozen=True)
class MorphologyCoarseningConfig:
    """Configuration for WSI tile coarsening.

    ``merge_similarity`` takes precedence over ``merge_quantile``. When no
    absolute threshold is supplied, the topology strategy merges neighboring
    tiles whose cosine similarity is at or above the per-slide quantile.
    """

    keep_ratio: float
    strategy: str = "morphology_topology"
    seed: int = 0

    spatial_mode: str = "auto"  # auto, grid, knn
    connectivity: int = 8
    spatial_k: int = 8
    knn_chunk_size: int = 512

    merge_similarity: float | None = None
    merge_quantile: float = 0.25
    min_region_size: int = 1

    size_exponent: float = 0.5
    heterogeneity_weight: float = 1.0
    boundary_weight: float = 1.0
    rarity_weight: float = 2.0
    boundary_fraction: float = 0.35

    projection_dim: int = 32
    max_fps_iterations: int = 512
    diagnostic_top_fraction: float = 0.10
    diagnostic_chunk_size: int = 512

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        if self.strategy not in COARSENING_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {COARSENING_STRATEGIES}; got {self.strategy!r}."
            )
        if self.spatial_mode not in ("auto", "grid", "knn"):
            raise ValueError("spatial_mode must be 'auto', 'grid', or 'knn'.")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")
        if self.spatial_k <= 0:
            raise ValueError("spatial_k must be positive.")
        if self.knn_chunk_size <= 0:
            raise ValueError("knn_chunk_size must be positive.")
        if self.merge_similarity is not None and not -1.0 <= self.merge_similarity <= 1.0:
            raise ValueError("merge_similarity must be in [-1, 1] when provided.")
        if not 0.0 <= self.merge_quantile <= 1.0:
            raise ValueError("merge_quantile must be in [0, 1].")
        if self.min_region_size <= 0:
            raise ValueError("min_region_size must be positive.")
        if not 0.0 < self.size_exponent <= 1.0:
            raise ValueError("size_exponent must be in (0, 1].")
        for name in (
            "heterogeneity_weight",
            "boundary_weight",
            "rarity_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0.0 <= self.boundary_fraction <= 1.0:
            raise ValueError("boundary_fraction must be in [0, 1].")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")
        if self.max_fps_iterations <= 0:
            raise ValueError("max_fps_iterations must be positive.")
        if not 0.0 < self.diagnostic_top_fraction <= 1.0:
            raise ValueError("diagnostic_top_fraction must be in (0, 1].")
        if self.diagnostic_chunk_size <= 0:
            raise ValueError("diagnostic_chunk_size must be positive.")


@dataclass(frozen=True)
class CoarseningResult:
    """Selection result for one slide."""

    selected_indices: torch.Tensor
    diagnostics: dict[str, Any]
    region_ids: torch.Tensor | None = None
    boundary_scores: torch.Tensor | None = None


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def coarsen_wsi_tiles(
    tile_features: torch.Tensor,
    coords: torch.Tensor | None,
    config: MorphologyCoarseningConfig,
    *,
    attention: torch.Tensor | None = None,
) -> CoarseningResult:
    """Select a fixed-budget subset of WSI tile tokens.

    Args:
        tile_features: Tile embeddings with shape ``[N, D]``.
        coords: Tile coordinates with shape ``[N, 2]`` or ``[N, 4]``.
        config: Selection configuration.
        attention: Optional full-slide attention target used only for
            diagnostics; it never influences selection.
    """

    features = _validate_features(tile_features)
    coords_xy = _validate_coords(coords, features.shape[0])
    target_attention = _validate_attention(attention, features.shape[0])

    n_tiles = int(features.shape[0])
    n_keep = min(n_tiles, max(1, math.ceil(n_tiles * config.keep_ratio)))
    projected = _project_features(features, config.projection_dim, config.seed)

    if n_keep == n_tiles:
        selected = torch.arange(n_tiles, dtype=torch.long)
        diagnostics = _common_diagnostics(
            projected,
            coords_xy,
            selected,
            target_attention,
            config,
        )
        if coords_xy is not None:
            diagnostics.update(
                _diagnose_topology(features, coords_xy, selected, config)
            )
        return CoarseningResult(selected_indices=selected, diagnostics=diagnostics)

    if config.strategy == "random":
        generator = torch.Generator().manual_seed(config.seed)
        selected = torch.randperm(n_tiles, generator=generator)[:n_keep]
        selected = torch.sort(selected).values
        diagnostics = _common_diagnostics(
            projected,
            coords_xy,
            selected,
            target_attention,
            config,
        )
        if coords_xy is not None:
            diagnostics.update(
                _diagnose_topology(features, coords_xy, selected, config)
            )
        return CoarseningResult(selected_indices=selected, diagnostics=diagnostics)

    if config.strategy == "spatial_fps":
        if coords_xy is None:
            raise ValueError("spatial_fps requires coordinates.")
        points = _normalize_points(coords_xy.to(torch.float32))
        selected = _farthest_point_selection(
            points,
            n_keep,
            max_iterations=config.max_fps_iterations,
        )
        selected = torch.sort(selected).values
        diagnostics = _common_diagnostics(
            projected,
            coords_xy,
            selected,
            target_attention,
            config,
        )
        diagnostics.update(
            _diagnose_topology(features, coords_xy, selected, config)
        )
        return CoarseningResult(selected_indices=selected, diagnostics=diagnostics)

    if config.strategy == "morphology_fps":
        selected = _farthest_point_selection(
            projected,
            n_keep,
            max_iterations=config.max_fps_iterations,
        )
        selected = torch.sort(selected).values
        diagnostics = _common_diagnostics(
            projected,
            coords_xy,
            selected,
            target_attention,
            config,
        )
        if coords_xy is not None:
            diagnostics.update(
                _diagnose_topology(features, coords_xy, selected, config)
            )
        return CoarseningResult(selected_indices=selected, diagnostics=diagnostics)

    if coords_xy is None:
        raise ValueError("morphology_topology requires coordinates.")

    edges, spatial_diagnostics = build_spatial_edges(
        coords_xy,
        mode=config.spatial_mode,
        connectivity=config.connectivity,
        spatial_k=config.spatial_k,
        chunk_size=config.knn_chunk_size,
    )
    if edges.numel() == 0:
        # Degenerate slides (for example one-dimensional malformed coordinates)
        # still receive a deterministic morphology-only fallback.
        selected = _farthest_point_selection(
            projected,
            n_keep,
            max_iterations=config.max_fps_iterations,
        )
        selected = torch.sort(selected).values
        diagnostics = _common_diagnostics(
            projected,
            coords_xy,
            selected,
            target_attention,
            config,
        )
        diagnostics.update(spatial_diagnostics)
        diagnostics.update(
            {
                "n_regions": n_tiles,
                "region_coverage": n_keep / n_tiles,
                "rare_region_coverage": n_keep / n_tiles,
                "merge_threshold": None,
                "topology_fallback": "morphology_fps_no_edges",
            }
        )
        return CoarseningResult(selected_indices=selected, diagnostics=diagnostics)

    edge_similarities = (features[edges[:, 0]] * features[edges[:, 1]]).sum(dim=1)
    boundary_scores = _boundary_scores(n_tiles, edges, edge_similarities)
    if config.merge_similarity is None:
        merge_threshold = float(torch.quantile(edge_similarities, config.merge_quantile))
    else:
        merge_threshold = float(config.merge_similarity)

    region_ids = _connected_regions(
        n_tiles,
        edges,
        edge_similarities >= merge_threshold,
    )
    regions = _region_index_lists(region_ids)
    metrics = _region_metrics(features, region_ids, regions, boundary_scores)
    budgets = _allocate_region_budgets(
        region_sizes=metrics["size"],
        heterogeneity=metrics["heterogeneity"],
        boundary=metrics["boundary"],
        rarity=metrics["rarity"],
        n_keep=n_keep,
        config=config,
    )

    selected_parts: list[torch.Tensor] = []
    for region_index, region_members in enumerate(regions):
        budget = int(budgets[region_index])
        if budget <= 0:
            continue
        selected_parts.append(
            _select_within_region(
                region_members,
                projected,
                boundary_scores,
                budget,
                boundary_fraction=config.boundary_fraction,
                max_fps_iterations=config.max_fps_iterations,
            )
        )

    selected = torch.cat(selected_parts) if selected_parts else torch.empty(0, dtype=torch.long)
    selected = torch.unique(selected, sorted=True)
    if selected.numel() != n_keep:
        selected = _repair_budget(selected, projected, n_keep)

    diagnostics = _common_diagnostics(
        projected,
        coords_xy,
        selected,
        target_attention,
        config,
    )
    selected_regions = torch.unique(region_ids[selected])
    significant = metrics["size"] >= config.min_region_size
    significant_ids = torch.nonzero(significant, as_tuple=False).flatten()
    if significant_ids.numel() > 0:
        covered_significant = torch.isin(significant_ids, selected_regions).to(torch.float32).mean()
        significant_coverage = float(covered_significant)
    else:
        significant_coverage = 1.0

    n_rare = max(1, math.ceil(len(regions) * config.diagnostic_top_fraction))
    rare_regions = torch.topk(metrics["rarity"], k=min(n_rare, len(regions))).indices
    rare_coverage = float(torch.isin(rare_regions, selected_regions).to(torch.float32).mean())

    top_boundary_n = max(
        1, math.ceil(n_tiles * config.diagnostic_top_fraction)
    )
    top_boundary = torch.topk(
        boundary_scores, k=min(top_boundary_n, n_tiles)
    ).indices
    diagnostics["boundary_top_recall"] = float(
        torch.isin(top_boundary, selected).to(torch.float32).mean()
    )
    diagnostics.update(spatial_diagnostics)
    diagnostics.update(
        {
            "n_regions": len(regions),
            "n_significant_regions": int(significant.sum()),
            "region_coverage": float(selected_regions.numel() / len(regions)),
            "significant_region_coverage": significant_coverage,
            "rare_region_coverage": rare_coverage,
            "merge_threshold": merge_threshold,
            "mean_region_size": float(metrics["size"].to(torch.float32).mean()),
            "max_region_size": int(metrics["size"].max()),
            "mean_region_heterogeneity": float(metrics["heterogeneity"].mean()),
            "mean_region_boundary": float(metrics["boundary"].mean()),
            "mean_region_rarity": float(metrics["rarity"].mean()),
        }
    )
    return CoarseningResult(
        selected_indices=selected,
        diagnostics=diagnostics,
        region_ids=region_ids,
        boundary_scores=boundary_scores,
    )


def build_spatial_edges(
    coords: torch.Tensor,
    *,
    mode: str = "auto",
    connectivity: int = 8,
    spatial_k: int = 8,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build undirected spatial edges from WSI coordinates."""

    coords_xy = _validate_coords(coords, coords.shape[0])
    assert coords_xy is not None
    if mode not in ("auto", "grid", "knn"):
        raise ValueError("mode must be 'auto', 'grid', or 'knn'.")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    if mode in ("auto", "grid"):
        grid_edges, diagnostics = _grid_edges(coords_xy, connectivity=connectivity)
        # A regular tissue grid normally has O(N) edges. Auto mode falls back
        # when quantization produced too few relationships.
        minimum_reasonable_edges = max(1, coords_xy.shape[0] // 3)
        grid_is_reliable = (
            grid_edges.shape[0] >= minimum_reasonable_edges
            and float(diagnostics["grid_quantization_error"]) <= 0.10
        )
        if mode == "grid" or grid_is_reliable:
            diagnostics["spatial_edge_mode"] = "grid"
            diagnostics["neighbor_edge_count"] = int(grid_edges.shape[0])
            return grid_edges, diagnostics

    knn_edges = _knn_edges(coords_xy, k=spatial_k, chunk_size=chunk_size)
    return knn_edges, {
        "spatial_edge_mode": "knn",
        "neighbor_edge_count": int(knn_edges.shape[0]),
        "grid_quantization_error": None,
        "grid_x_step": None,
        "grid_y_step": None,
    }


def _diagnose_topology(
    features: torch.Tensor,
    coords: torch.Tensor,
    selected: torch.Tensor,
    config: MorphologyCoarseningConfig,
) -> dict[str, Any]:
    """Evaluate any selected subset against the same tissue-region prior."""

    edges, spatial_diagnostics = build_spatial_edges(
        coords,
        mode=config.spatial_mode,
        connectivity=config.connectivity,
        spatial_k=config.spatial_k,
        chunk_size=config.knn_chunk_size,
    )
    diagnostics: dict[str, Any] = dict(spatial_diagnostics)
    if edges.numel() == 0:
        diagnostics.update(
            {
                "n_regions": int(features.shape[0]),
                "n_significant_regions": int(features.shape[0]),
                "region_coverage": float(selected.numel() / features.shape[0]),
                "significant_region_coverage": float(
                    selected.numel() / features.shape[0]
                ),
                "rare_region_coverage": float(selected.numel() / features.shape[0]),
                "boundary_top_recall": None,
                "merge_threshold": None,
            }
        )
        return diagnostics

    edge_similarities = (features[edges[:, 0]] * features[edges[:, 1]]).sum(dim=1)
    boundary_scores = _boundary_scores(features.shape[0], edges, edge_similarities)
    merge_threshold = (
        float(torch.quantile(edge_similarities, config.merge_quantile))
        if config.merge_similarity is None
        else float(config.merge_similarity)
    )
    region_ids = _connected_regions(
        int(features.shape[0]),
        edges,
        edge_similarities >= merge_threshold,
    )
    regions = _region_index_lists(region_ids)
    metrics = _region_metrics(features, region_ids, regions, boundary_scores)
    selected_regions = torch.unique(region_ids[selected])
    significant = metrics["size"] >= config.min_region_size
    significant_ids = torch.nonzero(significant, as_tuple=False).flatten()
    significant_coverage = (
        float(torch.isin(significant_ids, selected_regions).to(torch.float32).mean())
        if significant_ids.numel()
        else 1.0
    )
    n_rare = max(1, math.ceil(len(regions) * config.diagnostic_top_fraction))
    rare_regions = torch.topk(metrics["rarity"], k=min(n_rare, len(regions))).indices
    top_boundary_n = max(
        1, math.ceil(features.shape[0] * config.diagnostic_top_fraction)
    )
    top_boundary = torch.topk(
        boundary_scores, k=min(top_boundary_n, features.shape[0])
    ).indices
    diagnostics.update(
        {
            "n_regions": len(regions),
            "n_significant_regions": int(significant.sum()),
            "region_coverage": float(selected_regions.numel() / len(regions)),
            "significant_region_coverage": significant_coverage,
            "rare_region_coverage": float(
                torch.isin(rare_regions, selected_regions).to(torch.float32).mean()
            ),
            "boundary_top_recall": float(
                torch.isin(top_boundary, selected).to(torch.float32).mean()
            ),
            "merge_threshold": merge_threshold,
        }
    )
    return diagnostics


def config_to_dict(config: MorphologyCoarseningConfig) -> dict[str, Any]:
    """Return a JSON-serializable configuration mapping."""

    return asdict(config)


def _validate_features(tile_features: torch.Tensor) -> torch.Tensor:
    if not isinstance(tile_features, torch.Tensor):
        raise TypeError("tile_features must be a torch.Tensor.")
    if tile_features.ndim != 2:
        raise ValueError(
            "tile_features must have shape [n_tiles, feature_dim]; "
            f"got {tuple(tile_features.shape)}."
        )
    if tile_features.shape[0] <= 0 or tile_features.shape[1] <= 0:
        raise ValueError("tile_features must be non-empty in both dimensions.")
    if not torch.is_floating_point(tile_features):
        raise TypeError("tile_features must be floating point.")
    if not torch.isfinite(tile_features).all():
        raise ValueError("tile_features contain NaN or Inf.")
    return F.normalize(tile_features.detach().cpu().to(torch.float32), dim=1)


def _validate_coords(coords: torch.Tensor | None, n_tiles: int) -> torch.Tensor | None:
    if coords is None:
        return None
    if not isinstance(coords, torch.Tensor):
        raise TypeError("coords must be a torch.Tensor when provided.")
    if coords.ndim != 2 or coords.shape[0] != n_tiles or coords.shape[1] not in (2, 4):
        raise ValueError(
            "coords must have shape [n_tiles, 2] or [n_tiles, 4]; "
            f"got {tuple(coords.shape)}."
        )
    coords_xy = coords[:, :2].detach().cpu().to(torch.float64)
    if not torch.isfinite(coords_xy).all():
        raise ValueError("coords contain NaN or Inf.")
    return coords_xy


def _validate_attention(attention: torch.Tensor | None, n_tiles: int) -> torch.Tensor | None:
    if attention is None:
        return None
    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a torch.Tensor when provided.")
    if attention.ndim != 1 or attention.shape[0] != n_tiles:
        raise ValueError(f"attention must have shape [{n_tiles}].")
    values = attention.detach().cpu().to(torch.float32)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("attention must be finite and non-negative.")
    return values


def _project_features(features: torch.Tensor, projection_dim: int, seed: int) -> torch.Tensor:
    if features.shape[1] <= projection_dim:
        return F.normalize(features, dim=1)
    generator = torch.Generator().manual_seed(seed + 17_071)
    projection = torch.randn(
        features.shape[1],
        projection_dim,
        generator=generator,
        dtype=features.dtype,
    ) / math.sqrt(projection_dim)
    return F.normalize(features @ projection, dim=1)


def _axis_step(values: torch.Tensor) -> float:
    unique = torch.unique(values).sort().values
    if unique.numel() <= 1:
        return 1.0
    differences = torch.diff(unique)
    positive = differences[differences > 0]
    if positive.numel() == 0:
        return 1.0
    return float(positive.min())


def _grid_edges(coords: torch.Tensor, *, connectivity: int) -> tuple[torch.Tensor, dict[str, Any]]:
    x_step = _axis_step(coords[:, 0])
    y_step = _axis_step(coords[:, 1])
    origin = coords.min(dim=0).values
    grid = torch.empty_like(coords, dtype=torch.long)
    grid[:, 0] = torch.round((coords[:, 0] - origin[0]) / x_step).to(torch.long)
    grid[:, 1] = torch.round((coords[:, 1] - origin[1]) / y_step).to(torch.long)
    reconstructed = torch.empty_like(coords)
    reconstructed[:, 0] = origin[0] + grid[:, 0].to(coords.dtype) * x_step
    reconstructed[:, 1] = origin[1] + grid[:, 1].to(coords.dtype) * y_step
    scale = max(x_step, y_step, 1e-12)
    quantization_error = float((coords - reconstructed).abs().max() / scale)

    key_to_indices: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(grid.tolist()):
        key_to_indices.setdefault((int(row[0]), int(row[1])), []).append(index)

    offsets = [(1, 0), (0, 1)]
    if connectivity == 8:
        offsets.extend([(1, 1), (1, -1)])
    edge_set: set[tuple[int, int]] = set()
    for key, source_indices in key_to_indices.items():
        for dx, dy in offsets:
            targets = key_to_indices.get((key[0] + dx, key[1] + dy), [])
            for source in source_indices:
                for target in targets:
                    left, right = sorted((source, target))
                    if left != right:
                        edge_set.add((left, right))
    if edge_set:
        edges = torch.tensor(sorted(edge_set), dtype=torch.long)
    else:
        edges = torch.empty((0, 2), dtype=torch.long)
    return edges, {
        "grid_quantization_error": quantization_error,
        "grid_x_step": x_step,
        "grid_y_step": y_step,
    }


def _knn_edges(coords: torch.Tensor, *, k: int, chunk_size: int) -> torch.Tensor:
    n_tiles = int(coords.shape[0])
    if n_tiles <= 1:
        return torch.empty((0, 2), dtype=torch.long)
    k = min(k, n_tiles - 1)
    coords_float = coords.to(torch.float32)
    edge_set: set[tuple[int, int]] = set()
    for start in range(0, n_tiles, chunk_size):
        stop = min(start + chunk_size, n_tiles)
        distances = torch.cdist(coords_float[start:stop], coords_float)
        local_rows = torch.arange(stop - start)
        global_rows = torch.arange(start, stop)
        distances[local_rows, global_rows] = torch.inf
        neighbors = torch.topk(distances, k=k, largest=False).indices
        for local_index, neighbor_row in enumerate(neighbors.tolist()):
            source = start + local_index
            for target in neighbor_row:
                left, right = sorted((source, int(target)))
                if left != right:
                    edge_set.add((left, right))
    return torch.tensor(sorted(edge_set), dtype=torch.long)


def _boundary_scores(
    n_tiles: int,
    edges: torch.Tensor,
    similarities: torch.Tensor,
) -> torch.Tensor:
    dissimilarity = (1.0 - similarities).clamp_min(0.0)
    sums = torch.zeros(n_tiles, dtype=torch.float32)
    counts = torch.zeros(n_tiles, dtype=torch.float32)
    for column in (0, 1):
        indices = edges[:, column]
        sums.scatter_add_(0, indices, dissimilarity)
        counts.scatter_add_(0, indices, torch.ones_like(dissimilarity))
    return sums / counts.clamp_min(1.0)


def _connected_regions(
    n_tiles: int,
    edges: torch.Tensor,
    merge_mask: torch.Tensor,
) -> torch.Tensor:
    union_find = _UnionFind(n_tiles)
    for edge, should_merge in zip(edges.tolist(), merge_mask.tolist()):
        if should_merge:
            union_find.union(int(edge[0]), int(edge[1]))
    root_to_region: dict[int, int] = {}
    region_ids = torch.empty(n_tiles, dtype=torch.long)
    for index in range(n_tiles):
        root = union_find.find(index)
        if root not in root_to_region:
            root_to_region[root] = len(root_to_region)
        region_ids[index] = root_to_region[root]
    return region_ids


def _region_index_lists(region_ids: torch.Tensor) -> list[torch.Tensor]:
    n_regions = int(region_ids.max()) + 1
    return [
        torch.nonzero(region_ids == region, as_tuple=False).flatten()
        for region in range(n_regions)
    ]


def _region_metrics(
    features: torch.Tensor,
    region_ids: torch.Tensor,
    regions: list[torch.Tensor],
    boundary_scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sizes = torch.tensor([members.numel() for members in regions], dtype=torch.long)
    centroids = []
    heterogeneity = []
    boundary = []
    for members in regions:
        centroid = F.normalize(features[members].mean(dim=0, keepdim=True), dim=1).squeeze(0)
        centroids.append(centroid)
        heterogeneity.append(1.0 - (features[members] @ centroid).mean())
        boundary.append(boundary_scores[members].mean())
    centroid_tensor = torch.stack(centroids)
    rarity = _nearest_other_distance(centroid_tensor)
    return {
        "size": sizes,
        "centroid": centroid_tensor,
        "heterogeneity": torch.stack(heterogeneity).to(torch.float32),
        "boundary": torch.stack(boundary).to(torch.float32),
        "rarity": rarity,
        "region_ids": region_ids,
    }


def _nearest_other_distance(centroids: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    n_regions = int(centroids.shape[0])
    if n_regions <= 1:
        return torch.zeros(n_regions, dtype=torch.float32)
    result = torch.empty(n_regions, dtype=torch.float32)
    for start in range(0, n_regions, chunk_size):
        stop = min(start + chunk_size, n_regions)
        similarities = centroids[start:stop] @ centroids.T
        rows = torch.arange(stop - start)
        columns = torch.arange(start, stop)
        similarities[rows, columns] = -torch.inf
        result[start:stop] = 1.0 - similarities.max(dim=1).values
    return result.clamp_min(0.0)


def _minmax(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float32)
    minimum = values.min()
    maximum = values.max()
    if float(maximum - minimum) <= 1e-12:
        return torch.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _allocate_region_budgets(
    *,
    region_sizes: torch.Tensor,
    heterogeneity: torch.Tensor,
    boundary: torch.Tensor,
    rarity: torch.Tensor,
    n_keep: int,
    config: MorphologyCoarseningConfig,
) -> torch.Tensor:
    n_regions = int(region_sizes.numel())
    size_term = region_sizes.to(torch.float32).pow(config.size_exponent)
    priorities = (1.0 + size_term) * (
        1.0
        + config.heterogeneity_weight * _minmax(heterogeneity)
        + config.boundary_weight * _minmax(boundary)
        + config.rarity_weight * _minmax(rarity)
    )

    budgets = torch.zeros(n_regions, dtype=torch.long)
    if n_regions >= n_keep:
        chosen = torch.topk(priorities, k=n_keep).indices
        budgets[chosen] = 1
        return budgets

    budgets[:] = 1
    remaining = n_keep - n_regions
    capacities = region_sizes - budgets
    while remaining > 0:
        eligible = capacities > 0
        if not eligible.any():
            break
        allocation_scores = priorities / (budgets.to(torch.float32) + 1.0)
        allocation_scores[~eligible] = -torch.inf
        region = int(allocation_scores.argmax())
        budgets[region] += 1
        capacities[region] -= 1
        remaining -= 1
    return budgets


def _select_within_region(
    members: torch.Tensor,
    projected: torch.Tensor,
    boundary_scores: torch.Tensor,
    budget: int,
    *,
    boundary_fraction: float,
    max_fps_iterations: int,
) -> torch.Tensor:
    if budget >= members.numel():
        return members.clone()
    local_points = projected[members]
    centroid = F.normalize(local_points.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    prototype_local = int((local_points @ centroid).argmax())
    selected_local = [prototype_local]

    boundary_budget = min(
        budget - 1,
        int(round((budget - 1) * boundary_fraction)),
    )
    if boundary_budget > 0:
        order = torch.argsort(boundary_scores[members], descending=True)
        for candidate in order.tolist():
            if candidate not in selected_local:
                selected_local.append(int(candidate))
            if len(selected_local) >= 1 + boundary_budget:
                break

    remaining = budget - len(selected_local)
    if remaining > 0:
        extra = _farthest_point_selection(
            local_points,
            budget,
            initial_indices=torch.tensor(selected_local, dtype=torch.long),
            max_iterations=max_fps_iterations,
        )
        selected_local = extra.tolist()
    return members[torch.tensor(selected_local[:budget], dtype=torch.long)]


def _farthest_point_selection(
    points: torch.Tensor,
    k: int,
    *,
    initial_indices: torch.Tensor | None = None,
    max_iterations: int = 512,
) -> torch.Tensor:
    n_points = int(points.shape[0])
    if k >= n_points:
        return torch.arange(n_points, dtype=torch.long)
    if points.ndim != 2:
        raise ValueError("points must be 2D.")

    if initial_indices is None or initial_indices.numel() == 0:
        center = points.mean(dim=0, keepdim=True)
        first = int(torch.cdist(points, center).argmin())
        selected = [first]
    else:
        selected = torch.unique(initial_indices.to(torch.long), sorted=False).tolist()
        if len(selected) > k:
            return torch.tensor(selected[:k], dtype=torch.long)

    selected_mask = torch.zeros(n_points, dtype=torch.bool)
    selected_mask[selected] = True
    distances = torch.full((n_points,), torch.inf, dtype=torch.float32)
    for index in selected:
        distances = torch.minimum(
            distances,
            ((points - points[index]) ** 2).sum(dim=1),
        )
    distances[selected_mask] = -torch.inf

    exact_target = min(k, max_iterations)
    while len(selected) < exact_target:
        candidate = int(distances.argmax())
        selected.append(candidate)
        selected_mask[candidate] = True
        candidate_distance = ((points - points[candidate]) ** 2).sum(dim=1)
        distances = torch.minimum(distances, candidate_distance)
        distances[selected_mask] = -torch.inf

    if len(selected) < k:
        # At large keep budgets, cap the expensive greedy iterations and fill
        # with globally atypical points. This keeps the selector practical on
        # slides with tens of thousands of tiles.
        center = F.normalize(points.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        atypicality = 1.0 - (F.normalize(points, dim=1) @ center)
        atypicality[selected_mask] = -torch.inf
        fill = torch.topk(atypicality, k=k - len(selected)).indices.tolist()
        selected.extend(int(index) for index in fill)
    return torch.tensor(selected, dtype=torch.long)


def _repair_budget(
    selected: torch.Tensor,
    projected: torch.Tensor,
    n_keep: int,
) -> torch.Tensor:
    selected = torch.unique(selected.to(torch.long), sorted=True)
    if selected.numel() > n_keep:
        return selected[:n_keep]
    if selected.numel() == n_keep:
        return selected
    selected_mask = torch.zeros(projected.shape[0], dtype=torch.bool)
    selected_mask[selected] = True
    center = F.normalize(projected.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    atypicality = 1.0 - (projected @ center)
    atypicality[selected_mask] = -torch.inf
    fill = torch.topk(atypicality, k=n_keep - selected.numel()).indices
    return torch.sort(torch.cat([selected, fill])).values


def _normalize_points(points: torch.Tensor) -> torch.Tensor:
    minimum = points.min(dim=0).values
    span = (points.max(dim=0).values - minimum).clamp_min(1e-12)
    return (points - minimum) / span


def _mean_feature_coverage(
    projected: torch.Tensor,
    selected: torch.Tensor,
    *,
    chunk_size: int,
) -> float:
    selected_points = projected[selected]
    maxima = []
    for start in range(0, projected.shape[0], chunk_size):
        stop = min(start + chunk_size, projected.shape[0])
        maxima.append((projected[start:stop] @ selected_points.T).max(dim=1).values)
    return float(torch.cat(maxima).mean())


def _mean_spatial_distance(
    coords: torch.Tensor | None,
    selected: torch.Tensor,
    *,
    chunk_size: int,
) -> float | None:
    if coords is None:
        return None
    normalized = _normalize_points(coords.to(torch.float32))
    selected_points = normalized[selected]
    minima = []
    for start in range(0, normalized.shape[0], chunk_size):
        stop = min(start + chunk_size, normalized.shape[0])
        minima.append(torch.cdist(normalized[start:stop], selected_points).min(dim=1).values)
    return float(torch.cat(minima).mean())


def _common_diagnostics(
    projected: torch.Tensor,
    coords: torch.Tensor | None,
    selected: torch.Tensor,
    attention: torch.Tensor | None,
    config: MorphologyCoarseningConfig,
) -> dict[str, Any]:
    n_tiles = int(projected.shape[0])
    top_n = max(1, math.ceil(n_tiles * config.diagnostic_top_fraction))
    diagnostics: dict[str, Any] = {
        "strategy": config.strategy,
        "requested_keep_ratio": config.keep_ratio,
        "n_tiles_input": n_tiles,
        "n_tiles_kept": int(selected.numel()),
        "effective_keep_ratio": float(selected.numel() / n_tiles),
        "projected_feature_coverage": _mean_feature_coverage(
            projected,
            selected,
            chunk_size=config.diagnostic_chunk_size,
        ),
        "mean_normalized_spatial_distance": _mean_spatial_distance(
            coords,
            selected,
            chunk_size=config.diagnostic_chunk_size,
        ),
    }
    if attention is not None and float(attention.sum()) > 0:
        total = attention.sum()
        retained = attention[selected].sum() / total
        oracle = attention[torch.topk(attention, k=selected.numel()).indices].sum() / total
        diagnostics["attention_mass_retained"] = float(retained)
        diagnostics["oracle_attention_mass_at_k"] = float(oracle)
        diagnostics["relative_attention_mass_retained"] = float(retained / oracle.clamp_min(1e-12))
        attention_top = torch.topk(attention, k=top_n).indices
        diagnostics["attention_top_recall"] = float(
            torch.isin(attention_top, selected).to(torch.float32).mean()
        )
    else:
        diagnostics["attention_mass_retained"] = None
        diagnostics["oracle_attention_mass_at_k"] = None
        diagnostics["relative_attention_mass_retained"] = None
        diagnostics["attention_top_recall"] = None
    return diagnostics
