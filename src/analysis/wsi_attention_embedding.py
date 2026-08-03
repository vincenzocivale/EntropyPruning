"""Label-free analysis of WSI attention against tile-level embeddings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def normalize_attention(values: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    """Normalize a 1D attention/logit vector for weighted summaries."""

    x = torch.as_tensor(values, dtype=torch.float32).flatten()
    if x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("attention must be a non-empty finite vector.")

    if mode == "auto":
        mode = "l1" if bool((x >= 0).all()) and float(x.sum()) > 0 else "softmax"
    if mode == "none":
        return x
    if mode == "softmax":
        return torch.softmax(x, dim=0)
    if mode == "l1":
        if bool((x < 0).any()):
            raise ValueError("l1 normalization requires non-negative attention values.")
        total = x.sum()
        if float(total) <= 0:
            raise ValueError("l1 normalization requires a positive sum.")
        return x / total
    if mode == "minmax":
        shifted = x - x.min()
        if float(shifted.max()) == 0:
            return torch.full_like(x, 1.0 / x.numel())
        shifted = shifted / shifted.max()
        return shifted / shifted.sum()
    raise ValueError("attention normalization must be one of: auto, none, softmax, l1, minmax.")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _safe_pearson(_rankdata(x), _rankdata(y))


def _top_indices(values: np.ndarray, fraction: float, *, largest: bool) -> np.ndarray:
    count = max(1, int(math.ceil(values.size * fraction)))
    if count >= values.size:
        return np.arange(values.size)
    if largest:
        return np.argpartition(values, values.size - count)[-count:]
    return np.argpartition(values, count - 1)[:count]


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(len(set(a.tolist()).intersection(b.tolist())) / max(1, len(a)))


def _poly_r2(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    if x.size <= degree or np.unique(x).size <= degree:
        return float("nan")
    design = np.vander(x, N=degree + 1, increasing=False)
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    prediction = design @ coefficients
    denominator = float(np.square(y - y.mean()).sum())
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.square(y - prediction).sum() / denominator)


def _knn_mean_cosine_distance(
    features: torch.Tensor,
    *,
    k: int,
    reference_size: int,
    chunk_size: int,
    seed: int,
) -> torch.Tensor:
    n_tiles = features.shape[0]
    if k <= 0:
        raise ValueError("k must be positive.")
    if n_tiles < 2:
        raise ValueError("kNN distance requires at least two tiles.")
    if reference_size <= 0 or chunk_size <= 0:
        raise ValueError("reference_size and chunk_size must be positive.")
    reference_size = min(reference_size, n_tiles)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if reference_size == n_tiles:
        reference_indices = torch.arange(n_tiles)
    else:
        reference_indices = torch.randperm(n_tiles, generator=generator)[:reference_size]

    normalized = F.normalize(features.to(torch.float32), dim=1)
    reference = normalized[reference_indices]
    effective_k = min(k, max(1, reference_size - 1))
    outputs: list[torch.Tensor] = []

    for start in range(0, n_tiles, chunk_size):
        stop = min(n_tiles, start + chunk_size)
        similarity = normalized[start:stop] @ reference.T
        query_indices = torch.arange(start, stop)
        self_mask = query_indices[:, None] == reference_indices[None, :]
        similarity = similarity.masked_fill(self_mask, -torch.inf)
        nearest = torch.topk(similarity, k=effective_k, dim=1).values
        outputs.append((1.0 - nearest).mean(dim=1))
    return torch.cat(outputs)


def compute_tile_descriptors(
    tile_features: torch.Tensor,
    coords: torch.Tensor | None = None,
    *,
    knn_k: int = 0,
    knn_reference_size: int = 4096,
    knn_chunk_size: int = 2048,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Compute geometry/redundancy descriptors from tile embeddings only."""

    features = torch.as_tensor(tile_features, dtype=torch.float32).cpu()
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("tile_features must have shape [n_tiles, feature_dim].")
    if not torch.isfinite(features).all():
        raise ValueError("tile_features contain NaN or Inf.")

    centroid = features.mean(dim=0, keepdim=True)
    centered = features - centroid
    cosine_similarity = F.cosine_similarity(features, centroid.expand_as(features), dim=1)
    euclidean_distance = centered.norm(dim=1)
    variance = features.var(dim=0, unbiased=False).clamp_min(1e-8)
    diagonal_mahalanobis = (centered.square() / variance).mean(dim=1)

    descriptors: dict[str, torch.Tensor] = {
        "feature_norm": features.norm(dim=1),
        "cosine_similarity_to_centroid": cosine_similarity,
        "cosine_distance_to_centroid": 1.0 - cosine_similarity,
        "euclidean_distance_to_centroid": euclidean_distance,
        "diagonal_mahalanobis_to_centroid": diagonal_mahalanobis,
    }

    if coords is not None:
        xy = torch.as_tensor(coords, dtype=torch.float32).cpu()
        if xy.ndim != 2 or xy.shape[0] != features.shape[0] or xy.shape[1] not in (2, 4):
            raise ValueError("coords must align with features and have 2 or 4 columns.")
        xy = xy[:, :2]
        spatial_distance = (xy - xy.mean(dim=0, keepdim=True)).norm(dim=1)
        scale = spatial_distance.median().clamp_min(1.0)
        descriptors["spatial_distance_to_centroid"] = spatial_distance
        descriptors["spatial_distance_to_centroid_scaled"] = spatial_distance / scale

    if knn_k > 0:
        descriptors["knn_mean_cosine_distance"] = _knn_mean_cosine_distance(
            features,
            k=knn_k,
            reference_size=knn_reference_size,
            chunk_size=knn_chunk_size,
            seed=seed,
        )

    return {name: value.numpy() for name, value in descriptors.items()}


def _attention_probability(raw: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
    if normalized.numel() == 0:
        raise ValueError("attention is empty.")
    if bool((normalized >= 0).all()) and float(normalized.sum()) > 0:
        return normalized / normalized.sum()
    return torch.softmax(raw, dim=0)


def _descriptor_metrics(
    descriptor: np.ndarray,
    attention_raw: np.ndarray,
    attention_probability: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "pearson": _safe_pearson(descriptor, attention_raw),
        "spearman": _safe_spearman(descriptor, attention_raw),
        "mean": float(descriptor.mean()),
        "std": float(descriptor.std()),
        "attention_weighted_mean": float(np.sum(descriptor * attention_probability)),
        "linear_r2": _poly_r2(descriptor, attention_raw, 1),
        "quadratic_r2": _poly_r2(descriptor, attention_raw, 2),
    }
    metrics["quadratic_r2_gain"] = metrics["quadratic_r2"] - metrics["linear_r2"]

    high_attention: dict[str, Any] = {}
    for fraction in _FRACTIONS:
        key = f"top_{int(fraction * 100)}pct"
        attention_top = _top_indices(attention_raw, fraction, largest=True)
        descriptor_high = _top_indices(descriptor, fraction, largest=True)
        descriptor_low = _top_indices(descriptor, fraction, largest=False)
        high_attention[key] = {
            "descriptor_mean": float(descriptor[attention_top].mean()),
            "overlap_high": _overlap(attention_top, descriptor_high),
            "overlap_low": _overlap(attention_top, descriptor_low),
        }
    metrics["high_attention"] = high_attention

    quantile_edges = np.quantile(descriptor, np.linspace(0.0, 1.0, 11))
    quantile_edges = np.unique(quantile_edges)
    bins: list[dict[str, float | int]] = []
    if quantile_edges.size >= 2:
        assignments = np.digitize(descriptor, quantile_edges[1:-1], right=True)
        for index in range(quantile_edges.size - 1):
            mask = assignments == index
            if mask.any():
                bins.append(
                    {
                        "bin": index,
                        "count": int(mask.sum()),
                        "descriptor_mean": float(descriptor[mask].mean()),
                        "attention_mean": float(attention_raw[mask].mean()),
                        "attention_mass": float(attention_probability[mask].sum()),
                    }
                )
    metrics["quantile_bins"] = bins
    return metrics


def analyze_attention_embedding_relation(
    tile_features: torch.Tensor,
    attention: torch.Tensor,
    coords: torch.Tensor | None = None,
    *,
    attention_normalization: str = "auto",
    knn_k: int = 0,
    knn_reference_size: int = 4096,
    knn_chunk_size: int = 2048,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Analyze one slide and return summary metrics plus per-tile arrays."""

    raw = torch.as_tensor(attention, dtype=torch.float32).flatten().cpu()
    if raw.shape[0] != tile_features.shape[0]:
        raise ValueError("attention and tile_features must contain the same number of tiles.")
    normalized = normalize_attention(raw, attention_normalization)
    probability = _attention_probability(raw, normalized)
    descriptors = compute_tile_descriptors(
        tile_features,
        coords,
        knn_k=knn_k,
        knn_reference_size=knn_reference_size,
        knn_chunk_size=knn_chunk_size,
        seed=seed,
    )

    probability_np = probability.numpy()
    raw_np = raw.numpy()
    entropy = float(-(probability * probability.clamp_min(1e-12).log()).sum())
    sorted_probability = torch.sort(probability, descending=True).values

    attention_metrics: dict[str, Any] = {
        "n_tiles": int(raw.numel()),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "raw_std": float(raw.std(unbiased=False)),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(max(2, raw.numel())),
        "effective_tiles": float(math.exp(entropy)),
    }
    for fraction in _FRACTIONS:
        count = max(1, int(math.ceil(raw.numel() * fraction)))
        attention_metrics[f"mass_top_{int(fraction * 100)}pct"] = float(
            sorted_probability[:count].sum()
        )

    descriptor_metrics = {
        name: _descriptor_metrics(values, raw_np, probability_np)
        for name, values in descriptors.items()
    }
    summary = {
        "attention": attention_metrics,
        "descriptors": descriptor_metrics,
    }
    tile_arrays = {
        "attention_raw": raw_np,
        "attention_probability": probability_np,
        **descriptors,
    }
    if coords is not None:
        tile_arrays["coords"] = torch.as_tensor(coords).cpu().numpy()
    return summary, tile_arrays


def flatten_numeric_metrics(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float | int]:
    """Flatten numeric leaves while excluding variable-length bin tables."""

    output: dict[str, float | int] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if key == "quantile_bins":
            continue
        if isinstance(child, Mapping):
            output.update(flatten_numeric_metrics(child, prefix=name))
        elif isinstance(child, (int, float, np.integer, np.floating)):
            output[name] = int(child) if isinstance(child, (int, np.integer)) else float(child)
    return output


def aggregate_slide_metrics(rows: list[Mapping[str, float | int]]) -> dict[str, Any]:
    """Aggregate numeric slide metrics with mean, median, and finite count."""

    keys = sorted({key for row in rows for key in row})
    aggregate: dict[str, Any] = {"n_slides": len(rows), "metrics": {}}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            aggregate["metrics"][key] = {
                "count": int(finite.size),
                "mean": float(finite.mean()),
                "median": float(np.median(finite)),
                "std": float(finite.std()),
            }
    return aggregate
