from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from .io import read_tile_feature_record, read_wsi_output_record
from .utils import json_dump_atomic, stable_seed


@dataclass(frozen=True)
class MajorityAnalysisConfig:
    feature_key: str = "final"
    attention_key: str = "probs"
    attention_reduction: str = "mean"
    knn_values: tuple[int, ...] = (16, 32, 64)
    cluster_values: tuple[int, ...] = (4, 8, 16)
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20)
    n_permutations: int = 200
    discovery_fraction: float = 0.125
    seed: int = 17
    max_tiles_for_clustering_fit: int = 20000


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def _knn_density(values: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k + 1, len(values))
    model = NearestNeighbors(n_neighbors=k_eff, metric="cosine", algorithm="auto", n_jobs=-1)
    model.fit(values)
    distances, _ = model.kneighbors(values, return_distance=True)
    if k_eff <= 1:
        return np.ones(len(values), dtype=np.float32)
    return (1.0 - distances[:, 1:]).mean(axis=1).astype(np.float32)


def _dominant_cluster_metrics(
    values: np.ndarray,
    attention: np.ndarray,
    n_clusters: int,
    seed: int,
    top_fractions: tuple[float, ...],
    max_fit: int,
) -> tuple[dict, list[dict], np.ndarray]:
    n_clusters = max(2, min(n_clusters, len(values)))
    rng = np.random.default_rng(seed)
    if len(values) > max_fit:
        fit_idx = rng.choice(len(values), size=max_fit, replace=False)
        fit_values = values[fit_idx]
    else:
        fit_values = values
    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, max(256, len(fit_values))),
        n_init=10,
    )
    clusterer.fit(fit_values)
    labels = clusterer.predict(values)
    counts = np.bincount(labels, minlength=n_clusters)
    dominant = int(np.argmax(counts))
    mask = labels == dominant
    dominant_fraction = float(mask.mean())
    attention_mass = float(attention[mask].sum())
    enrichment = attention_mass / max(dominant_fraction, 1e-12)
    mean_dom = float(attention[mask].mean())
    mean_other = float(attention[~mask].mean()) if (~mask).any() else np.nan
    metrics = {
        "dominant_cluster": dominant,
        "dominant_fraction": dominant_fraction,
        "dominant_attention_mass": attention_mass,
        "dominant_enrichment": enrichment,
        "dominant_mean_attention": mean_dom,
        "other_mean_attention": mean_other,
        "dominant_mean_attention_ratio": mean_dom / mean_other if np.isfinite(mean_other) and mean_other > 0 else np.nan,
    }
    for fraction in top_fractions:
        n_top = max(1, int(np.ceil(len(values) * fraction)))
        top_idx = np.argpartition(attention, -n_top)[-n_top:]
        observed = float(mask[top_idx].mean())
        metrics[f"top_{int(fraction * 100)}pct_dominant_fraction"] = observed
        metrics[f"top_{int(fraction * 100)}pct_enrichment"] = observed / max(dominant_fraction, 1e-12)

    cluster_rows: list[dict] = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        fraction = float(cluster_mask.mean())
        mass = float(attention[cluster_mask].sum())
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(cluster_mask.sum()),
                "cluster_fraction": fraction,
                "attention_mass": mass,
                "attention_enrichment": mass / max(fraction, 1e-12),
                "mean_attention": float(attention[cluster_mask].mean()),
                "is_dominant": cluster_id == dominant,
            }
        )
    return metrics, cluster_rows, labels


def _permutation_pvalue(
    attention: np.ndarray,
    density: np.ndarray,
    observed_rho: float,
    n_permutations: int,
    seed: int,
) -> float:
    if n_permutations <= 0 or not np.isfinite(observed_rho):
        return np.nan
    rng = np.random.default_rng(seed)
    attention_ranks = _rankdata(attention)
    density_ranks = _rankdata(density)
    attention_centered = attention_ranks - attention_ranks.mean()
    density_centered = density_ranks - density_ranks.mean()
    denom = np.sqrt((attention_centered**2).sum() * (density_centered**2).sum())
    if denom == 0:
        return np.nan
    exceed = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(attention_centered)
        rho = float(np.dot(shuffled, density_centered) / denom)
        exceed += abs(rho) >= abs(observed_rho)
    return (exceed + 1) / (n_permutations + 1)


def analyze_slide(
    feature_path: Path,
    output_path: Path,
    *,
    config: MajorityAnalysisConfig,
    case_id: str | None = None,
    project: str | None = None,
) -> tuple[dict, list[dict]]:
    feature_record = read_tile_feature_record(feature_path)
    output_record = read_wsi_output_record(output_path)
    if feature_record.slide_id != output_record.slide_id:
        raise ValueError(f"Slide mismatch: {feature_record.slide_id} vs {output_record.slide_id}")
    if config.feature_key not in feature_record.embeddings:
        if len(feature_record.embeddings) == 1:
            feature_key = next(iter(feature_record.embeddings))
        else:
            raise KeyError(f"Feature key {config.feature_key!r} unavailable")
    else:
        feature_key = config.feature_key
    if config.attention_key not in output_record.attention:
        raise KeyError(
            f"Attention key {config.attention_key!r} unavailable in {output_path}; "
            f"keys={list(output_record.attention)}. Use a WSI model with native tile attention."
        )

    values = np.asarray(feature_record.embeddings[feature_key])
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    attention = np.asarray(output_record.attention[config.attention_key], dtype=np.float64)
    if attention.ndim > 1:
        if attention.shape[-1] != len(values):
            raise ValueError(
                f"Attention tile axis mismatch: features={len(values)}, attention={attention.shape}."
            )
        axes = tuple(range(attention.ndim - 1))
        if config.attention_reduction == "mean":
            attention = attention.mean(axis=axes)
        elif config.attention_reduction == "max":
            attention = attention.max(axis=axes)
        else:
            raise ValueError("attention_reduction must be 'mean' or 'max'")
    attention = attention.reshape(-1)
    if len(values) != len(attention):
        raise ValueError(f"N mismatch: features={len(values)}, attention={len(attention)}")
    attention = np.clip(attention, 0, None)
    if attention.sum() <= 0:
        raise ValueError("Attention has non-positive total mass")
    attention /= attention.sum()
    normalized = _normalize_rows(values)
    centroid = _normalize_rows(values.mean(axis=0, keepdims=True))[0]
    centroid_similarity = normalized @ centroid
    embedding_norm = np.linalg.norm(values, axis=1)

    row = {
        "slide_id": feature_record.slide_id,
        "case_id": case_id or feature_record.slide_id,
        "project": project,
        "n_tiles": len(values),
        "feature_key": feature_key,
        "attention_key": config.attention_key,
        "attention_entropy": float(-(attention * np.log(np.clip(attention, 1e-12, None))).sum()),
        "effective_tile_count": float(np.exp(-(attention * np.log(np.clip(attention, 1e-12, None))).sum())),
        "rho_attention_centroid_similarity": float(spearmanr(attention, centroid_similarity).statistic),
        "rho_attention_embedding_norm": float(spearmanr(attention, embedding_norm).statistic),
    }
    seed = stable_seed(feature_record.slide_id, config.seed)
    densities: dict[int, np.ndarray] = {}
    for k in config.knn_values:
        density = _knn_density(normalized, k)
        densities[k] = density
        rho = float(spearmanr(attention, density).statistic)
        row[f"rho_attention_knn_k{k}"] = rho
        row[f"p_perm_attention_knn_k{k}"] = _permutation_pvalue(
            attention, density, rho, config.n_permutations, seed + k
        )
        baseline = float(density.mean())
        sd = float(density.std())
        weighted = float(np.dot(attention, density))
        row[f"density_lift_k{k}"] = (weighted - baseline) / sd if sd > 0 else np.nan

    cluster_rows: list[dict] = []
    for n_clusters in config.cluster_values:
        metrics, rows, labels = _dominant_cluster_metrics(
            normalized,
            attention,
            n_clusters,
            seed + n_clusters,
            config.top_fractions,
            config.max_tiles_for_clustering_fit,
        )
        for key, value in metrics.items():
            row[f"k{n_clusters}_{key}"] = value
        primary_density = densities[config.knn_values[len(config.knn_values) // 2]]
        for cluster_row in rows:
            mask = labels == cluster_row["cluster_id"]
            cluster_row.update(
                {
                    "slide_id": feature_record.slide_id,
                    "case_id": case_id or feature_record.slide_id,
                    "project": project,
                    "n_clusters": n_clusters,
                    "mean_knn_density": float(primary_density[mask].mean()),
                }
            )
            cluster_rows.append(cluster_row)
    return row, cluster_rows


def _bootstrap_median_ci(values: np.ndarray, groups: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    mask = np.isfinite(values)
    values, groups = values[mask], groups[mask]
    if len(values) == 0:
        return np.nan, np.nan
    unique_groups = np.unique(groups)
    grouped = {group: values[groups == group] for group in unique_groups}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        draw_values = np.concatenate([grouped[group] for group in sampled])
        draws[index] = np.median(draw_values)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def aggregate_results(per_slide: pd.DataFrame, config: MajorityAnalysisConfig) -> dict:
    primary_k = config.knn_values[len(config.knn_values) // 2]
    primary_clusters = config.cluster_values[len(config.cluster_values) // 2]
    metrics = [
        f"rho_attention_knn_k{primary_k}",
        f"density_lift_k{primary_k}",
        f"k{primary_clusters}_dominant_enrichment",
        f"k{primary_clusters}_top_10pct_enrichment",
    ]
    groups = per_slide["case_id"].astype(str).to_numpy()
    summary: dict = {
        "n_slides": int(len(per_slide)),
        "n_cases": int(per_slide["case_id"].nunique()),
        "primary_knn_k": primary_k,
        "primary_n_clusters": primary_clusters,
        "metrics": {},
    }
    for metric in metrics:
        values = per_slide[metric].to_numpy(dtype=float)
        low, high = _bootstrap_median_ci(values, groups, config.seed)
        summary["metrics"][metric] = {
            "median": float(np.nanmedian(values)),
            "q25": float(np.nanquantile(values, 0.25)),
            "q75": float(np.nanquantile(values, 0.75)),
            "positive_fraction": float(np.nanmean(values > (1.0 if "enrichment" in metric else 0.0))),
            "bootstrap95": [float(low), float(high)],
        }
    rho_ci = summary["metrics"][f"rho_attention_knn_k{primary_k}"]["bootstrap95"]
    enrichment_ci = summary["metrics"][f"k{primary_clusters}_dominant_enrichment"]["bootstrap95"]
    summary["verdict"] = {
        "supports_majority_hypothesis": bool(rho_ci[0] > 0 and enrichment_ci[0] > 1),
        "rule": "lower 95% bootstrap bound of median density correlation > 0 and dominant-cluster enrichment > 1",
    }
    return summary


def run_majority_analysis(
    rows: Iterable[dict],
    *,
    output_dir: Path,
    config: MajorityAnalysisConfig,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_slide_rows: list[dict] = []
    cluster_rows: list[dict] = []
    error_rows: list[dict] = []
    for item in tqdm(list(rows), desc="Attention-majority analysis", unit="slide"):
        try:
            slide_row, slide_clusters = analyze_slide(
                Path(item["feature_path"]),
                Path(item["output_path"]),
                config=config,
                case_id=item.get("case_id"),
                project=item.get("project"),
            )
            split_key = item.get("case_id") or item["slide_id"]
            fraction = stable_seed(str(split_key), config.seed) / float(2**31 - 1)
            slide_row["analysis_split"] = "discovery" if fraction < config.discovery_fraction else "confirmation"
            per_slide_rows.append(slide_row)
            cluster_rows.extend(slide_clusters)
        except Exception as exc:
            error_rows.append({"slide_id": item.get("slide_id"), "error": repr(exc)})

    per_slide = pd.DataFrame(per_slide_rows)
    per_cluster = pd.DataFrame(cluster_rows)
    errors = pd.DataFrame(error_rows)
    per_slide.to_csv(output_dir / "per_slide.csv", index=False)
    per_cluster.to_csv(output_dir / "per_cluster.csv", index=False)
    errors.to_csv(output_dir / "errors.csv", index=False)
    aggregate = {
        "all": aggregate_results(per_slide, config) if not per_slide.empty else {},
        "discovery": aggregate_results(per_slide[per_slide["analysis_split"] == "discovery"], config)
        if not per_slide.empty and (per_slide["analysis_split"] == "discovery").any()
        else {},
        "confirmation": aggregate_results(per_slide[per_slide["analysis_split"] == "confirmation"], config)
        if not per_slide.empty and (per_slide["analysis_split"] == "confirmation").any()
        else {},
        "config": config.__dict__,
    }
    json_dump_atomic(aggregate, output_dir / "aggregate.json")
    return aggregate
