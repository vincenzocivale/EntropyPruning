from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .registry import load_artifacts
from .utils import json_dump_atomic, stable_seed


@dataclass(frozen=True)
class FeatureViewSpec:
    name: str
    manifest: Path
    feature_set_id: str | None
    feature_key: str
    artifact_type: str | None = "tile_features"


@dataclass(frozen=True)
class SignalDiscoveryConfig:
    attention_key: str = "global_to_tiles_mass_share"
    target_layer: int = -1
    n_head_groups: int = 6
    positive_fraction: float = 0.10
    negative_fraction: float = 0.50
    retention_fractions: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    seed: int = 17
    max_train_tiles: int = 100_000
    max_tiles_per_slide: int = 2048
    projection_dim: int = 256
    knn_k: int = 32
    knn_backend: str = "auto"
    prototype_count: int = 16
    prototype_temperature: float = 10.0
    slide_prototype_count: int = 16
    max_slide_fit_tiles: int = 4096
    spatial_neighbors: int = 0
    bootstrap_replicates: int = 500
    methods: tuple[str, ...] = ("knn", "linear", "prototype", "context_linear")

    def validate(self) -> None:
        if not 0 < self.positive_fraction < 0.5:
            raise ValueError("positive_fraction must be in (0, 0.5)")
        if not self.positive_fraction < self.negative_fraction < 1:
            raise ValueError("negative_fraction must exceed positive_fraction and be < 1")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0,1)")
        if not 0 <= self.validation_fraction < 1 - self.train_fraction:
            raise ValueError("validation_fraction leaves no test split")
        if self.n_head_groups < 1:
            raise ValueError("n_head_groups must be positive")
        if self.knn_backend not in {"auto", "faiss", "sklearn"}:
            raise ValueError("knn_backend must be auto, faiss, or sklearn")
        allowed = {"knn", "linear", "prototype", "context_linear"}
        unknown = set(self.methods) - allowed
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")


@dataclass
class RandomProjector:
    input_dim: int
    output_dim: int
    seed: int
    matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.output_dim <= 0 or self.output_dim >= self.input_dim:
            self.output_dim = self.input_dim
            self.matrix = None
        elif self.matrix is None:
            rng = np.random.default_rng(self.seed)
            self.matrix = (
                rng.standard_normal((self.input_dim, self.output_dim), dtype=np.float32)
                / math.sqrt(self.output_dim)
            )

    def transform(self, values: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"Expected [N,{self.input_dim}], got {values.shape}")
        if self.matrix is None:
            return values.copy()
        result = np.empty((len(values), self.output_dim), dtype=np.float32)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            result[start:stop] = values[start:stop] @ self.matrix
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            input_dim=np.int64(self.input_dim),
            output_dim=np.int64(self.output_dim),
            seed=np.int64(self.seed),
            matrix=np.empty((0, 0), dtype=np.float32) if self.matrix is None else self.matrix,
        )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def _resolve_manifest_paths(table: pd.DataFrame, manifest: Path) -> pd.DataFrame:
    table = table.copy()
    def resolve(value: str) -> str:
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        manifest_relative = manifest.parent / path
        if manifest_relative.exists():
            return str(manifest_relative.resolve())
        if path.exists():
            return str(path.resolve())
        return str(manifest_relative.resolve())

    table["path"] = table["path"].map(resolve)
    return table


def load_attention_manifest(path: Path, model_filter: str | None = None) -> pd.DataFrame:
    path = Path(path)
    table = pd.read_csv(path)
    if "slide_id" not in table or "path" not in table:
        raise ValueError(f"Attention manifest requires slide_id,path: {path}")
    if "status" in table:
        table = table[table["status"].isin(["complete", "available", "valid", "skipped"])]
    if model_filter:
        for column in ("model", "model_name", "model_id"):
            if column in table:
                table = table[table[column].astype(str).str.contains(model_filter, case=False, regex=False)]
                break
    table = _resolve_manifest_paths(table, path)
    if table["slide_id"].duplicated().any():
        duplicates = table.loc[table["slide_id"].duplicated(), "slide_id"].astype(str).tolist()[:10]
        raise ValueError(f"Duplicate attention slide IDs: {duplicates}")
    return table[["slide_id", "path"]].rename(columns={"path": "attention_path"}).reset_index(drop=True)


def load_feature_view(spec: FeatureViewSpec) -> pd.DataFrame:
    table = load_artifacts(
        spec.manifest,
        artifact_type=spec.artifact_type,
        feature_set_id=spec.feature_set_id,
    )
    if "status" in table:
        table = table[table["status"].isin(["complete", "available", "valid", "skipped"])]
    if table["slide_id"].duplicated().any():
        duplicates = table.loc[table["slide_id"].duplicated(), "slide_id"].astype(str).tolist()[:10]
        raise ValueError(f"Duplicate feature slide IDs for {spec.name}: {duplicates}")
    return table[["slide_id", "path"]].rename(columns={"path": f"feature_path__{spec.name}"})


def load_metadata(path: Path | None, slide_ids: Sequence[str]) -> pd.DataFrame:
    if path is None:
        table = pd.DataFrame({"slide_id": list(slide_ids)})
    else:
        table = pd.read_csv(path)
        if "slide_id" not in table:
            raise ValueError("metadata requires slide_id")
        if "patient_id" in table and "case_id" not in table:
            table = table.rename(columns={"patient_id": "case_id"})
        if "tcga_project" in table and "project" not in table:
            table = table.rename(columns={"tcga_project": "project"})
        keep = [column for column in ("slide_id", "case_id", "project") if column in table]
        table = table[keep].drop_duplicates("slide_id")
    if "case_id" not in table:
        table["case_id"] = table["slide_id"].astype(str).map(
            lambda value: "-".join(value.split("-")[:3]) if value.startswith("TCGA-") else value
        )
    if "project" not in table:
        table["project"] = "unknown"
    table["case_id"] = table["case_id"].fillna(table["slide_id"]).astype(str)
    table["project"] = table["project"].fillna("unknown").astype(str)
    return table


def build_analysis_table(
    attention_manifest: Path,
    feature_views: Sequence[FeatureViewSpec],
    metadata_path: Path | None,
    model_filter: str | None = None,
) -> pd.DataFrame:
    table = load_attention_manifest(attention_manifest, model_filter=model_filter)
    metadata = load_metadata(metadata_path, table["slide_id"].astype(str).tolist())
    table = table.merge(metadata, on="slide_id", how="left", validate="one_to_one")
    for spec in feature_views:
        view = load_feature_view(spec)
        table = table.merge(view, on="slide_id", how="inner", validate="one_to_one")
    if table.empty:
        raise RuntimeError("No slide has attention and all requested feature views")
    return table.reset_index(drop=True)


def assign_case_splits(table: pd.DataFrame, config: SignalDiscoveryConfig) -> pd.DataFrame:
    cases = table[["case_id", "project"]].drop_duplicates("case_id").copy()
    assignments: list[dict] = []
    for project, group in cases.groupby("project", sort=True):
        ordered = group.assign(
            _hash=group["case_id"].astype(str).map(lambda case: stable_seed(f"{project}:{case}", config.seed))
        ).sort_values(["_hash", "case_id"])
        n = len(ordered)
        if n == 1:
            n_train, n_val = 1, 0
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train = max(1, int(round(config.train_fraction * n)))
            n_val = max(1, int(round(config.validation_fraction * n)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        for index, row in enumerate(ordered.itertuples(index=False)):
            split = "train" if index < n_train else ("validation" if index < n_train + n_val else "test")
            assignments.append({"case_id": str(row.case_id), "analysis_split": split})
    split_table = pd.DataFrame(assignments)
    result = table.merge(split_table, on="case_id", how="left", validate="many_to_one")
    if result["analysis_split"].isna().any():
        raise RuntimeError("Failed to assign every case to a split")
    return result


def _read_feature_array(path: Path, feature_key: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "embeddings" in handle and feature_key in handle["embeddings"]:
            values = np.asarray(handle["embeddings"][feature_key][:])
        elif feature_key in handle:
            values = np.asarray(handle[feature_key][:])
        elif "features" in handle and feature_key == "final":
            values = np.asarray(handle["features"][:])
        else:
            available = list(handle.keys())
            if "embeddings" in handle:
                available += [f"embeddings/{name}" for name in handle["embeddings"].keys()]
            raise KeyError(f"Feature key {feature_key!r} not found in {path}; available={available}")
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]
        if values.ndim != 2:
            raise ValueError(f"Expected feature matrix [N,D], got {values.shape} in {path}")
        if "coords" not in handle:
            raise KeyError(f"Missing coords in {path}")
        coords = np.asarray(handle["coords"][:])
    if len(coords) != len(values):
        raise ValueError(f"Feature/coordinate mismatch in {path}: {len(values)} vs {len(coords)}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite features in {path}")
    return values.astype(np.float32, copy=False), coords


def _read_attention_array(path: Path, key: str, layer: int) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        dataset_path = key if key.startswith("attention/") else f"attention/{key}"
        if dataset_path not in handle:
            available = list(handle.get("attention", {}).keys()) if "attention" in handle else list(handle.keys())
            raise KeyError(f"Attention key {dataset_path!r} not found in {path}; available={available}")
        values = np.asarray(handle[dataset_path][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:]) if "coords" in handle else None
    if values.ndim == 3:
        layer_index = layer if layer >= 0 else values.shape[0] + layer
        if not 0 <= layer_index < values.shape[0]:
            raise IndexError(f"Layer {layer} outside attention shape {values.shape}")
        values = values[layer_index]
    elif values.ndim == 2:
        pass
    elif values.ndim == 1:
        values = values[None, :]
    else:
        raise ValueError(f"Attention must be [L,H,N], [H,N], or [N], got {values.shape}")
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(f"Attention contains an empty head in {path}")
    values = values / totals
    return values.astype(np.float32, copy=False), coords


def validate_alignment(feature_coords: np.ndarray, attention_coords: np.ndarray | None, n_attention: int, path: Path) -> None:
    if len(feature_coords) != n_attention:
        raise ValueError(f"Tile/attention length mismatch for {path}: {len(feature_coords)} vs {n_attention}")
    if attention_coords is not None:
        if attention_coords.shape != feature_coords.shape or not np.array_equal(attention_coords, feature_coords):
            raise ValueError(f"Coordinate mismatch between feature and attention artifacts for {path}")


def _head_rank_correlation(attention: np.ndarray) -> np.ndarray:
    ranks = np.vstack([rankdata(row, method="average") for row in attention])
    corr = np.corrcoef(ranks)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def fit_head_groups(
    train_rows: pd.DataFrame,
    config: SignalDiscoveryConfig,
    output_dir: Path,
) -> dict:
    correlations: list[np.ndarray] = []
    n_heads: int | None = None
    for row in tqdm(train_rows.itertuples(index=False), total=len(train_rows), desc="Fit TITAN head groups", unit="slide"):
        attention, _ = _read_attention_array(Path(row.attention_path), config.attention_key, config.target_layer)
        if n_heads is None:
            n_heads = attention.shape[0]
        elif attention.shape[0] != n_heads:
            raise ValueError(f"Inconsistent number of heads: {attention.shape[0]} vs {n_heads}")
        correlations.append(_head_rank_correlation(attention))
    if not correlations or n_heads is None:
        raise RuntimeError("No train attention maps available for head grouping")
    median_corr = np.median(np.stack(correlations), axis=0)
    n_groups = min(config.n_head_groups, n_heads)
    distance = np.clip(1.0 - median_corr, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    if n_groups == 1:
        labels = np.zeros(n_heads, dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=n_groups,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance)
    raw_groups = [sorted(np.flatnonzero(labels == label).astype(int).tolist()) for label in np.unique(labels)]
    groups = sorted(raw_groups, key=lambda heads: heads[0])
    payload = {
        "attention_key": config.attention_key,
        "target_layer": config.target_layer,
        "n_heads": n_heads,
        "n_groups": len(groups),
        "groups": groups,
        "fit_split": "train",
        "n_fit_slides": int(len(correlations)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump_atomic(payload, output_dir / "head_groups.json")
    pd.DataFrame(median_corr).to_csv(output_dir / "median_head_rank_correlation.csv", index=False)
    np.save(output_dir / "median_head_rank_correlation.npy", median_corr)
    return payload


def group_attention(attention: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    grouped = []
    for heads in groups:
        if not heads:
            raise ValueError("Empty head group")
        values = attention[np.asarray(heads, dtype=int)].mean(axis=0)
        values = values / np.clip(values.sum(), 1e-12, None)
        grouped.append(values)
    return np.stack(grouped).astype(np.float32)


def attention_percentiles(grouped_attention: np.ndarray) -> np.ndarray:
    n = grouped_attention.shape[1]
    return np.vstack([rankdata(row, method="average") / n for row in grouped_attention]).T.astype(np.float32)


def _sample_indices(n_tiles: int, quota: int, seed: int) -> np.ndarray:
    if quota >= n_tiles:
        return np.arange(n_tiles, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_tiles, size=quota, replace=False)).astype(np.int64)


def _fit_slide_prototypes(values: np.ndarray, n_clusters: int, max_fit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalize_rows(values)
    n_clusters = max(1, min(n_clusters, len(normalized)))
    fit_idx = _sample_indices(len(normalized), min(max_fit, len(normalized)), seed)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(2048, len(fit_idx)),
        n_init=3,
        max_iter=100,
    ).fit(normalized[fit_idx])
    centers = _normalize_rows(model.cluster_centers_)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for start in range(0, len(normalized), 8192):
        labels = np.argmax(normalized[start : start + 8192] @ centers.T, axis=1)
        counts += np.bincount(labels, minlength=n_clusters)
    prevalence = counts.astype(np.float32) / max(1, counts.sum())
    return centers, prevalence


def compute_context_descriptors(
    values: np.ndarray,
    coords: np.ndarray,
    indices: np.ndarray,
    *,
    n_prototypes: int,
    max_fit_tiles: int,
    spatial_neighbors: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    normalized = _normalize_rows(values)
    centers, prevalence = _fit_slide_prototypes(values, n_prototypes, max_fit_tiles, seed)
    similarities = normalized[indices] @ centers.T
    order = np.argsort(similarities, axis=1)
    first = order[:, -1]
    top1 = similarities[np.arange(len(indices)), first]
    top2 = similarities[np.arange(len(indices)), order[:, -2]] if centers.shape[0] > 1 else top1
    assigned_prevalence = prevalence[first]
    logits = similarities / 0.10
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    assignment_entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=1)
    assignment_entropy /= max(math.log(max(2, centers.shape[0])), 1e-8)

    embedding_norm = np.linalg.norm(values[indices], axis=1)
    all_norms = np.linalg.norm(values, axis=1)
    norm_z = (embedding_norm - all_norms.mean()) / max(float(all_norms.std()), 1e-8)

    xy = coords[:, :2]
    minimum = xy.min(axis=0)
    maximum = xy.max(axis=0)
    span = np.clip(maximum - minimum, 1.0, None)
    xy_norm_all = (xy - minimum) / span
    xy_norm = xy_norm_all[indices]
    radial = np.linalg.norm(xy_norm - 0.5, axis=1)
    boundary = np.min(np.column_stack([xy_norm, 1.0 - xy_norm]), axis=1)

    descriptors = [
        top1,
        top1 - top2,
        assigned_prevalence,
        assignment_entropy,
        norm_z,
        xy_norm[:, 0],
        xy_norm[:, 1],
        radial,
        boundary,
    ]

    if spatial_neighbors > 0 and len(values) > 1:
        k = min(spatial_neighbors + 1, len(values))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="kd_tree", n_jobs=-1)
        nn.fit(xy_norm_all)
        neighbor_indices = nn.kneighbors(xy_norm, return_distance=False)
        spatial_similarity = np.empty(len(indices), dtype=np.float32)
        for row_index, (tile_index, neighbors) in enumerate(zip(indices, neighbor_indices)):
            neighbors = neighbors[neighbors != tile_index][:spatial_neighbors]
            spatial_similarity[row_index] = (
                float((normalized[tile_index] @ normalized[neighbors].T).mean()) if len(neighbors) else 1.0
            )
        descriptors.extend([spatial_similarity, 1.0 - spatial_similarity])

    return np.column_stack(descriptors).astype(np.float32)


class KNNPropensity:
    def __init__(self, k: int, backend: str = "auto") -> None:
        self.k = k
        self.backend_requested = backend
        self.backend = ""
        self.labels: np.ndarray | None = None
        self.index = None
        self.train_values: np.ndarray | None = None

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "KNNPropensity":
        values = _normalize_rows(values)
        self.labels = np.asarray(labels, dtype=np.float32)
        use_faiss = self.backend_requested in {"auto", "faiss"}
        if use_faiss:
            try:
                import faiss  # type: ignore

                index = faiss.IndexFlatIP(values.shape[1])
                index.add(np.ascontiguousarray(values, dtype=np.float32))
                self.index = index
                self.backend = "faiss"
                return self
            except ImportError:
                if self.backend_requested == "faiss":
                    raise
        self.index = NearestNeighbors(
            n_neighbors=min(self.k, len(values)), metric="cosine", algorithm="auto", n_jobs=-1
        ).fit(values)
        self.train_values = values
        self.backend = "sklearn"
        return self

    def predict(self, values: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        if self.labels is None or self.index is None:
            raise RuntimeError("KNNPropensity is not fitted")
        values = _normalize_rows(values)
        output = np.empty((len(values), self.labels.shape[1]), dtype=np.float32)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch = np.ascontiguousarray(values[start:stop], dtype=np.float32)
            if self.backend == "faiss":
                similarities, indices = self.index.search(batch, min(self.k, len(self.labels)))
                weights = np.exp(10.0 * (similarities - similarities.max(axis=1, keepdims=True)))
            else:
                distances, indices = self.index.kneighbors(batch, return_distance=True)
                similarities = 1.0 - distances
                weights = np.exp(10.0 * (similarities - similarities.max(axis=1, keepdims=True)))
            weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
            output[start:stop] = np.einsum("bk,bkg->bg", weights, self.labels[indices])
        return output


class LinearPropensity:
    def __init__(self, n_groups: int, positive_fraction: float, negative_fraction: float, seed: int) -> None:
        self.n_groups = n_groups
        self.positive_fraction = positive_fraction
        self.negative_fraction = negative_fraction
        self.seed = seed
        self.scaler = StandardScaler()
        self.models: list[SGDClassifier] = []

    def fit(self, values: np.ndarray, percentiles: np.ndarray) -> "LinearPropensity":
        scaled = self.scaler.fit_transform(np.asarray(values, dtype=np.float32))
        self.models = []
        positive_threshold = 1.0 - self.positive_fraction
        for group in range(self.n_groups):
            target = percentiles[:, group]
            mask = (target >= positive_threshold) | (target <= self.negative_fraction)
            labels = (target[mask] >= positive_threshold).astype(np.int64)
            if len(np.unique(labels)) < 2:
                raise RuntimeError(f"Group {group} has only one class after sampling")
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=2000,
                tol=1e-4,
                random_state=self.seed + group,
                average=True,
            )
            model.fit(scaled[mask], labels)
            self.models.append(model)
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(values, dtype=np.float32))
        return np.column_stack([model.predict_proba(scaled)[:, 1] for model in self.models]).astype(np.float32)


class PrototypePropensity:
    def __init__(self, n_groups: int, n_prototypes: int, temperature: float, seed: int) -> None:
        self.n_groups = n_groups
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        self.seed = seed
        self.positive: list[np.ndarray] = []
        self.negative: list[np.ndarray] = []

    @staticmethod
    def _fit_centers(values: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
        n_clusters = min(n_clusters, len(values))
        if n_clusters < 1:
            raise RuntimeError("Cannot fit prototypes without samples")
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(2048, len(values)),
            n_init=5,
            max_iter=200,
        ).fit(values)
        return _normalize_rows(model.cluster_centers_)

    def fit(self, values: np.ndarray, percentiles: np.ndarray, positive_fraction: float, negative_fraction: float) -> "PrototypePropensity":
        values = _normalize_rows(values)
        positive_threshold = 1.0 - positive_fraction
        self.positive = []
        self.negative = []
        for group in range(self.n_groups):
            target = percentiles[:, group]
            self.positive.append(
                self._fit_centers(values[target >= positive_threshold], self.n_prototypes, self.seed + group)
            )
            self.negative.append(
                self._fit_centers(values[target <= negative_fraction], self.n_prototypes, self.seed + 10_000 + group)
            )
        return self

    def predict(self, values: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        values = _normalize_rows(values)
        output = np.empty((len(values), self.n_groups), dtype=np.float32)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch = values[start:stop]
            for group, (positive, negative) in enumerate(zip(self.positive, self.negative)):
                positive_score = logsumexp(self.temperature * (batch @ positive.T), axis=1)
                negative_score = logsumexp(self.temperature * (batch @ negative.T), axis=1)
                output[start:stop, group] = positive_score - negative_score
        return output

    def save(self, path: Path) -> None:
        payload: dict[str, np.ndarray] = {}
        for group, values in enumerate(self.positive):
            payload[f"positive_{group}"] = values
        for group, values in enumerate(self.negative):
            payload[f"negative_{group}"] = values
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)


def _ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(relevance)))
    predicted = np.argpartition(scores, -k)[-k:]
    predicted = predicted[np.argsort(scores[predicted])[::-1]]
    ideal = np.argpartition(relevance, -k)[-k:]
    ideal = ideal[np.argsort(relevance[ideal])[::-1]]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum((np.power(2.0, relevance[predicted]) - 1.0) * discounts))
    idcg = float(np.sum((np.power(2.0, relevance[ideal]) - 1.0) * discounts))
    return dcg / idcg if idcg > 0 else np.nan


def _top_indices(scores: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, min(len(scores), int(math.ceil(fraction * len(scores)))))
    return np.argpartition(scores, -k)[-k:]


def evaluate_scores(
    scores: np.ndarray,
    grouped_attention: np.ndarray,
    *,
    slide_id: str,
    case_id: str,
    project: str,
    split: str,
    view: str,
    method: str,
    config: SignalDiscoveryConfig,
) -> tuple[list[dict], list[dict]]:
    n_tiles = scores.shape[0]
    n_groups = scores.shape[1]
    percentiles = attention_percentiles(grouped_attention)
    positive_threshold = 1.0 - config.positive_fraction
    group_rows: list[dict] = []
    for group in range(n_groups):
        relevance = percentiles[:, group]
        target_attention = grouped_attention[group]
        positive = relevance >= positive_threshold
        score = scores[:, group]
        rho = spearmanr(score, target_attention).statistic
        row = {
            "slide_id": slide_id,
            "case_id": case_id,
            "project": project,
            "analysis_split": split,
            "view": view,
            "method": method,
            "group": group,
            "n_tiles": n_tiles,
            "spearman": float(rho) if np.isfinite(rho) else np.nan,
            "average_precision_top": float(average_precision_score(positive.astype(int), score)),
            "positive_fraction_realized": float(positive.mean()),
        }
        for retention in config.retention_fractions:
            selected = _top_indices(score, retention)
            recall = float(positive[selected].sum() / max(1, positive.sum()))
            mass = float(target_attention[selected].sum())
            row[f"top_recall_at_{retention:.2f}"] = recall
            row[f"attention_mass_at_{retention:.2f}"] = mass
            row[f"recall_lift_at_{retention:.2f}"] = recall - retention
            row[f"mass_lift_at_{retention:.2f}"] = mass - retention
            row[f"ndcg_at_{retention:.2f}"] = _ndcg_at_k(relevance, score, len(selected))
        group_rows.append(row)

    score_percentiles = np.column_stack([rankdata(scores[:, group]) / n_tiles for group in range(n_groups)])
    coverage_score = score_percentiles.max(axis=1)
    coverage_rows: list[dict] = []
    for retention in config.retention_fractions:
        selected = _top_indices(coverage_score, retention)
        recalls = []
        masses = []
        for group in range(n_groups):
            relevance = percentiles[:, group]
            positive = relevance >= positive_threshold
            recalls.append(float(positive[selected].sum() / max(1, positive.sum())))
            masses.append(float(grouped_attention[group, selected].sum()))
        coverage_rows.append(
            {
                "slide_id": slide_id,
                "case_id": case_id,
                "project": project,
                "analysis_split": split,
                "view": view,
                "method": method,
                "n_tiles": n_tiles,
                "retention": retention,
                "mean_group_top_recall": float(np.mean(recalls)),
                "worst_group_top_recall": float(np.min(recalls)),
                "mean_group_attention_mass": float(np.mean(masses)),
                "worst_group_attention_mass": float(np.min(masses)),
                "mean_recall_lift": float(np.mean(recalls) - retention),
                "worst_recall_lift": float(np.min(recalls) - retention),
                "mean_mass_lift": float(np.mean(masses) - retention),
                "worst_mass_lift": float(np.min(masses) - retention),
            }
        )
    return group_rows, coverage_rows


def _bootstrap_case_median(values: np.ndarray, groups: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    mask = np.isfinite(values)
    values, groups = values[mask], groups[mask]
    if len(values) == 0:
        return np.nan, np.nan
    case_table = pd.DataFrame({"value": values, "case_id": groups})
    case_values = case_table.groupby("case_id", sort=False)["value"].median().to_numpy(dtype=np.float64)
    if len(case_values) == 1 or n_boot <= 0:
        value = float(case_values[0])
        return value, value
    rng = np.random.default_rng(seed)
    # Vectorized case bootstrap. Chunking keeps memory bounded for large replicate counts.
    draws = np.empty(n_boot, dtype=np.float64)
    chunk_size = max(1, min(256, n_boot))
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        indices = rng.integers(0, len(case_values), size=(stop - start, len(case_values)))
        draws[start:stop] = np.median(case_values[indices], axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def aggregate_metrics(table: pd.DataFrame, config: SignalDiscoveryConfig, group_columns: Sequence[str]) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    id_columns = {
        "slide_id",
        "case_id",
        "project",
        "analysis_split",
        "view",
        "method",
        "group",
        "n_tiles",
        "retention",
    }
    metric_columns = [column for column in table.columns if column not in id_columns]
    rows: list[dict] = []
    for keys, frame in table.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        groups = frame["case_id"].astype(str).to_numpy()
        bootstrap_metrics = {
            "spearman",
            "average_precision_top",
            "mean_group_top_recall",
            "worst_group_top_recall",
            "mean_group_attention_mass",
            "worst_group_attention_mass",
        }
        for metric in metric_columns:
            values = frame[metric].to_numpy(dtype=float)
            if metric in bootstrap_metrics:
                low, high = _bootstrap_case_median(
                    values,
                    groups,
                    config.seed + stable_seed(metric, config.seed),
                    config.bootstrap_replicates,
                )
            else:
                low, high = np.nan, np.nan
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_slides": int(np.isfinite(values).sum()),
                    "n_cases": int(frame.loc[np.isfinite(values), "case_id"].nunique()),
                    "median": float(np.nanmedian(values)),
                    "q25": float(np.nanquantile(values, 0.25)),
                    "q75": float(np.nanquantile(values, 0.75)),
                    "bootstrap95_low": float(low),
                    "bootstrap95_high": float(high),
                }
            )
    return pd.DataFrame(rows)


def _collect_training_data(
    rows: pd.DataFrame,
    spec: FeatureViewSpec,
    groups: Sequence[Sequence[int]],
    config: SignalDiscoveryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RandomProjector, list[dict]]:
    train_rows = rows[rows["analysis_split"] == "train"].copy()
    if train_rows.empty:
        raise RuntimeError("No training slides")
    quota = min(config.max_tiles_per_slide, max(1, config.max_train_tiles // len(train_rows)))
    projected_parts: list[np.ndarray] = []
    context_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    sample_manifest: list[dict] = []
    projector: RandomProjector | None = None
    feature_column = f"feature_path__{spec.name}"
    for row in tqdm(train_rows.itertuples(index=False), total=len(train_rows), desc=f"Collect train tiles [{spec.name}]", unit="slide"):
        feature_path = Path(getattr(row, feature_column))
        values, coords = _read_feature_array(feature_path, spec.feature_key)
        attention, attention_coords = _read_attention_array(Path(row.attention_path), config.attention_key, config.target_layer)
        validate_alignment(coords, attention_coords, attention.shape[1], feature_path)
        grouped = group_attention(attention, groups)
        percentiles = attention_percentiles(grouped)
        indices = _sample_indices(len(values), min(quota, len(values)), stable_seed(str(row.slide_id), config.seed))
        if projector is None:
            projector = RandomProjector(values.shape[1], config.projection_dim, config.seed)
        elif values.shape[1] != projector.input_dim:
            raise ValueError(f"Feature dimension changed in view {spec.name}")
        projected_parts.append(projector.transform(values[indices]))
        target_parts.append(percentiles[indices])
        context_parts.append(
            compute_context_descriptors(
                values,
                coords,
                indices,
                n_prototypes=config.slide_prototype_count,
                max_fit_tiles=config.max_slide_fit_tiles,
                spatial_neighbors=config.spatial_neighbors,
                seed=stable_seed(str(row.slide_id), config.seed + 1),
            )
        )
        sample_manifest.append(
            {
                "slide_id": str(row.slide_id),
                "case_id": str(row.case_id),
                "n_tiles": len(values),
                "n_sampled": len(indices),
            }
        )
    assert projector is not None
    projected = np.concatenate(projected_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    context = np.concatenate(context_parts, axis=0)
    if len(projected) > config.max_train_tiles:
        keep = _sample_indices(len(projected), config.max_train_tiles, config.seed + 99)
        projected, targets, context = projected[keep], targets[keep], context[keep]
    return projected, context, targets, projector, sample_manifest


def _fit_methods(
    projected: np.ndarray,
    context: np.ndarray,
    targets: np.ndarray,
    config: SignalDiscoveryConfig,
    model_dir: Path,
) -> dict[str, object]:
    models: dict[str, object] = {}
    model_dir.mkdir(parents=True, exist_ok=True)
    n_groups = targets.shape[1]
    if "knn" in config.methods:
        models["knn"] = KNNPropensity(config.knn_k, config.knn_backend).fit(projected, targets)
    if "linear" in config.methods:
        model = LinearPropensity(
            n_groups, config.positive_fraction, config.negative_fraction, config.seed
        ).fit(projected, targets)
        models["linear"] = model
        joblib.dump(model, model_dir / "linear.joblib", compress=3)
    if "prototype" in config.methods:
        model = PrototypePropensity(
            n_groups, config.prototype_count, config.prototype_temperature, config.seed
        ).fit(projected, targets, config.positive_fraction, config.negative_fraction)
        models["prototype"] = model
        model.save(model_dir / "prototypes.npz")
    if "context_linear" in config.methods:
        combined = np.concatenate([projected, context], axis=1)
        model = LinearPropensity(
            n_groups, config.positive_fraction, config.negative_fraction, config.seed + 1000
        ).fit(combined, targets)
        models["context_linear"] = model
        joblib.dump(model, model_dir / "context_linear.joblib", compress=3)
    return models


def _predict_methods(
    models: dict[str, object],
    projected: np.ndarray,
    context: np.ndarray,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        if name == "context_linear":
            values = np.concatenate([projected, context], axis=1)
        else:
            values = projected
        predictions[name] = model.predict(values)  # type: ignore[attr-defined]
    return predictions


def run_feature_view(
    rows: pd.DataFrame,
    spec: FeatureViewSpec,
    groups: Sequence[Sequence[int]],
    config: SignalDiscoveryConfig,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict:
    view_dir = output_dir / "views" / spec.name
    complete_path = view_dir / "complete.json"
    if resume and complete_path.exists():
        return json.loads(complete_path.read_text())
    view_dir.mkdir(parents=True, exist_ok=True)
    projected, context, targets, projector, sample_manifest = _collect_training_data(rows, spec, groups, config)
    projector.save(view_dir / "models" / "projector.npz")
    pd.DataFrame(sample_manifest).to_csv(view_dir / "training_sample_manifest.csv", index=False)
    json_dump_atomic(
        {
            "n_training_tiles": int(len(projected)),
            "projected_dim": int(projected.shape[1]),
            "context_dim": int(context.shape[1]),
            "target_groups": int(targets.shape[1]),
        },
        view_dir / "training_data_summary.json",
    )
    models = _fit_methods(projected, context, targets, config, view_dir / "models")
    del projected, context, targets

    group_rows: list[dict] = []
    coverage_rows: list[dict] = []
    error_rows: list[dict] = []
    feature_column = f"feature_path__{spec.name}"
    eval_rows = rows[rows["analysis_split"].isin(["validation", "test"])]
    for row in tqdm(eval_rows.itertuples(index=False), total=len(eval_rows), desc=f"Evaluate signals [{spec.name}]", unit="slide"):
        try:
            feature_path = Path(getattr(row, feature_column))
            values, coords = _read_feature_array(feature_path, spec.feature_key)
            attention, attention_coords = _read_attention_array(
                Path(row.attention_path), config.attention_key, config.target_layer
            )
            validate_alignment(coords, attention_coords, attention.shape[1], feature_path)
            grouped = group_attention(attention, groups)
            projected_values = projector.transform(values)
            indices = np.arange(len(values), dtype=np.int64)
            context_values = compute_context_descriptors(
                values,
                coords,
                indices,
                n_prototypes=config.slide_prototype_count,
                max_fit_tiles=config.max_slide_fit_tiles,
                spatial_neighbors=config.spatial_neighbors,
                seed=stable_seed(str(row.slide_id), config.seed + 1),
            )
            predictions = _predict_methods(models, projected_values, context_values)
            for method, scores in predictions.items():
                slide_group, slide_coverage = evaluate_scores(
                    scores,
                    grouped,
                    slide_id=str(row.slide_id),
                    case_id=str(row.case_id),
                    project=str(row.project),
                    split=str(row.analysis_split),
                    view=spec.name,
                    method=method,
                    config=config,
                )
                group_rows.extend(slide_group)
                coverage_rows.extend(slide_coverage)
        except Exception as exc:
            error_rows.append(
                {
                    "slide_id": str(row.slide_id),
                    "case_id": str(row.case_id),
                    "analysis_split": str(row.analysis_split),
                    "view": spec.name,
                    "error": repr(exc),
                }
            )

    per_group = pd.DataFrame(group_rows)
    coverage = pd.DataFrame(coverage_rows)
    errors = pd.DataFrame(error_rows)
    per_group.to_csv(view_dir / "per_slide_group_metrics.csv", index=False)
    coverage.to_csv(view_dir / "per_slide_coverage_metrics.csv", index=False)
    errors.to_csv(view_dir / "errors.csv", index=False)
    aggregate_group = aggregate_metrics(
        per_group,
        config,
        ["analysis_split", "view", "method", "group"],
    )
    aggregate_coverage = aggregate_metrics(
        coverage,
        config,
        ["analysis_split", "view", "method", "retention"],
    )
    aggregate_group.to_csv(view_dir / "aggregate_group_metrics.csv", index=False)
    aggregate_coverage.to_csv(view_dir / "aggregate_coverage_metrics.csv", index=False)

    summary = {
        "view": spec.name,
        "n_eval_slides": int(coverage["slide_id"].nunique()) if not coverage.empty else 0,
        "n_errors": int(len(errors)),
        "methods": sorted(models),
        "knn_backend": getattr(models.get("knn"), "backend", None),
        "files": {
            "per_slide_group_metrics": str(view_dir / "per_slide_group_metrics.csv"),
            "per_slide_coverage_metrics": str(view_dir / "per_slide_coverage_metrics.csv"),
            "aggregate_group_metrics": str(view_dir / "aggregate_group_metrics.csv"),
            "aggregate_coverage_metrics": str(view_dir / "aggregate_coverage_metrics.csv"),
        },
    }
    json_dump_atomic(summary, complete_path)
    return summary


def select_validation_winners(output_dir: Path, views: Sequence[FeatureViewSpec], config: SignalDiscoveryConfig) -> dict:
    primary_retention = min(config.retention_fractions, key=lambda value: abs(value - 0.50))
    winners: dict[str, dict] = {}
    comparison_rows: list[dict] = []
    for spec in views:
        path = output_dir / "views" / spec.name / "aggregate_coverage_metrics.csv"
        table = pd.read_csv(path)
        candidates = table[
            (table["analysis_split"] == "validation")
            & np.isclose(table["retention"], primary_retention)
            & (table["metric"] == "worst_group_top_recall")
        ].sort_values(["median", "method"], ascending=[False, True])
        if candidates.empty:
            winners[spec.name] = {"error": "No validation metric available"}
            continue
        winner = str(candidates.iloc[0]["method"])
        winners[spec.name] = {
            "selected_method": winner,
            "selection_split": "validation",
            "selection_metric": "worst_group_top_recall",
            "retention": primary_retention,
            "validation_median": float(candidates.iloc[0]["median"]),
        }
        test_rows = table[
            (table["analysis_split"] == "test")
            & np.isclose(table["retention"], primary_retention)
            & (table["method"] == winner)
        ]
        for metric in (
            "mean_group_top_recall",
            "worst_group_top_recall",
            "mean_group_attention_mass",
            "worst_group_attention_mass",
        ):
            match = test_rows[test_rows["metric"] == metric]
            if not match.empty:
                comparison_rows.append(
                    {
                        "view": spec.name,
                        "selected_method": winner,
                        "retention": primary_retention,
                        "metric": metric,
                        "test_median": float(match.iloc[0]["median"]),
                        "test_bootstrap95_low": float(match.iloc[0]["bootstrap95_low"]),
                        "test_bootstrap95_high": float(match.iloc[0]["bootstrap95_high"]),
                    }
                )
    pd.DataFrame(comparison_rows).to_csv(output_dir / "selected_view_comparison.csv", index=False)
    payload = {"primary_retention": primary_retention, "views": winners}
    json_dump_atomic(payload, output_dir / "selection.json")
    return payload


def run_attention_signal_discovery(
    *,
    attention_manifest: Path,
    feature_views: Sequence[FeatureViewSpec],
    metadata_path: Path | None,
    output_dir: Path,
    config: SignalDiscoveryConfig,
    model_filter: str | None = None,
    resume: bool = False,
) -> dict:
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_analysis_table(attention_manifest, feature_views, metadata_path, model_filter=model_filter)
    table = assign_case_splits(table, config)
    table.to_csv(output_dir / "analysis_manifest.csv", index=False)
    split_counts = (
        table.groupby("analysis_split").agg(n_slides=("slide_id", "nunique"), n_cases=("case_id", "nunique")).reset_index()
    )
    split_counts.to_csv(output_dir / "split_counts.csv", index=False)
    json_dump_atomic(asdict(config), output_dir / "config.json")
    json_dump_atomic(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        output_dir / "environment.json",
    )

    head_group_path = output_dir / "head_groups.json"
    if resume and head_group_path.exists():
        head_payload = json.loads(head_group_path.read_text())
    else:
        head_payload = fit_head_groups(table[table["analysis_split"] == "train"], config, output_dir)
    groups = head_payload["groups"]

    view_summaries = {}
    for spec in feature_views:
        view_summaries[spec.name] = run_feature_view(
            table,
            spec,
            groups,
            config,
            output_dir,
            resume=resume,
        )
    selection = select_validation_winners(output_dir, feature_views, config)
    summary = {
        "n_slides": int(table["slide_id"].nunique()),
        "n_cases": int(table["case_id"].nunique()),
        "head_groups": head_payload,
        "views": view_summaries,
        "selection": selection,
    }
    json_dump_atomic(summary, output_dir / "summary.json")
    return summary
