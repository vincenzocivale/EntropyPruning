"""Spatial fidelity and patient-level paired uncertainty, independent of torch."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spatial_edges(coords, neighbors=6):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.empty((0, 2), dtype=int)
    distances, indices = cKDTree(coords).query(coords, k=min(neighbors + 1, len(coords)))
    cutoff = 2.0 * np.median(distances[:, 1])
    pairs = set()
    for i in range(len(coords)):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            if distance <= cutoff and i != j:
                pairs.add(tuple(sorted((i, int(j)))))
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


def moran(values, edges):
    z = np.asarray(values) - np.mean(values)
    denominator = z @ z
    if len(edges) == 0 or denominator < 1e-12:
        return np.nan
    return float(len(z) * np.sum(z[edges[:, 0]] * z[edges[:, 1]]) / (len(edges) * denominator))


def map_metrics(truth, prediction, coords):
    edges = spatial_edges(coords)
    local_error = np.nan
    if len(edges):
        error = prediction - truth
        local_error = float(np.sqrt(np.mean((error[edges[:, 0]] - error[edges[:, 1]]) ** 2)))
    return {
        "pearson": correlation(truth, prediction),
        "spearman": correlation(rankdata(truth), rankdata(prediction)),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "edge_rmse": local_error,
        "moran_abs_error": abs(moran(truth, edges) - moran(prediction, edges)),
    }


def coverage_masks(coords, rectangles):
    """Half-open level-0 rectangles. Stream tiles to avoid spots x tiles memory."""
    covered = np.zeros(len(coords), dtype=bool)
    kept = covered.copy()
    for row in rectangles.itertuples(index=False):
        mask = ((coords[:, 0] >= row.x) & (coords[:, 0] < row.x + row.width)
                & (coords[:, 1] >= row.y) & (coords[:, 1] < row.y + row.height))
        covered |= mask
        if row.kept:
            kept |= mask
    return covered, kept


def niche_coverage(spots, rectangles, method):
    rows = []
    for slide, group in spots.groupby("slide_id", sort=True):
        tiles = rectangles[rectangles.slide_id == slide]
        if tiles.empty:
            raise ValueError(f"No selection rectangles for slide {slide}")
        eligible, retained = coverage_masks(group[["x", "y"]].to_numpy(), tiles)
        overall = retained.sum() / eligible.sum() if eligible.any() else np.nan
        strata = [("__all__", np.ones(len(group), dtype=bool))]
        if "niche" in group:
            strata.extend((n, group.niche.to_numpy() == n) for n in sorted(set(group.niche) - {""}))
        for niche, mask in strata:
            n = int((eligible & mask).sum())
            k = int((retained & mask).sum())
            fraction = k / n if n else np.nan
            rows.append(dict(method=method, slide_id=slide, patient_id=group.patient_id.iloc[0],
                             niche=niche, n_spots=int(mask.sum()), n_eligible=n, n_retained=k,
                             coverage=fraction, relative_coverage=fraction / overall if overall > 0 else np.nan,
                             lost=bool(n and not k), status="ok" if n else "no_spatial_overlap"))
    return pd.DataFrame(rows)


def paired_summary(results, reference, *, seed=42, bootstrap=2000):
    """Average sections within patient; paired CIs resample patients, never spots."""
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for (level, target, stratum, metric), group in results.groupby(["level", "target", "stratum", "metric"]):
        patient = group.groupby(["patient_id", "method"]).value.mean().unstack("method")
        for method in patient:
            values = patient[method].dropna().to_numpy()
            pairs = patient[[method, reference]].dropna() if method != reference else patient[[reference]].dropna()
            delta = (pairs[method] - pairs[reference]).to_numpy()
            low = high = np.nan
            if len(delta) >= 2:
                samples = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(bootstrap)])
                low, high = np.quantile(samples, [0.025, 0.975])
            rows.append(dict(level=level, target=target, stratum=stratum, metric=metric, method=method,
                             mean=float(values.mean()) if len(values) else np.nan,
                             n_patients=len(values), reference=reference, n_paired=len(delta),
                             delta=float(delta.mean()) if len(delta) else np.nan,
                             ci_low=low, ci_high=high,
                             status="ok" if len(delta) >= 2 else "insufficient_patients_for_ci"))
    return pd.DataFrame(rows)
