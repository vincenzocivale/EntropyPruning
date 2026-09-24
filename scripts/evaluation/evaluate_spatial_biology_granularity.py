#!/usr/bin/env python3
"""Two granularity-favorable tests for CONCH-frozen spot embeddings vs ST pathway
program scores, before deciding to abandon pathway-supervised EAF training.

Test A (local aggregation): does pooling embeddings over a 3x3 / 5x5 spot
neighborhood recover more pathway signal than single-spot regression?

Test B (hotspot discrimination): even if continuous regression stays weak, can
frozen embeddings separate program-high from program-low spots (top/bottom
percentile) well enough to be useful for tile/spot selection?

Reuses the same spots/expression/signature contracts as
src/evaluation/spatial, but is a standalone diagnostic -- it does not touch
the frozen, provenance-tracked evaluate_spatial_biology.py pipeline or its
registry-declared experiments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.spatial.data import aligned_matrix, identifiers, program_targets, read_signatures, read_spots


def read_expression_subset(path, ids, *, normalization):
    """Like data.read_expression, but tolerates expression covering a superset
    of ids (we may have dropped spots without an exported embedding)."""
    values = aligned_matrix(path, ids, value_key="expression", exact=False)
    with np.load(path, allow_pickle=False) as archive:
        genes = identifiers(archive["genes"], "genes")
    if (values < 0).any():
        raise ValueError("Expected nonnegative counts or log1p expression")
    if normalization == "counts":
        totals = values.sum(axis=1)
        if (totals <= 0).any():
            raise ValueError("Zero-library spots must be removed before evaluation")
        values = np.log1p(values / totals[:, None] * 10000.0)
    elif normalization != "log1p_cp10k":
        raise ValueError("normalization must be counts or log1p_cp10k")
    return values, genes


def local_aggregate(embeddings, spots, k):
    """Mean-pool each spot's embedding over its k nearest neighbours (incl. self),
    within the same slide only. k=1 is single-spot (no pooling)."""
    if k <= 1:
        return embeddings
    pooled = np.empty_like(embeddings)
    for slide, group in spots.groupby("slide_id", sort=False):
        idx = group.index.to_numpy()
        coords = group[["x", "y"]].to_numpy()
        n = min(k, len(idx))
        tree = cKDTree(coords)
        _, neighbor_pos = tree.query(coords, k=n)
        neighbor_pos = np.atleast_2d(neighbor_pos)
        if n == 1:
            neighbor_pos = neighbor_pos.reshape(-1, 1)
        for row, neighbors in enumerate(neighbor_pos):
            pooled[idx[row]] = embeddings[idx[neighbors]].mean(axis=0)
    return pooled


def fit_ridge_regression(train_x, val_x, test_x, train_y, val_y, alphas, seed):
    best_loss, best_pred, best_alpha = np.inf, None, None
    for alpha in sorted(alphas):
        model = Ridge(alpha=alpha, random_state=seed).fit(train_x, train_y)
        loss = float(np.mean((model.predict(val_x) - val_y) ** 2))
        if loss < best_loss:
            best_loss, best_pred, best_alpha = loss, model.predict(test_x), alpha
    return best_pred, best_alpha


def fit_logistic(train_x, val_x, test_x, train_y, val_y, cs, seed):
    if len(set(train_y)) < 2 or len(set(val_y)) < 2:
        return None, None
    best_loss, best_pred, best_c = np.inf, None, None
    for c in sorted(cs):
        model = LogisticRegression(C=c, max_iter=2000, random_state=seed).fit(train_x, train_y)
        proba = model.predict_proba(val_x)[:, 1]
        # validation loss proxy: negative log-likelihood
        eps = 1e-9
        loss = float(-np.mean(val_y * np.log(proba + eps) + (1 - val_y) * np.log(1 - proba + eps)))
        if loss < best_loss:
            best_loss, best_c = loss, c
            best_pred = model.predict_proba(test_x)[:, 1]
    return best_pred, best_c


def pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def embedded_spot_ids(path):
    with np.load(path, allow_pickle=False) as archive:
        return set(archive["spot_id"].astype(str))


def run(args):
    spots = read_spots(args.spots)
    covered = embedded_spot_ids(args.embeddings)
    if not covered <= set(spots.spot_id):
        raise ValueError("Embeddings reference spot IDs absent from the spot table")
    if covered != set(spots.spot_id):
        dropped = len(spots) - len(covered)
        print(f"[granularity-eval] embeddings cover {len(covered)}/{len(spots)} spots; "
              f"dropping {dropped} spots without an exported embedding")
        spots = spots[spots.spot_id.isin(covered)].reset_index(drop=True)
        for split in ("train", "validation", "test"):
            if not (spots.split == split).any():
                raise ValueError(f"No {split} spots remain after restricting to embedded spots")
    expression, genes = read_expression_subset(args.expression, spots.spot_id, normalization=args.normalization)
    signatures = read_signatures(args.signatures)
    train_mask = spots.split.to_numpy() == "train"
    val_mask = spots.split.to_numpy() == "validation"
    test_mask = spots.split.to_numpy() == "test"
    targets, names, _, audit = program_targets(expression, genes, train_mask, signatures, args.min_signature_coverage)
    print(f"[granularity-eval] programs included: {names}")
    for name, info in audit.items():
        if info["status"] != "included":
            print(f"[granularity-eval] skipped {name}: {info['status']}")

    embeddings = aligned_matrix(args.embeddings, spots.spot_id)
    pca_components = min(args.pca_components, int(train_mask.sum()) - 1, embeddings.shape[1])

    ridge_alphas = [0.01, 0.1, 1, 10, 100, 1000]
    logreg_cs = [0.01, 0.1, 1, 10, 100]

    regression_rows = []
    for k, label in zip(args.neighbor_ks, args.neighbor_labels):
        pooled = local_aggregate(embeddings, spots, k)
        scaler = StandardScaler().fit(pooled[train_mask])
        pca = PCA(n_components=pca_components, svd_solver="full", random_state=args.seed).fit(
            scaler.transform(pooled[train_mask]))
        xtrain, xval, xtest = [pca.transform(scaler.transform(pooled[mask])) for mask in (train_mask, val_mask, test_mask)]
        for j, program in enumerate(names):
            pred, alpha = fit_ridge_regression(
                xtrain, xval, xtest, targets[train_mask, j], targets[val_mask, j], ridge_alphas, args.seed)
            r = pearson(targets[test_mask, j], pred)
            regression_rows.append(dict(test="local_aggregation", neighborhood=label, k=k,
                                        program=program, pearson=r, ridge_alpha=alpha,
                                        n_test_spots=int(test_mask.sum())))
            print(f"[granularity-eval] regression k={label:>10} program={program:<20} pearson={r:.4f}")

    hotspot_rows = []
    pooled = local_aggregate(embeddings, spots, args.hotspot_neighbor_k)
    scaler = StandardScaler().fit(pooled[train_mask])
    pca = PCA(n_components=pca_components, svd_solver="full", random_state=args.seed).fit(
        scaler.transform(pooled[train_mask]))
    xtrain, xval, xtest = [pca.transform(scaler.transform(pooled[mask])) for mask in (train_mask, val_mask, test_mask)]
    for j, program in enumerate(names):
        train_scores = targets[train_mask, j]
        lo, hi = np.quantile(train_scores, [args.hotspot_fraction, 1 - args.hotspot_fraction])
        split_sets = {}
        for mask_name, x, scores in (("train", xtrain, train_scores),
                                     ("val", xval, targets[val_mask, j]),
                                     ("test", xtest, targets[test_mask, j])):
            keep = (scores <= lo) | (scores >= hi)
            split_sets[mask_name] = (x[keep], (scores[keep] >= hi).astype(int))
        train_x, train_y = split_sets["train"]
        val_x, val_y = split_sets["val"]
        test_x, test_y = split_sets["test"]
        proba, c = fit_logistic(train_x, val_x, test_x, train_y, val_y, logreg_cs, args.seed)
        auroc = roc_auc_score(test_y, proba) if proba is not None and len(set(test_y)) > 1 else np.nan
        hotspot_rows.append(dict(test="hotspot_auroc", program=program, fraction=args.hotspot_fraction,
                                 logreg_c=c, auroc=auroc, n_test_hotspot_spots=int(len(test_y))))
        print(f"[granularity-eval] hotspot program={program:<20} n_test={len(test_y):<6} auroc={auroc:.4f}")

    regression_df = pd.DataFrame(regression_rows)
    hotspot_df = pd.DataFrame(hotspot_rows)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    regression_df.to_csv(Path(args.output_dir) / "local_aggregation_regression.csv", index=False)
    hotspot_df.to_csv(Path(args.output_dir) / "hotspot_auroc.csv", index=False)
    print(f"[granularity-eval] wrote {args.output_dir}/local_aggregation_regression.csv "
          f"and {args.output_dir}/hotspot_auroc.csv")
    return regression_df, hotspot_df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spots", required=True)
    p.add_argument("--expression", required=True)
    p.add_argument("--signatures", required=True)
    p.add_argument("--embeddings", required=True, help="NPZ with spot_id,embeddings (e.g. CONCH-full tile export)")
    p.add_argument("--normalization", default="counts", choices=["counts", "log1p_cp10k"])
    p.add_argument("--min-signature-coverage", type=float, default=0.8)
    p.add_argument("--pca-components", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neighbor-ks", type=int, nargs="+", default=[1, 9, 25])
    p.add_argument("--neighbor-labels", nargs="+", default=["single_spot", "3x3", "5x5"])
    p.add_argument("--hotspot-neighbor-k", type=int, default=1,
                    help="Neighborhood pooling to use for the hotspot test (default: single spot)")
    p.add_argument("--hotspot-fraction", type=float, default=0.15,
                    help="Top/bottom fraction of program score treated as hotspot/coldspot")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if len(args.neighbor_ks) != len(args.neighbor_labels):
        p.error("--neighbor-ks and --neighbor-labels must have the same length")
    if not 0 < args.hotspot_fraction < 0.5:
        p.error("--hotspot-fraction must be in (0, 0.5)")
    return args


if __name__ == "__main__":
    run(parse_args())
