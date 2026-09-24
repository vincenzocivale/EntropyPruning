#!/usr/bin/env python3
"""Patient-held-out cross-validation readout for the spatial biology tile check.

The frozen single-split protocol (evaluate_spatial_biology.py) has only 2 test
patients in this HEST breast cohort, which makes any single Pearson/Spearman
number unstable (a bad patient draw can flip the sign). This script rotates
every patient through the test role exactly once, keeping the same
patient-disjoint train/validation/test structure and the same frozen
program-score/ridge-head machinery, and reports the distribution of test
metrics across folds instead of one fixed split.

This is a diagnostic tool, not a registry-gated scientific artifact: it does
not write to $EAF_WSI_ROOT/results and is not referenced by any frozen
protocol_sha256.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.spatial.data import aligned_matrix, program_targets, read_expression, read_signatures, read_spots
from src.evaluation.spatial.metrics import map_metrics
from src.evaluation.spatial.protocol import fit_predict


def make_folds(patient_ids, *, n_folds, n_validation, seed):
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(set(patient_ids)))
    rng.shuffle(patients)
    folds = np.array_split(patients, n_folds)
    for i, test_patients in enumerate(folds):
        remaining = [p for p in patients if p not in set(test_patients)]
        validation_patients = remaining[i * n_validation:(i + 1) * n_validation] or remaining[:n_validation]
        train_patients = [p for p in remaining if p not in set(validation_patients)]
        yield list(test_patients), validation_patients, train_patients


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="spots.csv")
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--normalization", default="counts")
    parser.add_argument("--min-signature-coverage", type=float, default=0.8)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-validation-patients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "EMBEDDINGS_NPZ"), required=True,
                        help="Repeatable: --method full /path/full.npz --method eaf /path/eaf.npz")
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args(argv)

    spots = read_spots(args.manifest)
    signatures = read_signatures(args.signatures)
    if args.normalization != "counts":
        expression, genes = read_expression(args.expression, spots.spot_id, normalization=args.normalization)
    else:
        # Large multi-cohort panels can have tens of thousands of irrelevant genes;
        # log1p-CPM-normalizing all of them in float64 is the actual memory/CPU cost,
        # not anything the signatures need. Compute per-spot library size from the
        # full raw matrix (required for correct normalization) but only materialize
        # the log1p transform for genes actually referenced by a signature.
        with np.load(args.expression, allow_pickle=False) as archive:
            raw_genes = archive["genes"].astype(str)
            raw_ids = archive["spot_id"].astype(str)
            wanted = list(map(str, spots.spot_id))
            if set(wanted) != set(raw_ids):
                raise ValueError("Identifier coverage mismatch in expression file")
            order = {sid: i for i, sid in enumerate(raw_ids)}
            row_order = [order[sid] for sid in wanted]
            needed_genes = sorted(set().union(*(set(spec["genes"]) for spec in signatures.values())))
            gene_lookup = {g: i for i, g in enumerate(raw_genes)}
            cols = [gene_lookup[g] for g in needed_genes if g in gene_lookup]
            genes = raw_genes[cols]
            raw = archive["expression"]
            totals_all = raw.sum(axis=1).astype(np.float64)  # one pass, no row-reorder copy
            totals = totals_all[row_order]
            if (totals <= 0).any():
                raise ValueError("Zero-library spots must be removed before evaluation")
            subset = raw[np.ix_(row_order, cols)].astype(np.float64)  # single small copy (rows x ~40 genes)
            del raw
            expression = np.log1p(subset / totals[:, None] * 10000.0)
    raw_matrices = {name: aligned_matrix(path, spots.spot_id) for name, path in args.method}

    rows = []
    for fold, (test_p, val_p, train_p) in enumerate(make_folds(spots.patient_id, n_folds=args.n_folds,
                                                                n_validation=args.n_validation_patients, seed=args.seed)):
        fold_spots = spots.copy()
        fold_spots["split"] = np.select(
            [fold_spots.patient_id.isin(test_p), fold_spots.patient_id.isin(val_p), fold_spots.patient_id.isin(train_p)],
            ["test", "validation", "train"], default="unused")
        fold_spots = fold_spots[fold_spots.split != "unused"].reset_index(drop=True)
        keep = spots.spot_id.isin(fold_spots.spot_id).to_numpy()
        fold_expression = expression[keep]
        fold_matrices = {name: values[keep] for name, values in raw_matrices.items()}
        train_mask = fold_spots.split.to_numpy() == "train"
        targets, names, _transform, audit = program_targets(fold_expression, genes, train_mask, signatures,
                                                             args.min_signature_coverage)
        skipped = [n for n, a in audit.items() if a["status"] != "included"]
        if skipped:
            print(f"fold {fold}: skipped signatures (insufficient coverage on this fold's train set): {skipped}", file=sys.stderr)
        predictions, _fits = fit_predict(fold_matrices, targets, fold_spots, pca_components=args.pca_components,
                                         alphas=args.alphas, seed=args.seed)
        test_mask = fold_spots.split.to_numpy() == "test"
        test_spots = fold_spots[test_mask].reset_index(drop=True)
        truth = targets[test_mask]
        for method, prediction in predictions.items():
            for patient_id, group in test_spots.groupby("patient_id"):
                idx = group.index.to_numpy()
                for j, target in enumerate(names):
                    metrics = map_metrics(truth[idx, j], prediction[idx, j], group[["x", "y"]].to_numpy())
                    for metric, value in metrics.items():
                        rows.append(dict(fold=fold, method=method, target=target, patient_id=patient_id,
                                         n_spots=len(idx), metric=metric, value=value))

    results = pd.DataFrame(rows)
    if args.out_csv:
        results.to_csv(args.out_csv, index=False)
    summary = (results[results.metric.isin(["pearson", "spearman"])]
               .groupby(["method", "target", "metric"])["value"]
               .agg(["mean", "std", "count"]).reset_index())
    print(summary.to_string(index=False))
    print()
    print(f"Per-patient detail (n={results.patient_id.nunique()} patients, {args.n_folds} folds):")
    detail = (results[results.metric == "pearson"]
              .pivot_table(index=["target", "patient_id"], columns="method", values="value"))
    print(detail.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
