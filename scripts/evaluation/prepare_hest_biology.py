#!/usr/bin/env python3
"""Materialize the frozen HEST breast subset used by spatial biology evaluation.

This converts HEST h5ad expression into one aligned NPZ and writes a strict,
patient-disjoint spot manifest. Image patches remain in the downloaded H5
stores; ``patch_store``/``patch_index`` avoid a second multi-GB copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def _split_patients(patients, seed=42):
    ordered = sorted(set(map(str, patients)))
    keyed = sorted(ordered, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
    n = len(keyed)
    n_train, n_val = max(1, int(round(n * .6))), max(1, int(round(n * .2)))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    return {p: ("train" if i < n_train else "validation" if i < n_train + n_val else "test")
            for i, p in enumerate(keyed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--first", type=int, default=51)
    ap.add_argument("--last", type=int, default=80)
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated explicit HEST sample IDs, overriding --first/--last.")
    ap.add_argument("--cohort", type=str, default="hest_breast_idc",
                    help="Cohort label written to spots.csv; also namespaces patient_id to avoid cross-cohort collisions.")
    ap.add_argument("--append", action="store_true",
                    help="Merge into existing spots.csv/expression.npz under --root instead of overwriting.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    ids = [s.strip() for s in args.ids.split(",")] if args.ids else [f"SPA{i}" for i in range(args.first, args.last + 1)]
    metadata = pd.read_csv(root / "HEST_v1_3_0.csv", dtype=str)
    selected = metadata[metadata.id.isin(ids)].set_index("id")
    if set(selected.index) != set(ids):
        raise RuntimeError(f"Missing metadata for {sorted(set(ids) - set(selected.index))}")
    rows, matrices, gene_lists = [], [], []
    for sid in ids:
        h5ad = root / "st" / f"{sid}.h5ad"
        patch = root / "patches" / f"{sid}.h5"
        if not h5ad.exists() or not patch.exists():
            raise FileNotFoundError(f"Missing HEST files for {sid}")
        a = ad.read_h5ad(h5ad)
        x = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
        if np.any(x < 0) or not np.all(np.isfinite(x)):
            raise ValueError(f"Invalid expression values in {sid}")
        # var_names are gene identifiers directly on some HEST releases (e.g. the
        # Spatial Transcriptomics breast cohort uses Ensembl IDs), but on others
        # (e.g. Visium kidney) var_names are symbols and the Ensembl ID lives in
        # var["gene_ids"]. Cross-cohort merges must use one consistent namespace,
        # or every gene silently fails to match and reads as zero everywhere.
        current_genes = np.asarray(a.var["gene_ids"] if "gene_ids" in a.var else a.var_names, dtype=str)
        if len(set(current_genes)) != len(current_genes):
            raise ValueError(f"Duplicate gene identifiers in {sid}")
        gene_lists.append(current_genes)
        with h5py.File(patch, "r") as f:
            barcodes = np.asarray(f["barcode"]).reshape(-1).astype(str)
            coords = np.asarray(f["coords"])
            obs_barcodes = np.asarray(a.obs_names, dtype=str)
            if not set(barcodes).issubset(set(obs_barcodes)):
                raise ValueError(f"Patch/ST barcode mismatch in {sid}")
            order = {b: i for i, b in enumerate(obs_barcodes)}
            x = x[[order[b] for b in barcodes]]
            if coords.shape != (len(barcodes), 2) or f["img"].shape[0] != len(barcodes):
                raise ValueError(f"Malformed patch store in {sid}")
        patient = f"{args.cohort}:{selected.loc[sid, 'patient']}"
        for i, (barcode, xy) in enumerate(zip(barcodes, coords)):
            rows.append(dict(spot_id=f"{sid}:{barcode}", slide_id=sid, patient_id=patient,
                             cohort=args.cohort, split="", x=int(xy[0]), y=int(xy[1]),
                             niche="", patch_store=str(patch), patch_index=i))
        matrices.append((current_genes, x))
    split = _split_patients([r["patient_id"] for r in rows], args.seed)
    for row in rows:
        row["split"] = split[row["patient_id"]]
    table = pd.DataFrame(rows)
    genes = np.array(sorted(set().union(*(set(g) for g in gene_lists))), dtype=str)
    expression_parts = []
    for current_genes, matrix in matrices:
        aligned = np.zeros((matrix.shape[0], len(genes)), dtype=np.float32)
        lookup = {g: i for i, g in enumerate(current_genes)}
        aligned[:, [lookup[g] for g in genes if g in lookup]] = matrix[:, [lookup[g] for g in genes if g in lookup]]
        expression_parts.append(aligned)
    expression = np.concatenate(expression_parts, axis=0)

    if args.append and (root / "spots.csv").exists():
        existing_table = pd.read_csv(root / "spots.csv", dtype=str)
        existing_expr = np.load(root / "expression.npz", allow_pickle=False)
        existing_genes = existing_expr["genes"]
        merged_genes = np.array(sorted(set(existing_genes) | set(genes)), dtype=str)
        merged_lookup = {g: i for i, g in enumerate(merged_genes)}

        def reindex(values, old_genes):
            out = np.zeros((values.shape[0], len(merged_genes)), dtype=np.float32)
            cols = [merged_lookup[g] for g in old_genes]
            out[:, cols] = values
            return out

        merged_expression = np.concatenate([
            reindex(existing_expr["expression"], existing_genes),
            reindex(expression, genes),
        ], axis=0)
        merged_spot_ids = np.concatenate([existing_expr["spot_id"], table.spot_id.to_numpy(dtype=str)])
        if len(set(merged_spot_ids)) != len(merged_spot_ids):
            raise ValueError("Duplicate spot_id across cohorts; prefix collision")
        table = pd.concat([existing_table, table], ignore_index=True)
        genes, expression = merged_genes, merged_expression
        existing_split = json.loads((root / "split.json").read_text())["patient_to_split"]
        split = {**existing_split, **split}

    np.savez_compressed(root / "expression.npz", spot_id=table.spot_id.to_numpy(dtype=str),
                        genes=genes.astype(str), expression=expression)
    table.to_csv(root / "spots.csv", index=False)
    (root / "split.json").write_text(json.dumps({"seed": args.seed, "patient_to_split": split}, indent=2) + "\n")
    print(json.dumps({"spots": len(table), "genes": len(genes), "slides": table.slide_id.nunique(),
                      "patients": len(split), "split_counts": table.split.value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
