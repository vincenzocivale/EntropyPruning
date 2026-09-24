"""Strict, identifier-based contracts for spatial evaluation inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_spots(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"spot_id", "slide_id", "patient_id", "cohort", "split", "x", "y"}
    if required - set(table):
        raise ValueError(f"Spots missing columns: {sorted(required - set(table))}")
    if table.empty or (table[list(required)] == "").any().any():
        raise ValueError("Spot metadata must be nonempty, including patient IDs")
    if table.spot_id.duplicated().any():
        raise ValueError("spot_id must be globally unique (prefix barcodes with slide ID)")
    if "niche" in table and (table.niche == "__all__").any():
        raise ValueError("__all__ is reserved, not a valid niche label")
    if set(table.split) != {"train", "validation", "test"}:
        raise ValueError("Explicit train, validation and test splits are required")
    if (table.groupby("patient_id").split.nunique() > 1).any():
        raise ValueError("Patient leakage across splits")
    if (table.groupby("slide_id")[["patient_id", "cohort", "split"]].nunique() > 1).any().any():
        raise ValueError("Each slide must have one patient, cohort and split")
    table[["x", "y"]] = table[["x", "y"]].apply(pd.to_numeric)
    if not np.isfinite(table[["x", "y"]].to_numpy()).all():
        raise ValueError("Nonfinite spatial coordinates")
    if table.duplicated(["slide_id", "x", "y"]).any():
        raise ValueError("Duplicate spot coordinates within a slide")
    return table


def identifiers(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional string array")
    values = values.astype(str)
    if any(not x for x in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} contains empty or duplicate identifiers")
    return values


def aligned_matrix(path, ids, *, id_key="spot_id", value_key="embeddings", exact=True):
    with np.load(path, allow_pickle=False) as archive:
        keys = identifiers(archive[id_key], id_key)
        values = np.asarray(archive[value_key], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(keys) or values.shape[1] == 0:
        raise ValueError(f"Invalid matrix shape in {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite matrix in {path}")
    wanted = list(map(str, ids))
    if not set(wanted) <= set(keys) or (exact and set(wanted) != set(keys)):
        raise ValueError(f"Identifier coverage mismatch in {path}")
    lookup = {key: i for i, key in enumerate(keys)}
    return values[[lookup[key] for key in wanted]]


def read_expression(path, ids, *, normalization, layer=None):
    """Read one cohort-level NPZ or H5AD; no guessed normalization or ID order."""
    path = Path(path)
    if path.suffix == ".h5ad":
        import anndata  # optional; imported only for this format
        from scipy import sparse

        adata = anndata.read_h5ad(path)
        keys = identifiers(np.asarray(adata.obs_names, dtype=str), "spot_id")
        genes = identifiers(np.asarray(adata.var_names, dtype=str), "genes")
        if set(keys) != set(ids):
            raise ValueError("Expression/spot identifier coverage mismatch")
        adata = adata[list(ids)]
        matrix = adata.layers[layer] if layer else adata.X
        values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    else:
        values = aligned_matrix(path, ids, value_key="expression")
        with np.load(path, allow_pickle=False) as archive:
            genes = identifiers(archive["genes"], "genes")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(ids), len(genes)) or not np.isfinite(values).all():
        raise ValueError("Expression shape or finite-value validation failed")
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


def read_signatures(path):
    signatures = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3 or not all(fields) or fields[0] in signatures:
            raise ValueError("GMT requires unique name, source, and nonempty genes")
        signatures[fields[0]] = {"source": fields[1], "genes": sorted(set(fields[2:]))}
    if not signatures:
        raise ValueError("No signatures in GMT")
    return signatures


def program_targets(expression, genes, train, signatures, min_coverage=0.8):
    """Fit gene standardization on training spots only; return frozen transform."""
    if not 0 < min_coverage <= 1:
        raise ValueError("min_signature_coverage must be in (0, 1]")
    mean = expression[train].mean(axis=0)
    scale = expression[train].std(axis=0)
    lookup = {gene: i for i, gene in enumerate(genes)}
    weights, names, audit = [], [], {}
    for name, spec in signatures.items():
        present = [g for g in spec["genes"] if g in lookup]
        usable = [g for g in present if scale[lookup[g]] > 1e-12]
        accepted = len(usable) / len(spec["genes"]) >= min_coverage
        audit[name] = dict(spec, missing=sorted(set(spec["genes"]) - set(present)),
                           constant=sorted(set(present) - set(usable)),
                           status="included" if accepted else "skipped_insufficient_coverage")
        if accepted:
            w = np.zeros(len(genes))
            w[[lookup[g] for g in usable]] = 1.0 / len(usable)
            weights.append(w)
            names.append(name)
    if not weights:
        raise ValueError("No signatures meet gene coverage / training variance requirements")
    weights = np.stack(weights, axis=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return ((expression - mean) / scale) @ weights, names, (mean, scale, weights), audit
