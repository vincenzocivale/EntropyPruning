#!/usr/bin/env python
"""Audit concentration, layer stability, head agreement and spatial structure of TITAN attention."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from tqdm.auto import tqdm


def _resolve(value: str, manifest: Path) -> Path:
    path = Path(value)
    if path.is_absolute(): return path
    choices = [(manifest.parent / path).resolve(), (manifest.parent.parent / path).resolve(), (Path.cwd() / path).resolve()]
    return next((item for item in choices if item.exists()), choices[0])


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), 0, None)
    return values / np.clip(values.sum(axis=-1, keepdims=True), 1e-12, None)


def _top_mass(values: np.ndarray, fraction: float) -> float:
    n = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.partition(values, -n)[-n:].sum())


def _fraction_for_mass(values: np.ndarray, mass: float) -> float:
    return float(np.searchsorted(np.cumsum(np.sort(values)[::-1]), mass, side="left") + 1) / len(values)


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.clip(values, 0, None)); n = len(values); total = values.sum()
    return float((2 * np.arange(1, n + 1).dot(values) / (n * total)) - (n + 1) / n) if total else np.nan


def _top_recall(early: np.ndarray, final: np.ndarray, fraction: float) -> float:
    n = max(1, int(np.ceil(len(final) * fraction)))
    return len(set(np.argpartition(early, -n)[-n:]).intersection(np.argpartition(final, -n)[-n:])) / n


def _ndcg(early: np.ndarray, final: np.ndarray, fraction: float) -> float:
    n = max(1, int(np.ceil(len(final) * fraction))); order = np.argsort(early)[::-1][:n]
    discount = 1 / np.log2(np.arange(2, n + 2)); ideal = np.sort(final)[::-1][:n]
    return float((final[order] * discount).sum() / max((ideal * discount).sum(), 1e-12))


def _moran(values: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    if len(values) <= 2: return np.nan
    neighbors = cKDTree(coords[:, :2]).query(coords[:, :2], k=min(k + 1, len(values)))[1][:, 1:]
    centered = values - values.mean(); denominator = np.square(centered).sum()
    return float((len(values) / (len(values) * neighbors.shape[1])) * (centered[:, None] * centered[neighbors]).sum() / denominator) if denominator else np.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-output-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fractions", nargs="+", type=float, default=(.01, .05, .10, .20))
    parser.add_argument("--spatial-k", type=int, default=8)
    parser.add_argument("--max-slides", type=int)
    args = parser.parse_args()
    outputs = pd.read_csv(args.wsi_output_manifest); outputs = outputs[outputs.status.isin(["complete", "skipped"])]
    if args.max_slides: outputs = outputs.sort_values("slide_id").head(args.max_slides)
    metadata = pd.read_csv(args.metadata) if args.metadata else pd.DataFrame(columns=["slide_id"])
    if "tcga_project" in metadata and "project" not in metadata: metadata = metadata.rename(columns={"tcga_project": "project"})
    project = dict(zip(metadata.slide_id, metadata.get("project", pd.Series(dtype=str))))
    summary, layer_head, agreement, errors = [], [], [], []
    for item in tqdm(outputs.itertuples(index=False), total=len(outputs), desc="TITAN attention structure", unit="slide"):
        try:
            with h5py.File(_resolve(item.path, args.wsi_output_manifest)) as handle:
                global_values = _normalize(np.asarray(handle["attention"]["global_to_tiles_mass_share"]))
                received_values = _normalize(np.asarray(handle["attention"]["received_by_tiles_broadcast"]))
                coords = np.asarray(handle["coords"])
            layers, heads, n_tiles = global_values.shape; final_layer = layers - 1
            base = {"slide_id": item.slide_id, "case_id": "-".join(item.slide_id.split("-")[:3]), "project": project.get(item.slide_id), "n_tiles": n_tiles}
            head_rhos, global_received, morans = [], [], []
            for layer in range(layers):
                for head in range(heads):
                    values, received = global_values[layer, head], received_values[layer, head]
                    entropy = float(-(values * np.log(np.clip(values, 1e-12, None))).sum() / np.log(n_tiles))
                    row = {**base, "layer": layer, "head": head, "entropy_norm": entropy, "effective_tile_fraction": float(1 / np.square(values).sum() / n_tiles), "gini": _gini(values), "max_over_median": float(values.max() / max(np.median(values), 1e-12)), "spatial_moran": _moran(values, coords, args.spatial_k), "global_received_spearman": float(spearmanr(values, received).statistic)}
                    for fraction in args.top_fractions: row[f"top_{int(fraction * 100)}_mass"] = _top_mass(values, fraction)
                    for mass in (.5, .8, .9): row[f"fraction_for_{int(mass * 100)}_mass"] = _fraction_for_mass(values, mass)
                    layer_head.append(row); global_received.append(row["global_received_spearman"]); morans.append(row["spatial_moran"])
                for left in range(heads):
                    for right in range(left + 1, heads): head_rhos.append(float(spearmanr(global_values[layer, left], global_values[layer, right]).statistic))
            for early in range(min(3, layers - 1)):
                for head in range(heads):
                    early_values, final_values = global_values[early, head], global_values[final_layer, head]
                    row = {**base, "early_layer": early, "target_layer": final_layer, "head": head, "spearman": float(spearmanr(early_values, final_values).statistic), "ndcg_top10": _ndcg(early_values, final_values, .10)}
                    for fraction in (.05, .10, .20): row[f"top_{int(fraction * 100)}_recall"] = _top_recall(early_values, final_values, fraction)
                    agreement.append(row)
            final = global_values[final_layer].mean(axis=0)
            final /= final.sum()
            summary.append({**base, "entropy_norm_mean": float(np.mean([r["entropy_norm"] for r in layer_head[-layers * heads:]])), "effective_tile_fraction_mean": float(np.mean([r["effective_tile_fraction"] for r in layer_head[-layers * heads:]])), "head_agreement_mean": float(np.nanmean(head_rhos)), "global_received_spearman_mean": float(np.nanmean(global_received)), "spatial_moran_mean": float(np.nanmean(morans)), "final_top10_mass": _top_mass(final, .10), "final_fraction_for_80_mass": _fraction_for_mass(final, .8), "layer1_final_spearman": float(np.nanmean([r["spearman"] for r in agreement[-min(3, layers - 1) * heads:] if r["early_layer"] == 0])), "layer2_final_spearman": float(np.nanmean([r["spearman"] for r in agreement[-min(3, layers - 1) * heads:] if r["early_layer"] == 1])) if layers > 2 else np.nan})
        except Exception as exc: errors.append({"slide_id": item.slide_id, "error": repr(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output_dir / "attention_distribution_per_slide.csv", index=False)
    pd.DataFrame(layer_head).to_csv(args.output_dir / "per_layer_head.csv", index=False)
    pd.DataFrame(agreement).to_csv(args.output_dir / "early_final_agreement.csv", index=False)
    pd.DataFrame(errors).to_csv(args.output_dir / "errors.csv", index=False)
    print(f"slides={len(summary)} errors={len(errors)} results={args.output_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
