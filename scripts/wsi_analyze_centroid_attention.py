#!/usr/bin/env python
"""Analyze TITAN tile attention against cosine similarity to the WSI mean embedding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def resolve(value: str, manifest: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [(manifest.parent / path).resolve(), (manifest.parent.parent / path).resolve(), (Path.cwd() / path).resolve()]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--wsi-output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-set-id", default="conch_v15_final_d768")
    parser.add_argument("--attention-key", default="global_to_tiles_mass_share")
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")

    features = pd.read_csv(args.feature_manifest)
    features = features[(features.get("artifact_type") == "tile_features") & (features.get("feature_set_id") == args.feature_set_id)]
    outputs = pd.read_csv(args.wsi_output_manifest)
    outputs = outputs[outputs["status"].isin(["complete", "skipped"])]
    table = features[["slide_id", "path"]].merge(outputs[["slide_id", "path"]], on="slide_id", suffixes=("_feature", "_output"), validate="one_to_one")
    rows, top_rows, errors = [], [], []
    for item in table.itertuples(index=False):
        try:
            feature_path, output_path = resolve(item.path_feature, args.feature_manifest), resolve(item.path_output, args.wsi_output_manifest)
            with h5py.File(feature_path) as feature_file, h5py.File(output_path) as output_file:
                values = np.asarray(feature_file["features"] if "features" in feature_file else feature_file["embeddings"]["final"], dtype=np.float32)
                values = values[:, 0, :] if values.ndim == 3 and values.shape[1] == 1 else values
                coords = np.asarray(output_file["coords"], dtype=np.int32)
                attention = np.asarray(output_file["attention"][args.attention_key], dtype=np.float32)
            attention = attention.mean(axis=tuple(range(attention.ndim - 1)))
            attention = np.clip(attention, 0, None); attention /= attention.sum()
            normed = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)
            mean = values.mean(axis=0); mean /= max(np.linalg.norm(mean), 1e-8)
            similarity = normed @ mean
            n_top = max(1, int(np.ceil(len(attention) * args.top_fraction)))
            top = np.argsort(attention)[-n_top:][::-1]
            rows.append({"slide_id": item.slide_id, "case_id": "-".join(item.slide_id.split("-")[:3]), "n_tiles": len(attention), "rho_attention_centroid_similarity": float(spearmanr(attention, similarity).statistic), "attention_weighted_centroid_similarity": float(attention @ similarity), "mean_centroid_similarity": float(similarity.mean()), "top_attention_mean_centroid_similarity": float(similarity[top].mean()), "top_attention_mass": float(attention[top].sum())})
            top_rows.extend({"slide_id": item.slide_id, "rank": rank + 1, "x": int(coords[idx, 0]), "y": int(coords[idx, 1]), "attention": float(attention[idx]), "centroid_similarity": float(similarity[idx])} for rank, idx in enumerate(top))
        except Exception as exc:
            errors.append({"slide_id": item.slide_id, "error": repr(exc)})
    per_slide = pd.DataFrame(rows); args.output_dir.mkdir(parents=True, exist_ok=True)
    per_slide.to_csv(args.output_dir / "per_slide.csv", index=False); pd.DataFrame(top_rows).to_csv(args.output_dir / "top_attention_tiles.csv", index=False); pd.DataFrame(errors).to_csv(args.output_dir / "errors.csv", index=False)
    case_values = per_slide.groupby("case_id")["rho_attention_centroid_similarity"].median().to_numpy()
    rng = np.random.default_rng(args.seed); samples = np.array([np.median(rng.choice(case_values, len(case_values), replace=True)) for _ in range(args.n_bootstrap)])
    aggregate = {"n_slides": len(per_slide), "n_cases": len(case_values), "median_rho": float(np.median(case_values)), "bootstrap95_median_rho": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], "median_similarity_lift_top_attention": float(np.median(per_slide["top_attention_mean_centroid_similarity"] - per_slide["mean_centroid_similarity"])), "top_fraction": args.top_fraction, "attention_key": args.attention_key, "attention_reduction": "mean_over_layers_and_heads"}
    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
