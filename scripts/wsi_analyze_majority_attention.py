#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.wsi_pipeline.majority_analysis import MajorityAnalysisConfig, run_majority_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test whether WSI attention favors dense/majority regions of tile-embedding space."
    )
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--artifact-type", default="tile_features")
    parser.add_argument("--feature-set-id")
    parser.add_argument("--wsi-output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-slides", type=int, help="Analyze a deterministic prefix of matched slides for a preliminary run.")
    parser.add_argument("--metadata", type=Path, help="Optional CSV with slide_id,case_id,project")
    parser.add_argument("--feature-key", default="final")
    parser.add_argument("--attention-key", default="probs")
    parser.add_argument(
        "--attention-reduction",
        choices=("mean", "max"),
        default="mean",
        help="Reduce leading layer/head axes when the attention tensor is multi-dimensional.",
    )
    parser.add_argument("--knn", nargs="+", type=int, default=(16, 32, 64))
    parser.add_argument("--clusters", nargs="+", type=int, default=(4, 8, 16))
    parser.add_argument("--top-fractions", nargs="+", type=float, default=(0.05, 0.10, 0.20))
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--discovery-fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    features = pd.read_csv(args.feature_manifest)
    outputs = pd.read_csv(args.wsi_output_manifest)
    if args.artifact_type and "artifact_type" in features.columns:
        features = features[features["artifact_type"] == args.artifact_type]
    if args.feature_set_id is not None:
        if "feature_set_id" not in features.columns:
            raise ValueError("--feature-set-id requested but feature manifest lacks feature_set_id")
        features = features[features["feature_set_id"] == args.feature_set_id]
    if "status" in features.columns:
        features = features[features["status"].isin(["complete", "available", "valid", "skipped"])]
    if "status" in outputs.columns:
        outputs = outputs[outputs["status"].isin(["complete", "skipped"])]
    if "slide_id" not in features or "path" not in features:
        raise ValueError("feature manifest requires slide_id,path")
    if "slide_id" not in outputs or "path" not in outputs:
        raise ValueError("WSI output manifest requires slide_id,path")
    features = features[["slide_id", "path"]].rename(columns={"path": "feature_path"})
    outputs = outputs[["slide_id", "path"]].rename(columns={"path": "output_path"})
    table = features.merge(outputs, on="slide_id", validate="one_to_one")
    if args.max_slides is not None:
        if args.max_slides < 1:
            raise ValueError("--max-slides must be at least 1")
        table = table.sort_values("slide_id").head(args.max_slides)
    for column, manifest in (("feature_path", args.feature_manifest), ("output_path", args.wsi_output_manifest)):
        def resolve(value: str, manifest: Path = manifest) -> str:
            path = Path(value)
            if path.is_absolute():
                return str(path)
            candidates = [
                (manifest.parent / path).resolve(),
                (manifest.parent.parent / path).resolve(),
                (Path.cwd() / path).resolve(),
            ]
            return str(next((candidate for candidate in candidates if candidate.exists()), candidates[0]))
        table[column] = table[column].map(resolve)
    if args.metadata is not None:
        metadata = pd.read_csv(args.metadata)
        if "patient_id" in metadata.columns and "case_id" not in metadata.columns:
            metadata = metadata.rename(columns={"patient_id": "case_id"})
        if "tcga_project" in metadata.columns and "project" not in metadata.columns:
            metadata = metadata.rename(columns={"tcga_project": "project"})
        keep = [column for column in ("slide_id", "case_id", "project") if column in metadata.columns]
        table = table.merge(metadata[keep], on="slide_id", how="left")
    if "case_id" not in table.columns:
        table["case_id"] = table["slide_id"].map(
            lambda value: "-".join(str(value).split("-")[:3]) if str(value).startswith("TCGA-") else str(value)
        )
    config = MajorityAnalysisConfig(
        feature_key=args.feature_key,
        attention_key=args.attention_key,
        attention_reduction=args.attention_reduction,
        knn_values=tuple(args.knn),
        cluster_values=tuple(args.clusters),
        top_fractions=tuple(args.top_fractions),
        n_permutations=args.n_permutations,
        discovery_fraction=args.discovery_fraction,
        seed=args.seed,
    )
    aggregate = run_majority_analysis(table.to_dict("records"), output_dir=args.output_dir, config=config)
    confirmation = aggregate.get("confirmation", {})
    verdict = confirmation.get("verdict", {}).get("supports_majority_hypothesis")
    print(f"confirmation_supports_majority_hypothesis={verdict}")
    print(f"results={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
