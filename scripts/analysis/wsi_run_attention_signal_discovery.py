#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.wsi_pipeline.attention_signal import (
    FeatureViewSpec,
    SignalDiscoveryConfig,
    run_attention_signal_discovery,
)


def _feature_view(values: list[str]) -> FeatureViewSpec:
    name, manifest, feature_set_id, feature_key = values
    return FeatureViewSpec(
        name=name,
        manifest=Path(manifest),
        feature_set_id=None if feature_set_id in {"-", "none", "None"} else feature_set_id,
        feature_key=feature_key,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover pre-TITAN signals of final TITAN tile attention using case-disjoint kNN retrieval, "
            "linear probes, positive/negative prototypes, and cheap WSI-context features."
        )
    )
    parser.add_argument("--attention-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-view",
        action="append",
        nargs=4,
        metavar=("NAME", "MANIFEST", "FEATURE_SET_ID", "FEATURE_KEY"),
        required=True,
        help=(
            "Repeat for each view. Use FEATURE_SET_ID '-' when MANIFEST already contains one feature set. "
            "For legacy HDF5 files use FEATURE_KEY 'final'."
        ),
    )
    parser.add_argument("--attention-model-filter", help="Optional substring filter on model/model_name/model_id")
    parser.add_argument("--attention-key", default="global_to_tiles_mass_share")
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--head-groups", type=int, default=6)
    parser.add_argument("--positive-fraction", type=float, default=0.10)
    parser.add_argument("--negative-fraction", type=float, default=0.50)
    parser.add_argument("--retention", nargs="+", type=float, default=(0.30, 0.40, 0.50, 0.60))
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-tiles", type=int, default=100_000)
    parser.add_argument("--max-tiles-per-slide", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=256, help="0 keeps the full embedding dimension")
    parser.add_argument("--knn-k", type=int, default=32)
    parser.add_argument("--knn-backend", choices=("auto", "faiss", "sklearn"), default="auto")
    parser.add_argument("--prototype-count", type=int, default=16)
    parser.add_argument("--prototype-temperature", type=float, default=10.0)
    parser.add_argument("--slide-prototype-count", type=int, default=16)
    parser.add_argument("--max-slide-fit-tiles", type=int, default=4096)
    parser.add_argument(
        "--spatial-neighbors",
        type=int,
        default=0,
        help="Add embedding residual to k spatial neighbors; 0 disables this more expensive context feature.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("knn", "linear", "prototype", "context_linear"),
        default=("knn", "linear", "prototype", "context_linear"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    specs = [_feature_view(values) for values in args.feature_view]
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Feature-view names must be unique")
    config = SignalDiscoveryConfig(
        attention_key=args.attention_key,
        target_layer=args.target_layer,
        n_head_groups=args.head_groups,
        positive_fraction=args.positive_fraction,
        negative_fraction=args.negative_fraction,
        retention_fractions=tuple(args.retention),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_tiles=args.max_train_tiles,
        max_tiles_per_slide=args.max_tiles_per_slide,
        projection_dim=args.projection_dim,
        knn_k=args.knn_k,
        knn_backend=args.knn_backend,
        prototype_count=args.prototype_count,
        prototype_temperature=args.prototype_temperature,
        slide_prototype_count=args.slide_prototype_count,
        max_slide_fit_tiles=args.max_slide_fit_tiles,
        spatial_neighbors=args.spatial_neighbors,
        bootstrap_replicates=args.bootstrap_replicates,
        methods=tuple(args.methods),
    )
    summary = run_attention_signal_discovery(
        attention_manifest=args.attention_manifest,
        feature_views=specs,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        config=config,
        model_filter=args.attention_model_filter,
        resume=args.resume,
    )
    print(f"slides={summary['n_slides']} cases={summary['n_cases']}")
    for name, view in summary["selection"]["views"].items():
        print(f"{name}: {view}")
    print(f"results={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
