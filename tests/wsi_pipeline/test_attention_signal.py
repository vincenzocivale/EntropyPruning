from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.wsi_pipeline.attention_signal import (
    FeatureViewSpec,
    SignalDiscoveryConfig,
    attention_percentiles,
    compute_context_descriptors,
    fit_head_groups,
    group_attention,
    run_attention_signal_discovery,
)


def _write_feature(path: Path, slide_id: str, features: np.ndarray, coords: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "eaf.wsi.tile_features.v2"
        handle.attrs["complete"] = True
        handle.attrs["slide_id"] = slide_id
        handle.create_dataset("coords", data=coords)
        group = handle.create_group("embeddings")
        group.create_dataset("layer2", data=features.astype(np.float16))
        group.create_dataset("final", data=features.astype(np.float16))


def _write_attention(path: Path, slide_id: str, attention: np.ndarray, coords: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "eaf.wsi.fm_output.v1"
        handle.attrs["complete"] = True
        handle.attrs["slide_id"] = slide_id
        handle.create_dataset("slide_embedding", data=np.zeros(8, dtype=np.float16))
        handle.create_dataset("coords", data=coords)
        group = handle.create_group("attention")
        group.create_dataset("global_to_tiles_mass_share", data=attention.astype(np.float16))
        handle.create_group("auxiliary")


def _synthetic_dataset(root: Path, n_slides: int = 18) -> tuple[Path, Path, Path]:
    feature_rows = []
    attention_rows = []
    metadata_rows = []
    rng = np.random.default_rng(7)
    for slide_index in range(n_slides):
        slide_id = f"SLIDE-{slide_index:03d}"
        case_id = f"CASE-{slide_index:03d}"
        project = f"P{slide_index % 3}"
        n_tiles = 96
        features = rng.normal(size=(n_tiles, 16)).astype(np.float32)
        features[:, 0:3] += rng.normal(scale=0.2, size=(1, 3))
        coords = np.column_stack(
            [np.arange(n_tiles, dtype=np.int32) % 12, np.arange(n_tiles, dtype=np.int32) // 12]
        )
        heads = []
        for group in range(3):
            logits = 5.0 * features[:, group]
            for _ in range(2):
                noisy = logits + rng.normal(scale=0.05, size=n_tiles)
                weights = np.exp(noisy - noisy.max())
                heads.append(weights / weights.sum())
        attention = np.stack(heads)[None, :, :]
        feature_path = root / f"{slide_id}.features.h5"
        attention_path = root / f"{slide_id}.attention.h5"
        _write_feature(feature_path, slide_id, features, coords)
        _write_attention(attention_path, slide_id, attention, coords)
        feature_rows.append(
            {
                "slide_id": slide_id,
                "path": feature_path.name,
                "artifact_type": "tile_features",
                "feature_set_id": "synthetic",
                "status": "complete",
            }
        )
        attention_rows.append({"slide_id": slide_id, "path": attention_path.name, "status": "complete"})
        metadata_rows.append({"slide_id": slide_id, "case_id": case_id, "project": project})
    feature_manifest = root / "features.csv"
    attention_manifest = root / "attention.csv"
    metadata = root / "metadata.csv"
    pd.DataFrame(feature_rows).to_csv(feature_manifest, index=False)
    pd.DataFrame(attention_rows).to_csv(attention_manifest, index=False)
    pd.DataFrame(metadata_rows).to_csv(metadata, index=False)
    return feature_manifest, attention_manifest, metadata


def test_group_attention_and_percentiles() -> None:
    attention = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.3, 0.6],
        ],
        dtype=np.float32,
    )
    grouped = group_attention(attention, [[0, 1], [2, 3]])
    assert grouped.shape == (2, 3)
    assert np.allclose(grouped.sum(axis=1), 1.0)
    percentiles = attention_percentiles(grouped)
    assert percentiles.shape == (3, 2)
    assert percentiles[0, 0] == 1.0
    assert percentiles[2, 1] == 1.0


def test_context_descriptors_are_finite() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(50, 12)).astype(np.float32)
    coords = np.column_stack([np.arange(50) % 10, np.arange(50) // 10])
    result = compute_context_descriptors(
        values,
        coords,
        np.arange(50),
        n_prototypes=4,
        max_fit_tiles=50,
        spatial_neighbors=4,
        seed=17,
    )
    assert result.shape == (50, 11)
    assert np.isfinite(result).all()


def test_signal_discovery_end_to_end(tmp_path: Path) -> None:
    feature_manifest, attention_manifest, metadata = _synthetic_dataset(tmp_path)
    output = tmp_path / "results"
    config = SignalDiscoveryConfig(
        n_head_groups=3,
        retention_fractions=(0.5,),
        max_train_tiles=2000,
        max_tiles_per_slide=128,
        projection_dim=16,
        prototype_count=4,
        slide_prototype_count=4,
        max_slide_fit_tiles=96,
        bootstrap_replicates=20,
        methods=("linear", "prototype", "context_linear"),
        seed=17,
    )
    summary = run_attention_signal_discovery(
        attention_manifest=attention_manifest,
        feature_views=[FeatureViewSpec("synthetic", feature_manifest, "synthetic", "final")],
        metadata_path=metadata,
        output_dir=output,
        config=config,
    )
    assert summary["n_slides"] == 18
    groups = summary["head_groups"]["groups"]
    assert sorted(sorted(group) for group in groups) == [[0, 1], [2, 3], [4, 5]]
    aggregate = pd.read_csv(output / "views" / "synthetic" / "aggregate_coverage_metrics.csv")
    test = aggregate[
        (aggregate["analysis_split"] == "test")
        & (aggregate["metric"] == "mean_group_top_recall")
        & np.isclose(aggregate["retention"], 0.5)
    ]
    assert not test.empty
    assert float(test["median"].max()) > 0.70
    assert (output / "selection.json").exists()
    assert (output / "selected_view_comparison.csv").exists()
