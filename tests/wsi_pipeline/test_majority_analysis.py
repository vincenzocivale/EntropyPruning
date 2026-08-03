from pathlib import Path

import numpy as np

from src.wsi_pipeline.io import TileFeatureRecord, WSIOutputRecord, write_tile_feature_record, write_wsi_output_record
from src.wsi_pipeline.majority_analysis import MajorityAnalysisConfig, analyze_slide


def _make_clustered(seed=17):
    rng = np.random.default_rng(seed)
    majority_center = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    majority = majority_center + rng.normal(0, 0.03, size=(160, 8))
    minority = rng.normal(0, 1.0, size=(40, 8))
    x = np.vstack([majority, minority]).astype(np.float32)
    return x


def test_majority_attention_positive(tmp_path: Path):
    x = _make_clustered()
    coords = np.column_stack([np.arange(len(x)) * 512, np.zeros(len(x))]).astype(np.int32)
    attention = np.concatenate([np.full(160, 3.0), np.full(40, 1.0)])
    attention /= attention.sum()
    feature_path = tmp_path / "features.h5"
    output_path = tmp_path / "output.h5"
    write_tile_feature_record(feature_path, TileFeatureRecord("slide", coords, {"final": x}))
    write_wsi_output_record(output_path, WSIOutputRecord("slide", np.ones(4), coords, {"probs": attention}))
    row, _ = analyze_slide(
        feature_path,
        output_path,
        config=MajorityAnalysisConfig(knn_values=(16,), cluster_values=(2,), n_permutations=10),
    )
    assert row["rho_attention_knn_k16"] > 0
    assert row["k2_dominant_enrichment"] > 1


def test_minority_attention_refutes(tmp_path: Path):
    x = _make_clustered()
    coords = np.column_stack([np.arange(len(x)) * 512, np.zeros(len(x))]).astype(np.int32)
    attention = np.concatenate([np.full(160, 1.0), np.full(40, 8.0)])
    attention /= attention.sum()
    feature_path = tmp_path / "features.h5"
    output_path = tmp_path / "output.h5"
    write_tile_feature_record(feature_path, TileFeatureRecord("slide", coords, {"final": x}))
    write_wsi_output_record(output_path, WSIOutputRecord("slide", np.ones(4), coords, {"probs": attention}))
    row, _ = analyze_slide(
        feature_path,
        output_path,
        config=MajorityAnalysisConfig(knn_values=(16,), cluster_values=(2,), n_permutations=10),
    )
    assert row["k2_dominant_enrichment"] < 1
