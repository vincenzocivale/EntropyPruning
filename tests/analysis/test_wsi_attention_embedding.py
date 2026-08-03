from __future__ import annotations

import numpy as np
import torch

from src.analysis.wsi_attention_embedding import (
    aggregate_slide_metrics,
    analyze_attention_embedding_relation,
    flatten_numeric_metrics,
)


def test_attention_distance_relation_is_recovered() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
        ]
    )
    centroid = features.mean(dim=0, keepdim=True)
    attention = (features - centroid).norm(dim=1)
    summary, arrays = analyze_attention_embedding_relation(features, attention)

    metrics = summary["descriptors"]["euclidean_distance_to_centroid"]
    assert metrics["spearman"] > 0.99
    assert metrics["quadratic_r2"] > 0.99
    assert arrays["attention_probability"].shape == (5,)
    assert np.isclose(arrays["attention_probability"].sum(), 1.0)


def test_flatten_and_aggregate_metrics() -> None:
    features = torch.eye(4)
    attention = torch.tensor([0.1, 0.2, 0.3, 0.4])
    summary, _ = analyze_attention_embedding_relation(features, attention)
    flattened = flatten_numeric_metrics(summary)
    assert "attention.entropy" in flattened
    assert not any("quantile_bins" in key for key in flattened)

    aggregate = aggregate_slide_metrics([flattened, flattened])
    assert aggregate["n_slides"] == 2
    assert aggregate["metrics"]["attention.entropy"]["count"] == 2
