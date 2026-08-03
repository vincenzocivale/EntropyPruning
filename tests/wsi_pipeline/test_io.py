from pathlib import Path

import numpy as np

from src.wsi_pipeline.io import (
    TileFeatureRecord,
    WSIOutputRecord,
    read_tile_feature_record,
    read_wsi_output_record,
    write_tile_feature_record,
    write_wsi_output_record,
)


def test_roundtrip(tmp_path: Path):
    coords = np.array([[0, 0], [512, 0], [0, 512]], dtype=np.int32)
    layer2 = np.arange(12, dtype=np.float32).reshape(3, 4)
    final = np.arange(9, dtype=np.float32).reshape(3, 3)
    feature_path = tmp_path / "slide.h5"
    write_tile_feature_record(
        feature_path,
        TileFeatureRecord("slide", coords, {"layer2": layer2, "final": final}),
    )
    loaded = read_tile_feature_record(feature_path)
    assert loaded.slide_id == "slide"
    np.testing.assert_array_equal(loaded.coords, coords)
    np.testing.assert_allclose(loaded.embeddings["final"], final, rtol=1e-3, atol=1e-3)

    output_path = tmp_path / "output.h5"
    probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    tile_to_token = np.array([1, 1, 2], dtype=np.int64)
    write_wsi_output_record(
        output_path,
        WSIOutputRecord(
            "slide",
            np.ones(5),
            coords,
            {"probs": probs},
            {"tile_to_token": tile_to_token},
        ),
    )
    output = read_wsi_output_record(output_path)
    np.testing.assert_allclose(output.attention["probs"], probs, rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(output.auxiliary["tile_to_token"], tile_to_token)
