from __future__ import annotations

import csv
import json

import numpy as np
import torch

from scripts.analyze_wsi_attention_embeddings import main
from src.data.wsi.bag import WSIBag
from src.data.wsi.h5_feature_store import H5WSIFeatureStore


def test_cli_analyzes_native_cls_attention_manifest(tmp_path) -> None:
    feature_path = tmp_path / "features.h5"
    store = H5WSIFeatureStore(feature_path)
    features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    coords = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    store.write(WSIBag(slide_id="slide/one", tile_features=features, coords=coords))

    # [layer, head, query_token, key_token], with CLS at index 0.
    native = np.zeros((2, 2, 5, 5), dtype=np.float32)
    native[-1, :, 0, 1:] = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
    attention_path = tmp_path / "native.npz"
    np.savez(attention_path, attention=native, coords=coords.numpy())
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["slide_id", "attention_path", "coords_path", "coords_key"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "slide_id": "slide/one",
                "attention_path": attention_path.name,
                "coords_path": attention_path.name,
                "coords_key": "coords",
            }
        )

    output_dir = tmp_path / "audit"
    exit_code = main(
        [
            "--feature-store",
            str(feature_path),
            "--attention-manifest",
            str(manifest_path),
            "--tile-axis",
            "3",
            "--tile-slice-start",
            "1",
            "--attention-select",
            "0=-1",
            "--attention-select",
            "2=0",
            "--alignment",
            "coords",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    aggregate = json.loads((output_dir / "aggregate.json").read_text())
    assert aggregate["n_successful_slides"] == 1
    with (output_dir / "slide_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["slide_id"] == "slide/one"
    assert len(list((output_dir / "tiles").glob("*.npz"))) == 1
