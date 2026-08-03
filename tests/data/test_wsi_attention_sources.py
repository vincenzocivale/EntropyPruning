from __future__ import annotations

import csv

import numpy as np
import torch

from src.data.wsi.attention import ManifestAttentionSource, WSIAttention, align_attention_to_bag
from src.data.wsi.bag import WSIBag


def test_manifest_source_reads_legacy_target_path(tmp_path) -> None:
    attention_path = tmp_path / "signal.npz"
    np.savez(attention_path, importance=np.arange(6, dtype=np.float32).reshape(2, 3))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", "target_path", "tile_axis"])
        writer.writeheader()
        writer.writerow({"slide_id": "s1", "target_path": attention_path.name, "tile_axis": 1})

    source = ManifestAttentionSource(manifest, reduction="mean")
    record = source.read("s1", n_tiles=3)
    assert torch.allclose(record.values, torch.tensor([1.5, 2.5, 3.5]))


def test_coordinate_alignment_reorders_attention() -> None:
    coords = torch.tensor([[0, 0], [1, 0], [2, 0]])
    bag = WSIBag(
        slide_id="s1",
        tile_features=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        coords=coords,
    )
    attention = WSIAttention(
        slide_id="s1",
        values=torch.tensor([30.0, 10.0, 20.0]),
        coords=torch.tensor([[2, 0], [0, 0], [1, 0]]),
    )
    aligned_bag, aligned_attention = align_attention_to_bag(bag, attention, mode="coords")
    assert torch.equal(aligned_bag.coords, coords)
    assert torch.equal(aligned_attention.values, torch.tensor([10.0, 20.0, 30.0]))
