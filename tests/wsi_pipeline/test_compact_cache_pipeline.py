import csv
from pathlib import Path

import numpy as np
import pytest

from src.data.wsi.manifest import SlideRecord, write_manifest
from src.models.online_tile_eaf import OnlineAttentionTeacher
from src.wsi_pipeline.cache_contracts import TileCacheSpec
from src.wsi_pipeline.cache_index import build_tile_cache_index, read_tile_cache_index
from src.wsi_pipeline.cache_io import TileCacheWriter, validate_cache


def _write_coords(path: Path, coords: np.ndarray) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=coords)


def test_preallocated_writer_requires_exact_count(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"
    spec = TileCacheSpec(tile_encoder="test")
    with pytest.raises(RuntimeError, match="wrote 1, expected 2"):
        with TileCacheWriter(
            path, spec, slide_id="s1", case_id="c1", expected_n=2
        ) as writer:
            writer.append(
                coords=np.asarray([[0, 0]]),
                final_attention=np.ones((1, 4), dtype=np.float16) / 4,
                tile_embeddings=np.ones((1, 3), dtype=np.float16),
            )
    assert not path.exists()


def test_build_tile_cache_index_validates_coords(tmp_path: Path) -> None:
    coords = np.asarray([[0, 0], [8, 16]], dtype=np.int32)
    coords_path = tmp_path / "coords.h5"
    raw_path = tmp_path / "slide.svs"
    raw_path.write_bytes(b"placeholder")
    _write_coords(coords_path, coords)
    slides_path = write_manifest(
        tmp_path / "slides.csv",
        [
            SlideRecord(
                slide_id="s1", case_id="c1", source="histai",
                raw_path=str(raw_path), coords_path=str(coords_path), split="train",
            )
        ],
    )
    cache_root = tmp_path / "caches"
    cache_root.mkdir()
    spec = TileCacheSpec(tile_encoder="test")
    with TileCacheWriter(
        cache_root / "s1.h5", spec, slide_id="s1", case_id="c1", expected_n=2
    ) as writer:
        writer.append(
            coords=coords,
            final_attention=np.ones((2, 4), dtype=np.float16) / 4,
            tile_embeddings=np.ones((2, 3), dtype=np.float16),
        )

    output = tmp_path / "index.csv"
    rows = build_tile_cache_index(
        slides_path, [cache_root], output, expected_cache_id=spec.cache_id
    )
    assert len(rows) == 1
    assert read_tile_cache_index(output)["s1"] == cache_root / "s1.h5"
    assert validate_cache(cache_root / "s1.h5")["n_tiles"] == 2


def test_read_tile_cache_index_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("slide_id", "cache_path", "n_tiles", "cache_id")
        )
        writer.writeheader()
        writer.writerows(
            [
                {"slide_id": "s1", "cache_path": "a.h5", "n_tiles": "1", "cache_id": "x"},
                {"slide_id": "s1", "cache_path": "b.h5", "n_tiles": "1", "cache_id": "x"},
            ]
        )
    with pytest.raises(ValueError, match="duplicate"):
        read_tile_cache_index(path)


def test_online_teacher_extract_early_skips_later_blocks() -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return value + 1

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block(), Block()])

        def forward_features(self, value):
            for block in self.blocks:
                value = block(value)
            return value

    class Adapter:
        num_prefix_tokens = 1
        n_blocks = 3

        def __init__(self, model):
            self.model = model

        def get_blocks(self):
            return self.model.blocks

    backbone = Backbone()
    teacher = OnlineAttentionTeacher(backbone, Adapter(backbone), 1, 2)
    source = teacher.extract_early(torch.zeros(2, 5, 3))
    assert source.shape == (2, 4, 3)
    assert [block.calls for block in backbone.blocks] == [1, 1, 0]
