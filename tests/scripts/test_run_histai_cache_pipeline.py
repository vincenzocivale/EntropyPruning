from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from scripts.run_histai_cache_pipeline import _validate_cache_files
from src.wsi_pipeline.cache_contracts import TileCacheSpec
from src.wsi_pipeline.cache_io import TileCacheWriter


def test_validate_cache_files_reports_bad_files_in_order(tmp_path: Path) -> None:
    spec = TileCacheSpec(tile_encoder="test")
    good = tmp_path / "good.h5"
    with TileCacheWriter(good, spec, slide_id="s0", case_id="c0", expected_n=1) as writer:
        writer.append(
            coords=np.asarray([[0, 0]]),
            final_attention=np.ones((1, 4), dtype=np.float16) / 4,
            tile_embeddings=np.ones((1, 3), dtype=np.float16),
        )
    bad_path = tmp_path / "bad.h5"
    with h5py.File(bad_path, "w"):
        pass  # never marked complete -> validate_cache must reject it

    bad = _validate_cache_files([good, bad_path], max_workers=4)

    assert [path for path, _reason in bad] == [str(bad_path)]


def test_validate_cache_files_returns_empty_for_all_good(tmp_path: Path) -> None:
    spec = TileCacheSpec(tile_encoder="test")
    paths = []
    for index in range(3):
        path = tmp_path / f"good_{index}.h5"
        with TileCacheWriter(path, spec, slide_id=f"s{index}", case_id=f"c{index}", expected_n=1) as writer:
            writer.append(
                coords=np.asarray([[0, 0]]),
                final_attention=np.ones((1, 4), dtype=np.float16) / 4,
                tile_embeddings=np.ones((1, 3), dtype=np.float16),
            )
        paths.append(path)

    assert _validate_cache_files(paths, max_workers=4) == []
