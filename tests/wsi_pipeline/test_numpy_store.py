from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")
import h5py

from src.wsi_pipeline.numpy_store import NumpyStoreWriter, convert_h5, read_array, read_metadata, verify_conversion
from src.wsi_pipeline.io import WSIOutputRecord, read_wsi_output_record, write_wsi_output_record
from src.wsi_pipeline.cache_contracts import TileCacheSpec
from src.wsi_pipeline.cache_io import TileCacheWriter, validate_cache
from src.data.wsi_tile_stream import inspect_coordinate_file
from src.data.wsi.trident import read_trident_coords


def test_conversion_preserves_nested_arrays_and_metadata(tmp_path):
    source = tmp_path / "slide.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs["slide_id"] = "slide"
        handle.attrs["complete"] = True
        coords = handle.create_dataset("coords", data=np.arange(12, dtype=np.int32).reshape(6, 2))
        coords.attrs["patch_size_level0"] = 512
        handle.create_dataset("attention/global", data=np.arange(24, dtype=np.float16).reshape(2, 2, 6))
    target = convert_h5(source)
    verify_conversion(source, target)
    assert read_metadata(target)["slide_id"] == "slide"
    assert read_metadata(target)["_hdf5_object_attrs"]["coords"]["patch_size_level0"] == 512
    assert np.array_equal(read_array(target, "attention/global", mmap=True),
                          np.arange(24, dtype=np.float16).reshape(2, 2, 6))
    with pytest.raises(FileExistsError):
        with NumpyStoreWriter(target, {}):
            pass


def test_numpy_wsi_output_roundtrip(tmp_path):
    path = tmp_path / "slide.npyd"
    write_wsi_output_record(path, WSIOutputRecord(
        slide_id="slide", slide_embedding=np.array([1, 2], dtype=np.float32),
        coords=np.array([[0, 1]], dtype=np.int32),
        attention={"global_to_tiles": np.array([0.5], dtype=np.float32)},
        auxiliary={"hidden_layer_000": np.ones((1, 2), dtype=np.float16)},
        metadata={"wsi_model": "titan"},
    ))
    record = read_wsi_output_record(path)
    assert record.slide_id == "slide"
    assert record.metadata["wsi_model"] == "titan"
    assert record.attention["global_to_tiles"].dtype == np.float16


def test_numpy_tile_cache_is_complete_and_valid(tmp_path):
    spec = TileCacheSpec(tile_encoder="conch_v15", model_revision="test", early_layer=2,
                         input_mag=20, patch_size=512, stride=512, input_mpp=0.5,
                         dtype="float16", dataset="unit")
    path = tmp_path / "slide.npyd"
    with TileCacheWriter(path, spec, slide_id="slide", case_id="case", expected_n=2) as writer:
        writer.append(coords=np.array([[0, 0], [1, 1]]),
                      final_attention=np.ones((2, 4)), tile_embeddings=np.ones((2, 3)))
    info = validate_cache(path, expected_kind="tile_eaf")
    assert info["n_tiles"] == 2
    assert info["spec"]["cache_id"] == spec.cache_id


def test_converted_trident_coords_keep_patch_metadata(tmp_path):
    source = tmp_path / "slide_patches.h5"
    with h5py.File(source, "w") as handle:
        dataset = handle.create_dataset("coords", data=np.array([[0, 1], [2, 3]], dtype=np.int32))
        dataset.attrs["patch_size_level0"] = 1024
    converted = convert_h5(source)
    source.unlink()
    assert inspect_coordinate_file(converted, default_patch_size=512) == (2, 0, 1024, 1024)
    assert read_trident_coords(converted).tolist() == [[0, 1], [2, 3]]
