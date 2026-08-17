from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/data/wsi_tile_stream.py"
SPEC = importlib.util.spec_from_file_location("eaf_wsi_tile_stream", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SlideRecord = MODULE.SlideRecord
WSIBalancedBatchSampler = MODULE.WSIBalancedBatchSampler
WSITileDataset = MODULE.WSITileDataset
inspect_coordinate_file = MODULE.inspect_coordinate_file
load_wsi_manifest = MODULE.load_wsi_manifest


def _write_coords(
    path: Path, count: int = 12, patch_size_level0: int | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "coords",
            data=np.arange(count * 2, dtype=np.int64).reshape(count, 2),
        )
        dataset.attrs["patch_level"] = 0
        dataset.attrs["patch_size"] = 512
        if patch_size_level0 is not None:
            dataset.attrs["patch_size_level0"] = patch_size_level0


def test_manifest_paths_and_case_disjoint_split(tmp_path: Path) -> None:
    rows = []
    for case_index in range(6):
        split = "train" if case_index < 4 else "val"
        for slide_index in range(2):
            slide_id = f"case{case_index}_slide{slide_index}"
            raw = tmp_path / "sources" / f"{slide_id}.svs"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.touch()
            coords = tmp_path / "coords" / f"{slide_id}_patches.h5"
            _write_coords(coords, count=10 + slide_index)
            rows.append(
                {
                    "slide_id": slide_id,
                    "case_id": f"case{case_index}",
                    "cohort": "TCGA-A" if case_index % 2 == 0 else "TCGA-B",
                    "slide_group": "diagnostic",
                    "include_in_pretraining": "true",
                    "coords_available": "true",
                    "preprocessing_status": "coords_ready",
                    "raw_path": str(raw.relative_to(tmp_path)),
                    "coords_path": str(coords.relative_to(tmp_path)),
                    "split": split,
                }
            )

    manifest = tmp_path / "slides.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split_records = load_wsi_manifest(manifest, tmp_path)
    assert len(split_records["train"]) == 8
    assert len(split_records["val"]) == 4
    assert not split_records["test"]

    train_cases = {record.case_id for record in split_records["train"]}
    val_cases = {record.case_id for record in split_records["val"]}
    assert train_cases.isdisjoint(val_cases)
    assert all(record.raw_path.is_absolute() for record in split_records["train"])
    assert all(record.coord_count >= 10 for record in split_records["train"])

    count, level, patch_size, coordinate_window_size = inspect_coordinate_file(
        split_records["train"][0].coords_path
    )
    assert count in {10, 11}
    assert level == 0
    assert patch_size == 512


def test_manifest_maps_strict_holdout_to_test(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "val", "holdout"):
        slide_id = f"slide_{split}"
        raw = tmp_path / f"{slide_id}.svs"
        raw.touch()
        coords = tmp_path / f"{slide_id}.h5"
        _write_coords(coords)
        rows.append(
            {
                "slide_id": slide_id,
                "case_id": slide_id,
                "cohort": "A",
                "raw_path": str(raw),
                "coords_path": str(coords),
                "split": split,
            }
        )
    manifest = tmp_path / "slides.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    splits = load_wsi_manifest(manifest, tmp_path)
    assert [row.slide_id for row in splits["test"]] == ["slide_holdout"]



def test_trident_level0_patch_size_takes_precedence(tmp_path: Path) -> None:
    path = tmp_path / "slide_patches.h5"
    _write_coords(path, count=7, patch_size_level0=1024)
    count, level, patch_size, coordinate_window_size = inspect_coordinate_file(path)
    assert count == 7
    assert level == 0
    assert patch_size == 1024
    assert coordinate_window_size == 1024

    _, _, encoder_crop, parent_window = inspect_coordinate_file(
        path, crop_size_at_target_mag=256
    )
    assert encoder_crop == 512
    assert parent_window == 1024


def test_dataset_center_crops_encoder_fov(tmp_path: Path, monkeypatch) -> None:
    coords_path = tmp_path / "coords.h5"
    with h5py.File(coords_path, "w") as handle:
        handle.create_dataset("coords", data=np.asarray([[100, 200]], dtype=np.int64))

    calls = []

    class FakeSlide:
        def __init__(self, path: str) -> None:
            self.path = path

        def read_region(self, location, level, size):
            calls.append((location, level, size))
            return Image.new("RGBA", size, (255, 255, 255, 255))

        def close(self) -> None:
            pass

    class FakeOpenSlideModule:
        OpenSlide = FakeSlide

    monkeypatch.setitem(sys.modules, "openslide", FakeOpenSlideModule())
    raw_path = tmp_path / "slide.svs"
    raw_path.touch()
    record = SlideRecord(
        slide_id="slide",
        case_id="case",
        cohort="A",
        raw_path=raw_path,
        coords_path=coords_path,
        split="val",
        coord_count=1,
        patch_level=0,
        patch_size=512,
        coordinate_window_size=1024,
    )
    dataset = WSITileDataset([record], transform=lambda image: image, augment=False)
    image, slide_index = dataset[(0, 0)]
    assert image.mode == "RGB"
    assert slide_index == 0
    assert calls == [((356, 456), 0, (512, 512))]


def test_dataset_installs_worker_local_openslide_cache(tmp_path: Path, monkeypatch) -> None:
    coords_path = tmp_path / "coords.h5"
    _write_coords(coords_path, count=1)
    installed = []

    class FakeCache:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity

    class FakeSlide:
        def __init__(self, _path: str) -> None:
            pass

        def set_cache(self, cache) -> None:
            installed.append(cache.capacity)

        def close(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules, "openslide",
        type("FakeOpenSlideModule", (), {"OpenSlide": FakeSlide, "OpenSlideCache": FakeCache})(),
    )
    record = _record(0, "A")
    record = SlideRecord(**{**record.__dict__, "raw_path": tmp_path / "slide.svs", "coords_path": coords_path, "coord_count": 1})
    dataset = WSITileDataset(
        [record], transform=None, augment=False,
        openslide_cache_bytes=512 * 2**20,
    )

    dataset._get_slide(record.raw_path)

    assert installed == [512 * 2**20]

def _record(index: int, cohort: str) -> SlideRecord:
    return SlideRecord(
        slide_id=f"slide-{index}",
        case_id=f"case-{index}",
        cohort=cohort,
        raw_path=Path(f"/raw/slide-{index}.svs"),
        coords_path=Path(f"/coords/slide-{index}.h5"),
        split="train",
        coord_count=32,
        patch_level=0,
        patch_size=512,
        coordinate_window_size=512,
    )


def test_sampler_batches_and_rotating_wsi_coverage() -> None:
    records = [
        *[_record(index, "A") for index in range(8)],
        *[_record(8 + index, "B") for index in range(8)],
    ]
    sampler = WSIBalancedBatchSampler(
        records,
        batch_size=8,
        slides_per_batch=2,
        slides_per_epoch=8,
        tiles_per_slide=4,
        seed=17,
        cohort_balance_power=0.5,
    )

    epoch_slide_sets = []
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        assert len(batches) == len(sampler) == 4
        selected = set()
        for batch in batches:
            assert len(batch) == 8
            slide_counts = {}
            for slide_index, coord_index in batch:
                selected.add(slide_index)
                slide_counts[slide_index] = slide_counts.get(slide_index, 0) + 1
                assert 0 <= coord_index < records[slide_index].coord_count
            assert sorted(slide_counts.values()) == [4, 4]
        assert len(selected) == 8
        assert sampler.last_summary["scheduled_tiles"] == 32
        epoch_slide_sets.append(selected)

    # Equal cohorts and a 50% per-epoch budget should cover every slide after
    # two deterministic rotations, not repeatedly redraw the same small subset.
    assert len(epoch_slide_sets[0] | epoch_slide_sets[1]) == 16


def test_sampler_does_not_pad_non_divisible_tile_budget() -> None:
    records = [_record(index, "A") for index in range(4)]
    sampler = WSIBalancedBatchSampler(
        records,
        batch_size=16,
        slides_per_batch=4,
        slides_per_epoch=4,
        tiles_per_slide=5,
        seed=3,
    )

    batches = list(sampler)

    assert [len(batch) for batch in batches] == [16, 4]
    counts = {}
    for batch in batches:
        for slide_index, _ in batch:
            counts[slide_index] = counts.get(slide_index, 0) + 1
    assert counts == {0: 5, 1: 5, 2: 5, 3: 5}
    assert sampler.last_summary["scheduled_tiles"] == 20


def test_dataset_batched_reads_restore_sampler_order(tmp_path: Path, monkeypatch) -> None:
    coords_path = tmp_path / "coords.h5"
    with h5py.File(coords_path, "w") as handle:
        handle.create_dataset(
            "coords", data=np.asarray([[300, 0], [100, 0], [200, 0]], dtype=np.int64)
        )
    read_locations = []

    class FakeSlide:
        def __init__(self, _path: str) -> None:
            pass

        def read_region(self, location, _level, size):
            read_locations.append(location)
            return Image.new("RGBA", size, (location[0] % 255, 0, 0, 255))

        def close(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "openslide",
        type("FakeOpenSlideModule", (), {"OpenSlide": FakeSlide})(),
    )
    record = SlideRecord(
        slide_id="slide", case_id="case", cohort="A",
        raw_path=tmp_path / "slide.svs", coords_path=coords_path, split="train",
        coord_count=3, patch_level=0, patch_size=8, coordinate_window_size=8,
    )
    dataset = WSITileDataset(
        [record], transform=lambda image: image.getpixel((0, 0))[0], augment=False
    )

    result = dataset.__getitems__([(0, 0), (0, 1), (0, 2)])

    assert read_locations == [(100, 0), (200, 0), (300, 0)]
    assert [value for value, _ in result] == [300 % 255, 100, 200]
