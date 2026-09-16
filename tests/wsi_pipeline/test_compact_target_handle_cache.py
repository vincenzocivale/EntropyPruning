"""Compact-cache loaders keep one target file handle per slide in a batch."""

from types import SimpleNamespace

import pytest


def test_target_handle_capacity_covers_batch_group(monkeypatch):
    torch = pytest.importorskip("torch")
    from src.wsi_pipeline import compact_cache_dataset as module

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, records, _cache_paths, _transform, **kwargs):
            self.records = records
            self.target_cache_size = kwargs["target_cache_size"]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return index

    monkeypatch.setattr(module, "CompactCachedWSITileDataset", FakeDataset)
    records = [SimpleNamespace(cohort="A", coord_count=4) for _ in range(16)]
    train, val, _, _ = module.build_compact_cache_tile_loaders(
        {"train": records, "val": records}, {}, None,
        batch_size=32, slides_per_batch=16,
        train_slides_per_epoch=16, train_tiles_per_slide=4,
        val_slides_per_epoch=16, val_tiles_per_slide=4,
        num_workers=0, prefetch_factor=2, slide_cache_size=8,
        openslide_cache_bytes=0, cohort_balance_power=0.5, seed=17,
    )
    assert train.dataset.target_cache_size == 16
    assert val.dataset.target_cache_size == 16
