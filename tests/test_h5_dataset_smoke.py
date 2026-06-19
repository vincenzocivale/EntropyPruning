"""CPU-only smoke tests for the H5-backed dataset/cache infrastructure shared
by Phase 2 (forecaster) and Phase 3 (distillation) training.

Exercises both schemas ``BlockShuffleH5Dataset`` must support after being
generalized to a variable ``read_block()`` arity: the original 3-tensor
``H5ForecastDataset``/``MultiH5ForecastDataset`` (``emb, target, label``)
used by ``train_forecaster*.py`` (regression coverage -- this class had no
prior tests), and the new 3-tensor ``DistillH5Dataset``/
``MultiDistillH5Dataset`` (``seq_prune, teacher_cls, teacher_patches``) used
by ``distill_pruned.py``.
"""

import h5py
import torch

from src.data import (
    H5ForecastDataset,
    MultiH5ForecastDataset,
    DistillH5Dataset,
    MultiDistillH5Dataset,
    BlockShuffleH5Dataset,
)

N_PATCHES, EMBED_DIM, NUM_PREFIX, LAYER_SOURCE, LAYER_TARGET = 3, 2, 1, 2, 5


def _write_forecast_cache(path, n_rows, offset=0):
    """H5ForecastDataset schema: row i's emb/attn entries are filled with the
    constant ``i + offset``, so it doubles as an identity tag for assertions."""
    with h5py.File(path, "w") as f:
        grp = f.create_group("train")
        grp.create_dataset("labels", data=[i + offset for i in range(n_rows)])
        grp.create_dataset(
            f"emb_layer{LAYER_SOURCE}",
            data=[[[float(i + offset)] * EMBED_DIM] * N_PATCHES for i in range(n_rows)],
        )
        grp.create_dataset(
            f"attn_layer{LAYER_TARGET}",
            data=[[float(i + offset)] * N_PATCHES for i in range(n_rows)],
        )


def _write_distill_cache(path, n_rows, offset=0):
    """DistillH5Dataset schema: same identity-tag convention as above."""
    seq_len = NUM_PREFIX + N_PATCHES
    with h5py.File(path, "w") as f:
        grp = f.create_group("train")
        grp.create_dataset("labels", data=[i + offset for i in range(n_rows)])
        grp.create_dataset(
            "seq_prune",
            data=[[[float(i + offset)] * EMBED_DIM] * seq_len for i in range(n_rows)],
        )
        grp.create_dataset(
            "teacher_cls",
            data=[[float(i + offset)] * EMBED_DIM for i in range(n_rows)],
        )
        grp.create_dataset(
            "teacher_patches",
            data=[[[float(i + offset)] * EMBED_DIM] * N_PATCHES for i in range(n_rows)],
        )


# -- DistillH5Dataset / MultiDistillH5Dataset --------------------------------

def test_distill_h5_dataset_getitem_and_read_block_roundtrip(tmp_path):
    path = tmp_path / "a.h5"
    _write_distill_cache(path, n_rows=4)
    ds = DistillH5Dataset(path, "train")
    assert len(ds) == 4

    seq, cls, patches = ds[2]
    assert seq.shape == (NUM_PREFIX + N_PATCHES, EMBED_DIM)
    assert cls.shape == (EMBED_DIM,)
    assert patches.shape == (N_PATCHES, EMBED_DIM)
    assert torch.allclose(cls, torch.full((EMBED_DIM,), 2.0))

    seq_b, cls_b, patches_b = ds.read_block(1, 3)
    assert seq_b.shape == (2, NUM_PREFIX + N_PATCHES, EMBED_DIM)
    assert torch.allclose(cls_b, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_multi_distill_h5_dataset_concatenates_and_reports_ds_idx(tmp_path):
    path_a, path_b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_distill_cache(path_a, n_rows=3, offset=0)
    _write_distill_cache(path_b, n_rows=2, offset=100)
    multi = MultiDistillH5Dataset({"a": path_a, "b": path_b}, "train")
    assert len(multi) == 5

    _, cls, _, ds_idx = multi[0]
    assert ds_idx == 0
    assert torch.allclose(cls, torch.zeros(EMBED_DIM))

    _, cls, _, ds_idx = multi[4]
    assert ds_idx == 1
    assert torch.allclose(cls, torch.full((EMBED_DIM,), 101.0))


# -- BlockShuffleH5Dataset: generalized to variable read_block() arity ------

def test_block_shuffle_single_distill_dataset_yields_3_tuples_covering_all_rows(tmp_path):
    path = tmp_path / "a.h5"
    _write_distill_cache(path, n_rows=6)
    ds = DistillH5Dataset(path, "train")
    block_ds = BlockShuffleH5Dataset(ds, batch_size=2, micro_block_size=2, drop_last=True)

    seen = set()
    for batch in block_ds:
        assert len(batch) == 3
        seq, cls, patches = batch
        assert seq.shape == (2, NUM_PREFIX + N_PATCHES, EMBED_DIM)
        assert patches.shape == (2, N_PATCHES, EMBED_DIM)
        seen.update(int(v) for v in cls[:, 0].tolist())
    assert seen == set(range(6))


def test_block_shuffle_multi_distill_dataset_yields_4_tuples_with_correct_ds_idx(tmp_path):
    path_a, path_b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_distill_cache(path_a, n_rows=4, offset=0)
    _write_distill_cache(path_b, n_rows=4, offset=100)
    multi = MultiDistillH5Dataset({"a": path_a, "b": path_b}, "train")
    block_ds = BlockShuffleH5Dataset(multi, batch_size=4, micro_block_size=2, drop_last=True)

    n_batches = 0
    for batch in block_ds:
        assert len(batch) == 4
        _, cls, _, ds_idx = batch
        expected_ds_idx = (cls[:, 0] >= 100).long()
        assert torch.equal(ds_idx, expected_ds_idx)
        n_batches += 1
    assert n_batches == len(block_ds)


# -- Regression: original H5ForecastDataset / MultiH5ForecastDataset path ---

def test_block_shuffle_still_works_for_single_h5_forecast_dataset(tmp_path):
    path = tmp_path / "a.h5"
    _write_forecast_cache(path, n_rows=6)
    ds = H5ForecastDataset(path, "train", LAYER_SOURCE, LAYER_TARGET)
    block_ds = BlockShuffleH5Dataset(ds, batch_size=2, micro_block_size=2, drop_last=True)

    seen = set()
    for batch in block_ds:
        assert len(batch) == 3
        emb, target, label = batch
        assert emb.shape == (2, N_PATCHES, EMBED_DIM)
        assert target.shape == (2, N_PATCHES)
        seen.update(int(v) for v in label.tolist())
    assert seen == set(range(6))


def test_block_shuffle_still_works_for_multi_h5_forecast_dataset(tmp_path):
    path_a, path_b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_forecast_cache(path_a, n_rows=4, offset=0)
    _write_forecast_cache(path_b, n_rows=4, offset=100)
    multi = MultiH5ForecastDataset({"a": path_a, "b": path_b}, "train", LAYER_SOURCE, LAYER_TARGET)
    block_ds = BlockShuffleH5Dataset(multi, batch_size=4, micro_block_size=2, drop_last=True)

    for batch in block_ds:
        assert len(batch) == 4
        _, _, label, ds_idx = batch
        expected_ds_idx = (label >= 100).long()
        assert torch.equal(ds_idx, expected_ds_idx)
