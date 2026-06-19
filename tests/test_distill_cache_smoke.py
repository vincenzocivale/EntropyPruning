"""Fast, CPU-only smoke tests for src/collection/distill_cache.py.

Uses the same fake timm-like ViT as tests/test_distilled_pruned_smoke.py and
monkeypatches build_thunder_loaders so no real dataset/model checkpoint is
needed. Checks the property that matters most: the cached seq_prune /
teacher_cls / teacher_patches must exactly match what a plain, unmodified
forward pass through the same frozen backbone would have produced --
forward_from_seq's whole premise is that this cache is a faithful, reusable
substitute for re-running the frozen blocks every step.
"""

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.collection.distill_cache import build_distill_cache, distill_cache_is_valid
from src.models import ThunderBackboneAdapter
from tests._fakes import FakeViT

D, N_BLOCKS, N_PATCHES, NUM_PREFIX, PRUNE_LAYER, SEQ_LEN = 8, 4, 5, 1, 1, 6


def _loader(imgs, labels, batch_size=3):
    return DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=False)


def _patch_loaders(monkeypatch, train_loader, val_loader, test_loader, call_count=None):
    def _fake(*args, **kwargs):
        if call_count is not None:
            call_count["n"] += 1
        return train_loader, val_loader, test_loader, None, None

    monkeypatch.setattr("src.collection.distill_cache.build_thunder_loaders", _fake)


def test_distill_cache_is_valid_false_when_missing(tmp_path):
    assert distill_cache_is_valid(tmp_path / "missing.h5", N_PATCHES, D, NUM_PREFIX) is False


def test_build_distill_cache_captures_correct_intermediate_and_final_outputs(monkeypatch, tmp_path):
    torch.manual_seed(0)
    teacher = FakeViT(D, N_BLOCKS, N_PATCHES, NUM_PREFIX)
    teacher.eval()
    adapter = ThunderBackboneAdapter(teacher)

    imgs_train = torch.randn(6, SEQ_LEN, D)
    labels_train = torch.arange(6)
    _patch_loaders(
        monkeypatch,
        _loader(imgs_train, labels_train),
        _loader(torch.randn(2, SEQ_LEN, D), torch.arange(2)),
        _loader(torch.randn(2, SEQ_LEN, D), torch.arange(2)),
    )

    with torch.no_grad():
        expected_seq_prune = imgs_train
        for blk in teacher.blocks[:PRUNE_LAYER + 1]:
            expected_seq_prune = blk(expected_seq_prune)
        expected_final = expected_seq_prune
        for blk in teacher.blocks[PRUNE_LAYER + 1:]:
            expected_final = blk(expected_final)
        expected_final = teacher.norm(expected_final)

    save_path = tmp_path / "cache.h5"
    build_distill_cache(
        teacher, adapter, transform=None, dataset_name="fake", base_data_folder="unused",
        save_path=save_path, device=torch.device("cpu"), prune_layer=PRUNE_LAYER,
        batch_size=3, num_workers=0,
    )

    with h5py.File(save_path, "r") as f:
        grp = f["train"]
        assert grp["seq_prune"].shape == (6, SEQ_LEN, D)
        assert grp["teacher_cls"].shape == (6, D)
        assert grp["teacher_patches"].shape == (6, N_PATCHES, D)

        seq_prune = torch.from_numpy(grp["seq_prune"][:]).float()
        teacher_cls = torch.from_numpy(grp["teacher_cls"][:]).float()
        teacher_patches = torch.from_numpy(grp["teacher_patches"][:]).float()

    # fp16 storage -> loose tolerance
    assert torch.allclose(seq_prune, expected_seq_prune, atol=1e-2, rtol=1e-2)
    assert torch.allclose(teacher_cls, expected_final[:, 0], atol=1e-2, rtol=1e-2)
    assert torch.allclose(teacher_patches, expected_final[:, NUM_PREFIX:], atol=1e-2, rtol=1e-2)


def test_build_distill_cache_skips_extraction_once_valid(monkeypatch, tmp_path):
    teacher = FakeViT(D, N_BLOCKS, N_PATCHES, NUM_PREFIX)
    adapter = ThunderBackboneAdapter(teacher)
    call_count = {"n": 0}
    _patch_loaders(
        monkeypatch,
        _loader(torch.randn(4, SEQ_LEN, D), torch.arange(4)),
        _loader(torch.randn(2, SEQ_LEN, D), torch.arange(2)),
        _loader(torch.randn(2, SEQ_LEN, D), torch.arange(2)),
        call_count=call_count,
    )
    save_path = tmp_path / "cache.h5"
    kwargs = dict(
        teacher=teacher, adapter=adapter, transform=None, dataset_name="fake",
        base_data_folder="unused", save_path=save_path, device=torch.device("cpu"),
        prune_layer=PRUNE_LAYER, batch_size=2, num_workers=0,
    )

    build_distill_cache(**kwargs)
    assert call_count["n"] == 1

    build_distill_cache(**kwargs)
    assert call_count["n"] == 1, "second call should skip extraction (cache already valid)"
