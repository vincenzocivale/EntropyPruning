"""Unit tests for src/data/thunder_multi.py.

Tests are isolated from the real Thunder library via mocks.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_manifest(n_train: int, n_classes: int = 2):
    labels = list(range(n_classes)) * (n_train // n_classes) + list(range(n_train % n_classes))
    images = [f"img_{i}.jpg" for i in range(n_train)]
    return {
        "train": {"images": images, "labels": labels},
        "val": {"images": images[:4], "labels": labels[:4]},
        "test": {"images": images[:2], "labels": labels[:2]},
        "train_few_shot": {},
    }


def _make_splits_dir(tmp_path: Path, datasets: dict) -> Path:
    """Create fake data_splits/ JSON files. datasets = {name: n_train_samples}."""
    splits = tmp_path / "data_splits"
    splits.mkdir()
    for name, n in datasets.items():
        manifest = _make_manifest(n_train=n)
        (splits / f"{name}.json").write_text(json.dumps(manifest))
    return splits


# ---------------------------------------------------------------------------
# ThunderDatasetRegistry tests
# ---------------------------------------------------------------------------

class TestThunderDatasetRegistry:
    def _make_registry(self, tmp_path, dataset_sizes: dict, **kwargs):
        _make_splits_dir(tmp_path, dataset_sizes)
        # Mock _count_train_samples to return known values without Thunder lib
        sizes = dataset_sizes

        with patch("src.data.thunder_multi._count_train_samples",
                   side_effect=lambda name, _: sizes[name]):
            from src.data.thunder_multi import ThunderDatasetRegistry
            return ThunderDatasetRegistry(str(tmp_path), **kwargs)

    def test_selects_n_smallest_as_holdout(self, tmp_path):
        sizes = {"small_a": 100, "small_b": 200, "small_c": 300, "big_d": 5000, "big_e": 8000}
        reg = self._make_registry(tmp_path, sizes, n_holdout=3)
        assert set(reg.holdout_datasets) == {"small_a", "small_b", "small_c"}
        assert set(reg.train_datasets) == {"big_d", "big_e"}

    def test_n_holdout_1(self, tmp_path):
        sizes = {"a": 10, "b": 20, "c": 30}
        reg = self._make_registry(tmp_path, sizes, n_holdout=1)
        assert reg.holdout_datasets == ["a"]
        assert set(reg.train_datasets) == {"b", "c"}

    def test_explicit_holdout_overrides_size_ranking(self, tmp_path):
        sizes = {"a": 10, "b": 20, "c": 30, "d": 40}
        reg = self._make_registry(tmp_path, sizes,
                                   n_holdout=1, holdout_datasets=["c", "d"])
        assert set(reg.holdout_datasets) == {"c", "d"}
        assert set(reg.train_datasets) == {"a", "b"}

    def test_explicit_holdout_unknown_name_raises(self, tmp_path):
        sizes = {"a": 10, "b": 20}
        with pytest.raises(ValueError, match="not in data_splits"):
            self._make_registry(tmp_path, sizes, holdout_datasets=["nonexistent"])

    def test_missing_splits_dir_raises(self, tmp_path):
        with patch("src.data.thunder_multi._count_train_samples", return_value=100):
            from src.data.thunder_multi import ThunderDatasetRegistry
            with pytest.raises(FileNotFoundError):
                ThunderDatasetRegistry(str(tmp_path))

    def test_empty_splits_dir_raises(self, tmp_path):
        (tmp_path / "data_splits").mkdir()
        with patch("src.data.thunder_multi._count_train_samples", return_value=100):
            from src.data.thunder_multi import ThunderDatasetRegistry
            with pytest.raises(ValueError, match="No JSON manifests"):
                ThunderDatasetRegistry(str(tmp_path))

    def test_save_plan_and_from_plan_idempotent(self, tmp_path):
        sizes = {"s": 10, "m": 500, "l": 10000, "xl": 50000}
        reg = self._make_registry(tmp_path, sizes, n_holdout=2)

        plan_path = tmp_path / "holdout_plan.json"
        reg.save_plan(str(plan_path))
        assert plan_path.exists()

        from src.data.thunder_multi import ThunderDatasetRegistry
        reg2 = ThunderDatasetRegistry.from_plan(str(plan_path), str(tmp_path))
        assert reg2.holdout_datasets == reg.holdout_datasets
        assert reg2.train_datasets == reg.train_datasets
        assert reg2.sample_counts == reg.sample_counts

    def test_save_plan_json_structure(self, tmp_path):
        sizes = {"a": 1, "b": 2, "c": 3}
        reg = self._make_registry(tmp_path, sizes, n_holdout=1)
        plan_path = tmp_path / "plan.json"
        reg.save_plan(str(plan_path))

        plan = json.loads(plan_path.read_text())
        assert "train_datasets" in plan
        assert "holdout_datasets" in plan
        assert "sample_counts" in plan
        assert set(plan["train_datasets"]) | set(plan["holdout_datasets"]) == set(sizes)


# ---------------------------------------------------------------------------
# TaggedTupleDataset tests
# ---------------------------------------------------------------------------

class TestTaggedTupleDataset:
    def _make_patch_ds(self, n: int = 10):
        class _FakePatchDS(Dataset):
            def __len__(self):
                return n

            def __getitem__(self, idx):
                return {"image": torch.zeros(3, 8, 8), "label": idx % 3}

        return _FakePatchDS()

    def test_length_matches_underlying(self):
        from src.data.thunder_multi import TaggedTupleDataset
        ds = TaggedTupleDataset(self._make_patch_ds(7), dataset_idx=0)
        assert len(ds) == 7

    def test_returns_triple(self):
        from src.data.thunder_multi import TaggedTupleDataset
        ds = TaggedTupleDataset(self._make_patch_ds(5), dataset_idx=2)
        img, label, idx = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)
        assert idx == 2

    def test_dataset_idx_preserved(self):
        from src.data.thunder_multi import TaggedTupleDataset
        for expected_idx in [0, 3, 7]:
            ds = TaggedTupleDataset(self._make_patch_ds(4), dataset_idx=expected_idx)
            _, _, got_idx = ds[0]
            assert got_idx == expected_idx


# ---------------------------------------------------------------------------
# Sampler weight balance tests
# ---------------------------------------------------------------------------

class TestSamplerWeights:
    """Verify the cross-dataset balanced weights without building real DataLoaders."""

    def _compute_weights(self, dataset_labels: list[list[int]]) -> np.ndarray:
        """Replicate the weight computation from build_multi_thunder_train_loaders."""
        flat_labels = np.concatenate([np.array(l) for l in dataset_labels])
        idx_arr = np.array(
            [i for i, labels in enumerate(dataset_labels) for _ in labels]
        )
        weights = np.zeros(len(flat_labels), dtype=np.float64)
        for d_idx, labels_d in enumerate(dataset_labels):
            labels_arr = np.array(labels_d)
            n_classes = int(labels_arr.max()) + 1
            counts = np.bincount(labels_arr, minlength=n_classes)
            safe = np.where(counts > 0, counts, 1)
            mask = idx_arr == d_idx
            weights[mask] = 1.0 / (n_classes * safe[labels_arr])
        return weights

    def test_total_weight_equal_per_dataset(self):
        # 3 datasets with different sizes, same n_classes — each should sum to 1
        labels_d0 = [0, 1] * 5      # 10 samples, 2 classes
        labels_d1 = [0, 1] * 500    # 1000 samples, 2 classes
        labels_d2 = [0, 1, 2] * 3   # 9 samples, 3 classes

        weights = self._compute_weights([labels_d0, labels_d1, labels_d2])
        idx_arr = np.array(
            [i for i, labels in enumerate([labels_d0, labels_d1, labels_d2]) for _ in labels]
        )
        totals = [weights[idx_arr == i].sum() for i in range(3)]
        for t in totals:
            assert abs(t - 1.0) < 1e-9, f"Dataset total weight should be 1.0, got {t}"

    def test_all_weights_positive(self):
        labels = [[0, 0, 1, 2], [0, 1]]
        weights = self._compute_weights(labels)
        assert (weights > 0).all()
