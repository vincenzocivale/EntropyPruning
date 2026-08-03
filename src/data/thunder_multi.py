"""Multi-dataset loader and holdout registry for Thunder benchmark datasets.

Provides:
    ThunderDatasetRegistry  — discovers datasets, ranks by size, designates holdout
    TaggedTupleDataset      — wraps PatchDataset, emits (image, label, dataset_idx)
    build_multi_thunder_train_loaders — combined DataLoaders across N datasets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import thunder
from thunder.utils.data import PatchDataset, get_data


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class TaggedTupleDataset(Dataset):
    """Wraps Thunder's PatchDataset; returns (image, label, dataset_idx) triples."""

    def __init__(self, patch_ds: PatchDataset, dataset_idx: int):
        self._ds = patch_ds
        self._dataset_idx = dataset_idx

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        item = self._ds[idx]
        label = np.asarray(item["label"]).reshape(()).item()
        return item["image"], int(label), self._dataset_idx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _thunder_cfg_path(dataset_name: str) -> Path:
    return Path(thunder.__file__).parent / "config" / "dataset" / f"{dataset_name}.yaml"


def _dataset_meta(dataset_name: str, base_data_folder: str, data: dict) -> dict:
    """Return class info and train labels for a single Thunder dataset."""
    cfg_path = _thunder_cfg_path(dataset_name)
    h5_format = False
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        class_names = list(cfg.classes)
        n_classes = int(cfg.nb_classes)
        h5_format = bool(getattr(cfg, "h5_format", False))
    else:
        raw = data["train"]["labels"]
        n_classes = int(max(raw)) + 1
        class_names = [f"class_{i}" for i in range(n_classes)]

    if h5_format:
        labels_path = Path(base_data_folder) / dataset_name / str(data["train"]["labels"])
        with h5py.File(labels_path, "r") as f:
            train_labels = np.array(f["y"]).reshape(-1).astype(int)
    else:
        train_labels = np.array(data["train"]["labels"]).flatten().astype(int)

    return {
        "n_classes": n_classes,
        "class_names": class_names,
        "h5_format": h5_format,
        "train_labels": train_labels,
    }


def _count_train_samples(dataset_name: str, base_data_folder: str) -> int:
    data = get_data(dataset_name, base_data_folder)
    cfg_path = _thunder_cfg_path(dataset_name)
    h5_format = False
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        h5_format = bool(getattr(cfg, "h5_format", False))
    if h5_format:
        labels_path = Path(base_data_folder) / dataset_name / str(data["train"]["labels"])
        with h5py.File(labels_path, "r") as f:
            return int(len(f["y"]))
    return len(data["train"]["images"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ThunderDatasetRegistry:
    """Discovers all Thunder datasets and designates the N smallest as holdout.

    Holdout is determined by training sample count (not file count), so HDF5
    datasets are handled correctly.

    Args:
        base_data_folder: Root directory containing data_splits/ and datasets/.
        n_holdout:        Number of datasets to hold out (default 3).
        holdout_datasets: Explicit list of dataset names to hold out.
                          When provided, overrides n_holdout.
    """

    def __init__(
        self,
        base_data_folder: str,
        n_holdout: int = 3,
        holdout_datasets: Optional[list] = None,
    ):
        self._base = Path(base_data_folder)
        splits_dir = self._base / "data_splits"
        if not splits_dir.exists():
            raise FileNotFoundError(f"data_splits/ not found: {splits_dir}")

        all_names = sorted(p.stem for p in splits_dir.glob("*.json"))
        if not all_names:
            raise ValueError(f"No JSON manifests found in {splits_dir}")

        if holdout_datasets is not None:
            missing = set(holdout_datasets) - set(all_names)
            if missing:
                raise ValueError(f"Holdout names not in data_splits/: {sorted(missing)}")
            self._holdout: list = list(holdout_datasets)
            self._train: list = [d for d in all_names if d not in set(self._holdout)]
            self._sample_counts: dict = {
                d: _count_train_samples(d, str(self._base)) for d in all_names
            }
        else:
            self._sample_counts = {
                d: _count_train_samples(d, str(self._base)) for d in all_names
            }
            ranked = sorted(all_names, key=lambda d: self._sample_counts[d])
            self._holdout = ranked[:n_holdout]
            self._train = ranked[n_holdout:]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_datasets(self) -> list:
        """Dataset names used for training (holdout excluded), sorted by sample count."""
        return list(self._train)

    @property
    def holdout_datasets(self) -> list:
        """Dataset names held out from training (smallest N or explicit)."""
        return list(self._holdout)

    @property
    def sample_counts(self) -> dict:
        """Mapping dataset_name → train sample count."""
        return dict(self._sample_counts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_plan(self, path: str) -> None:
        """Write holdout_plan.json for reproducible reuse in Phase 2 and 3."""
        plan = {
            "train_datasets": self._train,
            "holdout_datasets": self._holdout,
            "sample_counts": {k: int(v) for k, v in self._sample_counts.items()},
        }
        Path(path).write_text(json.dumps(plan, indent=2))
        print(f"Holdout plan → {path}")
        print(f"  train  ({len(self._train)}): {self._train}")
        print(f"  holdout ({len(self._holdout)}): {self._holdout}")

    @classmethod
    def from_plan(cls, plan_path: str, base_data_folder: str) -> "ThunderDatasetRegistry":
        """Reconstruct registry from a saved holdout_plan.json (Phase 2 / 3)."""
        plan = json.loads(Path(plan_path).read_text())
        inst = cls.__new__(cls)
        inst._base = Path(base_data_folder)
        inst._train = plan["train_datasets"]
        inst._holdout = plan["holdout_datasets"]
        inst._sample_counts = plan.get("sample_counts", {})
        return inst


# ---------------------------------------------------------------------------
# Multi-dataset DataLoader builder
# ---------------------------------------------------------------------------

def build_multi_thunder_train_loaders(
    dataset_names: list,
    base_data_folder: str,
    transform,
    batch_size: int = 8,
    num_workers: int = 4,
    drop_last_train: bool = True,
) -> tuple:
    """Build train/val DataLoaders combining multiple Thunder datasets.

    Each loader yields ``(image, label, dataset_idx)`` triples where
    ``dataset_idx`` is the position of the dataset in ``dataset_names``.

    Sampling balances across datasets *and* classes:
      - Each dataset contributes equal total probability.
      - Within each dataset, each class contributes equal probability.

    Args:
        dataset_names:    Ordered list of dataset names; their index is dataset_idx.
        base_data_folder: Root containing data_splits/ and datasets/.
        transform:        Model-specific image transform from get_model_from_name.
        batch_size:       Batch size.
        num_workers:      DataLoader workers.
        drop_last_train:  Drop incomplete last batch during training.

    Returns:
        (train_loader, val_loader, dataset_info) where
        dataset_info = {dataset_idx (int): {name, n_classes, class_names}}.
    """
    train_dsets: list = []
    val_dsets: list = []
    dataset_info: dict = {}
    all_labels: list = []
    all_idx: list = []

    for dataset_idx, name in enumerate(dataset_names):
        data = get_data(name, base_data_folder)
        meta = _dataset_meta(name, base_data_folder, data)

        dataset_info[dataset_idx] = {
            "name": name,
            "n_classes": meta["n_classes"],
            "class_names": meta["class_names"],
        }

        def _make(split, _name=name, _h5=meta["h5_format"], _data=data, _idx=dataset_idx):
            return TaggedTupleDataset(
                PatchDataset(
                    images=_data[split]["images"],
                    labels=_data[split]["labels"],
                    transform=transform,
                    task_type="linear_probing",
                    dataset_name=_name,
                    base_data_folder=base_data_folder,
                    embeddings_folder=None,
                    image_pre_loading=False,
                    embedding_pre_loading=False,
                    div_patches=False,
                    h5_format=_h5,
                ),
                _idx,
            )

        train_dsets.append(_make("train"))
        val_dsets.append(_make("val"))
        all_labels.append(meta["train_labels"])
        all_idx.extend([dataset_idx] * len(meta["train_labels"]))

    # weight_i = 1 / (n_classes_d * count_of_class_d(i))
    # → total weight per dataset = 1 (equal contribution regardless of dataset size)
    flat_labels = np.concatenate(all_labels)
    idx_arr = np.array(all_idx, dtype=np.int64)
    weights = np.zeros(len(flat_labels), dtype=np.float64)

    for dataset_idx, name in enumerate(dataset_names):
        mask = idx_arr == dataset_idx
        labels_d = flat_labels[mask]
        n_classes = dataset_info[dataset_idx]["n_classes"]
        counts = np.bincount(labels_d, minlength=n_classes)
        safe_counts = np.where(counts > 0, counts, 1)
        weights[mask] = 1.0 / (n_classes * safe_counts[labels_d].astype(np.float64))

    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True
    )

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return (
        DataLoader(ConcatDataset(train_dsets), sampler=sampler,
                   drop_last=drop_last_train, **kw),
        DataLoader(ConcatDataset(val_dsets), shuffle=False, **kw),
        dataset_info,
    )
