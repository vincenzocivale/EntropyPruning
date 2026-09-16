"""Thin THUNDER benchmark loader used only for downstream evaluation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import thunder
from thunder.utils.data import PatchDataset, get_data


class _TupleDataset(Dataset):
    def __init__(self, patch_ds):
        self._ds = patch_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        item = self._ds[idx]
        label = np.asarray(item["label"]).reshape(()).item()
        return item["image"], int(label)


def discover_thunder_datasets(base_data_folder: str | Path) -> list[str]:
    splits = Path(base_data_folder) / "data_splits"
    if not splits.is_dir():
        raise FileNotFoundError(f"THUNDER data_splits/ not found: {splits}")
    return sorted(path.stem for path in splits.glob("*.json"))


def build_thunder_loaders(
    dataset_name: str,
    base_data_folder: str,
    transform,
    batch_size: int = 32,
    num_workers: int = 4,
    *,
    balanced_train: bool = False,
    drop_last_train: bool = False,
):
    """Return deterministic train/val/test loaders for one THUNDER dataset.

    ``balanced_train=False`` is the evaluation default: every original training
    example is embedded exactly once. Class balancing belongs in the downstream
    linear head, not in feature extraction.
    """
    split_path = Path(base_data_folder) / "data_splits" / f"{dataset_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Data split not found: {split_path}")

    data = get_data(dataset_name, base_data_folder)
    cfg_path = Path(thunder.__file__).parent / "config" / "dataset" / f"{dataset_name}.yaml"
    h5_format = False
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        class_names = list(cfg.classes)
        n_classes = int(cfg.nb_classes)
        h5_format = bool(getattr(cfg, "h5_format", False))
    else:
        labels = np.asarray(data["train"]["labels"]).reshape(-1).astype(int)
        n_classes = int(labels.max()) + 1
        class_names = [f"class_{i}" for i in range(n_classes)]

    def make(split: str) -> _TupleDataset:
        return _TupleDataset(
            PatchDataset(
                images=data[split]["images"],
                labels=data[split]["labels"],
                transform=transform,
                task_type="linear_probing",
                dataset_name=dataset_name,
                base_data_folder=base_data_folder,
                embeddings_folder=None,
                image_pre_loading=False,
                embedding_pre_loading=False,
                div_patches=False,
                h5_format=h5_format,
            )
        )

    train_ds, val_ds, test_ds = make("train"), make("val"), make("test")
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    if balanced_train:
        if h5_format:
            labels_path = Path(base_data_folder) / dataset_name / str(data["train"]["labels"])
            with h5py.File(labels_path, "r") as handle:
                train_labels = np.asarray(handle["y"]).reshape(-1).astype(int)
        else:
            train_labels = np.asarray(data["train"]["labels"]).reshape(-1).astype(int)
        counts = np.bincount(train_labels)
        weights = torch.from_numpy((1.0 / counts)[train_labels]).double()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, drop_last=drop_last_train, **kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=False, drop_last=False, **kwargs)

    return (
        train_loader,
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
        class_names,
        n_classes,
    )
