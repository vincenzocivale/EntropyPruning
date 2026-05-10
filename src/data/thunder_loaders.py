from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import thunder
from thunder.utils.data import PatchDataset, get_data


class _TupleDataset(Dataset):
    """
    Wraps Thunder's PatchDataset to return (image, label) tuples.
    EAF training loops expect (imgs, labels) from DataLoader, not dicts.
    """
    def __init__(self, patch_ds):
        self._ds = patch_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        item = self._ds[idx]
        label = np.asarray(item["label"]).reshape(()).item()
        return item["image"], torch.tensor(label, dtype=torch.long)


def build_thunder_loaders(
    dataset_name: str,
    base_data_folder: str,
    transform,
    batch_size: int = 8,
    num_workers: int = 4,
    drop_last_train: bool = True,
):
    """
    Build train/val/test DataLoaders from a Thunder benchmark dataset.

    Returns the same tuple as the legacy build_loaders():
        (train_loader, val_loader, test_loader, class_names, n_classes)

    Args:
        dataset_name:     Thunder dataset name (e.g. 'crc', 'break_his', 'mhist').
        base_data_folder: Root path containing data_splits/{name}.json and datasets/.
        transform:        Model-specific image transform from get_model_from_name.
        batch_size:       Batch size for all loaders.
        num_workers:      DataLoader worker count.
        drop_last_train:  Drop last incomplete training batch.

    Notes:
        - Requires {base_data_folder}/data_splits/{dataset_name}.json.
          Generate with: thunder download {dataset_name}
        - Training loader uses WeightedRandomSampler for class balance.
    """
    use_div_patches = False

    split_path = Path(base_data_folder) / "data_splits" / f"{dataset_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Data split not found: {split_path}\n"
            f"Generate it with: thunder download {dataset_name}"
        )

    data = get_data(dataset_name, base_data_folder)

    # Class names from Thunder's dataset YAML config
    dataset_cfg_path = (
        Path(thunder.__file__).parent / "config" / "dataset" / f"{dataset_name}.yaml"
    )
    h5_format = False
    if dataset_cfg_path.exists():
        cfg = OmegaConf.load(dataset_cfg_path)
        class_names = list(cfg.classes)
        n_classes = int(cfg.nb_classes)
        h5_format = bool(getattr(cfg, "h5_format", False))
    else:
        all_labels = data["train"]["labels"]
        n_classes = max(all_labels) + 1
        class_names = [f"class_{i}" for i in range(n_classes)]

    def _make_ds(split: str) -> _TupleDataset:
        return _TupleDataset(PatchDataset(
            images=data[split]["images"],
            labels=data[split]["labels"],
            transform=transform,
            task_type="linear_probing",
            dataset_name=dataset_name,
            base_data_folder=base_data_folder,
            embeddings_folder=None,
            image_pre_loading=False,
            embedding_pre_loading=False,
            div_patches=use_div_patches,
            h5_format=h5_format,
        ))

    train_ds = _make_ds("train")
    val_ds   = _make_ds("val")
    test_ds  = _make_ds("test")

    if h5_format:
        labels_path = Path(base_data_folder) / dataset_name / data["train"]["labels"]
        with h5py.File(labels_path, "r") as f:
            train_labels = np.array(f["y"]).reshape(-1).astype(int)
    else:
        train_labels = np.array(data["train"]["labels"]).flatten().astype(int)
    counts = np.bincount(train_labels)
    sample_weights = torch.from_numpy((1.0 / counts)[train_labels]).double()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return (
        DataLoader(train_ds, sampler=sampler, drop_last=drop_last_train, **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        class_names,
        n_classes,
    )
