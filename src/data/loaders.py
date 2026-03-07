import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import HistologicalImageDataset
from .transforms import get_train_transform, get_eval_transform


def build_loaders(data_dir, img_size=224, batch_size=8, num_workers=4,
                  drop_last_train=True):
    """Build train/val/test DataLoaders with balanced sampling.

    Returns:
        (train_loader, val_loader, test_loader, class_names, n_classes)
    """
    train_tf = get_train_transform(img_size)
    eval_tf = get_eval_transform(img_size)

    train_ds = HistologicalImageDataset(f"{data_dir}/train", transform=train_tf)
    val_ds = HistologicalImageDataset(f"{data_dir}/val", transform=eval_tf)
    test_ds = HistologicalImageDataset(f"{data_dir}/test", transform=eval_tf)

    counts = np.bincount(train_ds.labels)
    weights = torch.from_numpy((1.0 / counts)[train_ds.labels]).double()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, sampler=sampler,
                              drop_last=drop_last_train, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    class_names = train_ds.class_names
    n_classes = len(class_names)

    return train_loader, val_loader, test_loader, class_names, n_classes
