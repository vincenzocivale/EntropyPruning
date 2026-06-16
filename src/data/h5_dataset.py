import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

# Disable HDF5 file locking to avoid [Errno 11] on some filesystems
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class H5ForecastDataset(Dataset):
    """Lazy-loading dataset from HDF5 for forecaster training.

    Loads patch embeddings from one or more source layers and CLS attention
    from ``layer_target``.  When multiple source layers are given the
    per-layer embeddings are concatenated along the feature dimension, so the
    returned ``emb`` has shape ``(n_patches, len(layers_source) * embed_dim)``.

    ``layers_source`` accepts either a single ``int`` (backward-compatible) or
    a ``list[int]``.
    """
    def __init__(self, h5_path, split, layers_source, layer_target):
        self.h5_path = str(h5_path)
        self.split = split
        self.layers_source = [layers_source] if isinstance(layers_source, int) else list(layers_source)
        self.layer_target = layer_target
        self._file = None

        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[split]["labels"])

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()
        grp = f[self.split]
        embs = [torch.from_numpy(grp[f"emb_layer{ls}"][idx]).float() for ls in self.layers_source]
        emb = torch.cat(embs, dim=-1)
        target = torch.from_numpy(grp[f"attn_layer{self.layer_target}"][idx]).float()
        label = int(grp["labels"][idx])
        return emb, target, label


class MultiH5ForecastDataset(Dataset):
    """Concatenation of per-dataset ``H5ForecastDataset`` caches.

    Merges multiple HDF5 attention caches (one per source dataset) into a
    single dataset for training one universal AttentionForecaster, while
    keeping track of which sub-dataset each sample came from so per-dataset
    metrics can be reported.

    Args:
        cache_paths: ``{dataset_name: h5_path}`` mapping.
        split:       "train", "val", or "test".
        layer_source, layer_target: forwarded to each ``H5ForecastDataset``.

    ``__getitem__`` returns ``(emb, target, label, dataset_idx)``, where
    ``dataset_idx`` indexes into ``self.dataset_names``.
    """

    def __init__(self, cache_paths, split, layer_source, layer_target):
        self.dataset_names = list(cache_paths.keys())
        self.datasets = [
            H5ForecastDataset(path, split, layer_source, layer_target)
            for path in cache_paths.values()
        ]
        lengths = [len(d) for d in self.datasets]
        self._offsets = np.cumsum([0] + lengths)

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, idx):
        ds_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local_idx = idx - int(self._offsets[ds_idx])
        emb, target, label = self.datasets[ds_idx][local_idx]
        return emb, target, label, ds_idx
