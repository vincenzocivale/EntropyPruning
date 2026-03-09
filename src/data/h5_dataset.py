import os
import torch
from torch.utils.data import Dataset
import h5py

# Disable HDF5 file locking to avoid [Errno 11] on some filesystems
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class H5ForecastDataset(Dataset):
    """Lazy-loading dataset from HDF5 for forecaster training.
    Loads embeddings @ layer_source and attention @ layer_target.
    """
    def __init__(self, h5_path, split, layer_source, layer_target):
        self.h5_path = str(h5_path)
        self.split = split
        self.layer_source = layer_source
        self.layer_target = layer_target
        self._file = None

        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[split]["labels"])

    def _get_file(self):
        if self._file is None:
            # Re-open in each worker process
            self._file = h5py.File(self.h5_path, 'r')
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()
        grp = f[self.split]
        emb = torch.from_numpy(grp[f"emb_layer{self.layer_source}"][idx]).float()
        target = torch.from_numpy(grp[f"attn_layer{self.layer_target}"][idx]).float()
        label = int(grp["labels"][idx])
        return emb, target, label
