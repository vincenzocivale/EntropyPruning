from torch.utils.data import Dataset
from datasets import load_from_disk
import numpy as np
import torch

class HistologicalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        print(f"Loading from {data_dir}...")
        self.hf_dataset = load_from_disk(data_dir)

        label_feature = self.hf_dataset.features.get('label')
        if hasattr(label_feature, 'names'):
            self.class_names = list(label_feature.names)
        else:
            unique_labels = sorted(set(self.hf_dataset['label']))
            self.class_names = unique_labels if isinstance(unique_labels[0], str) \
                               else [f"Class_{i}" for i in unique_labels]

        self.labels = np.array(self.hf_dataset['label'], dtype=np.int64)
        print(f"Loaded {len(self)} samples, {len(self.class_names)} classes")
        self._print_class_distribution()

    def _print_class_distribution(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Class distribution:")
        for cls_idx, count in zip(unique, counts):
            print(f"  {self.class_names[cls_idx]}: {count} ({count/len(self)*100:.1f}%)")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = np.array(self.hf_dataset[idx]['image'])
        img   = torch.from_numpy(image).float()

        # HWC -> CHW
        if img.ndim == 3 and img.shape[2] in (3, 4):
            img = img.permute(2, 0, 1)

        # Rimuovi canale alpha se presente
        if img.shape[0] == 4:
            img = img[:3]

        # Normalizza in base al dtype originale
        if image.dtype == np.uint8:
            img = img / 255.0

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.long)