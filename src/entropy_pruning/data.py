from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class HistologicalImageDataset(Dataset):
    """Simple ImageFolder-like dataset with deterministic class indexing."""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root}")

        self.class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            for img_path in sorted(class_dir.rglob("*")):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No images found in: {self.root}")

        self.labels = np.array([lbl for _, lbl in self.samples], dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = T.ToTensor()(img)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label


class ArrowHistologyDataset(Dataset):
    """Dataset wrapper for Hugging Face Arrow shards saved with `save_to_disk`."""

    def __init__(
        self,
        root: str | Path,
        transform=None,
        image_column: str = "image",
        label_column: str = "label",
        class_names: list[str] | None = None,
    ):
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "Hugging Face `datasets` is required for Arrow dataset support. "
                "Install it with `pip install datasets`."
            ) from exc

        self.root = Path(root)
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root}")

        self.ds = load_from_disk(str(self.root))
        if len(self.ds) == 0:
            raise RuntimeError(f"Empty Arrow dataset: {self.root}")

        if self.image_column not in self.ds.column_names:
            raise RuntimeError(
                f"Image column `{self.image_column}` not found in {self.root}. "
                f"Available columns: {self.ds.column_names}"
            )
        if self.label_column not in self.ds.column_names:
            raise RuntimeError(
                f"Label column `{self.label_column}` not found in {self.root}. "
                f"Available columns: {self.ds.column_names}"
            )

        self.labels = np.array(self.ds[self.label_column], dtype=np.int64)
        self.class_names = class_names or self._extract_class_names()

    def _extract_class_names(self) -> list[str]:
        feat = self.ds.features.get(self.label_column, None)
        names = getattr(feat, "names", None)
        if names:
            return [str(x) for x in names]

        n_classes = int(self.labels.max()) + 1 if len(self.labels) else 0
        return [str(i) for i in range(n_classes)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        img = row[self.image_column]

        # datasets Image feature usually yields a PIL image, but support dict payload too
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"] is not None:
                from io import BytesIO

                img = Image.open(BytesIO(img["bytes"]))
            elif "path" in img and img["path"] is not None:
                img = Image.open(img["path"])
            else:
                raise RuntimeError("Unsupported image dict format in Arrow dataset.")

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if not isinstance(img, Image.Image):
            raise RuntimeError(f"Unsupported image type: {type(img)}")

        img = img.convert("RGB")
        tensor = T.ToTensor()(img)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, int(row[self.label_column])


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    n_classes: int


def make_transforms(img_size: int = 224):
    train_tf = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.1, 0.05)], p=0.5),
            T.Resize((img_size, img_size)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, eval_tf


def build_loaders(
    data_dir: str | Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    drop_last_train: bool = True,
    image_column: str = "image",
    label_column: str = "label",
) -> LoaderBundle:
    train_tf, eval_tf = make_transforms(img_size)
    root = Path(data_dir)

    def _is_arrow_split(split_path: Path) -> bool:
        return (split_path / "state.json").exists() and any(split_path.glob("*.arrow"))

    def _resolve_split(preferred: str, fallback: str | None = None) -> Path:
        primary = root / preferred
        if primary.exists():
            return primary
        if fallback is not None:
            alt = root / fallback
            if alt.exists():
                return alt
        raise FileNotFoundError(
            f"Split path not found. Tried: {primary}" + (f", {root / fallback}" if fallback else "")
        )

    train_path = _resolve_split("train")
    val_path = _resolve_split("val", fallback="validation")
    test_path = _resolve_split("test")

    if _is_arrow_split(train_path):
        train_ds = ArrowHistologyDataset(
            train_path,
            transform=train_tf,
            image_column=image_column,
            label_column=label_column,
        )
        class_names = train_ds.class_names
        val_ds = ArrowHistologyDataset(
            val_path,
            transform=eval_tf,
            image_column=image_column,
            label_column=label_column,
            class_names=class_names,
        )
        test_ds = ArrowHistologyDataset(
            test_path,
            transform=eval_tf,
            image_column=image_column,
            label_column=label_column,
            class_names=class_names,
        )
    else:
        train_ds = HistologicalImageDataset(train_path, transform=train_tf)
        class_names = train_ds.class_names
        val_ds = HistologicalImageDataset(val_path, transform=eval_tf)
        test_ds = HistologicalImageDataset(test_path, transform=eval_tf)

    counts = np.bincount(train_ds.labels)
    weights = torch.from_numpy((1.0 / counts)[train_ds.labels]).double()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, sampler=sampler, drop_last=drop_last_train, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        n_classes=len(class_names),
    )
