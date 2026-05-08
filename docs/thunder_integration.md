# Thunder Integration

EAF uses [THUNDER](https://github.com/MICS-Lab/thunder) as the backend for two things:
1. **Foundation model loading** — any of the 23+ supported models via a unified API
2. **Benchmark dataset access** — 16+ pathology datasets with standardised splits

EAF does not modify Thunder. All integration is implemented in `src/models/backbone_adapter.py` and `src/data/thunder_loaders.py`.

---

## Model loading

### How it works

```python
from thunder.models.pretrained_models import get_model_from_name

raw_backbone, transform, _ = get_model_from_name("uni", "cuda")
```

Returns:
- `raw_backbone` — a `torch.nn.Module` (timm `VisionTransformer` for supported models)
- `transform` — model-specific preprocessing pipeline (resize + normalise)
- `_` — Thunder's embedding extraction function (not used by EAF)

### `ThunderBackboneAdapter`

EAF wraps the raw backbone with `ThunderBackboneAdapter` to extract architecture metadata:

```python
from src.models import ThunderBackboneAdapter

adapter = ThunderBackboneAdapter(raw_backbone)
# adapter.embed_dim         → 1024   (CLS token dimension)
# adapter.n_blocks          → 24     (number of transformer blocks)
# adapter.n_patches         → 196    (spatial patch count, excl. prefix tokens)
# adapter.num_prefix_tokens → 1      (CLS + register tokens)
```

All values are read directly from timm model attributes — nothing is hardcoded.

**Supported model types:** timm-based `VisionTransformer` (UNI, H-optimus, Virchow, kaiko, DINOv2, ...).
HuggingFace-based models (Phikon, Hibou) raise `NotImplementedError`.

### Why `embed_dim` from the adapter, not from Thunder's config

Thunder's `emb_dim` in YAML configs sometimes reflects a concatenated embedding (e.g. Virchow: CLS + avg-pooled patches = 2560). EAF always uses the raw CLS token dimension (`model.embed_dim = 1280` for Virchow), which is what the backbone outputs by default.

---

## Dataset loading

### Thunder's data format

Thunder stores datasets as JSON split files + image directories:

```
{base_data_folder}/
├── data_splits/
│   └── crc.json        ← {"train": {"images": [...], "labels": [...]}, "val": ..., "test": ...}
└── datasets/
    └── crc/            ← image files
```

`get_data(dataset_name, base_data_folder)` reads the JSON.
`PatchDataset` loads images on demand and applies the model transform.

### `build_thunder_loaders`

EAF's bridge function wraps Thunder's `PatchDataset` to match the `(imgs, labels)` tuple format expected by EAF training loops:

```python
from src.data.thunder_loaders import build_thunder_loaders

train_loader, val_loader, test_loader, class_names, n_classes = build_thunder_loaders(
    dataset_name="crc",
    base_data_folder="/path/to/thunder/data",
    transform=transform,          # from get_model_from_name
    batch_size=8,
    num_workers=4,
)
```

**`_TupleDataset` wrapper** — Thunder's `PatchDataset.__getitem__` returns `{'image': tensor, 'label': int}`. The wrapper converts this to `(tensor, long_tensor)` without copying data.

**Class balancing** — the training loader always uses `WeightedRandomSampler` with inverse class frequency weights, matching EAF's original behaviour.

**Class names** — read from Thunder's dataset YAML config (`thunder/src/thunder/config/dataset/{name}.yaml`). Falls back to `["class_0", "class_1", ...]` if the config is not found.

---

## Transform handling

Each Foundation model has its own preprocessing pipeline (different image sizes, normalisation statistics). Thunder's `get_model_from_name` returns the correct transform, which is passed directly to `build_thunder_loaders`.

The transform is **eval-style** (resize + centre crop + normalise). Unlike the original EAF pipeline, no random augmentation is applied. To add augmentations for training:

```python
import torchvision.transforms as T

# Extract normalisation from Thunder's transform
normalize = transform.transforms[-1]  # assumed to be the last step

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    *transform.transforms,   # Thunder's resize + normalize
])
```

Pass `train_transform` to `build_thunder_loaders` for the training split.

---

## Supported datasets

All Thunder benchmark datasets (except `bracs`) are supported.
`bracs` requires `div_patches=True` (WSI-level patching with variable patch count) and raises `ValueError` if passed to `build_thunder_loaders`.

| Dataset | Task | Classes |
|---|---|---|
| `crc` | Colorectal cancer tissue typing | 9 |
| `break_his` | Breast tumour malignancy | 2/8 |
| `mhist` | Colorectal polyp classification | 2 |
| `patch_camelyon` | Lymph node metastasis | 2 |
| `bach` | Breast cancer grading | 4 |
| `wilds` | Tumour detection (domain shift) | 2 |
| `spider_breast`, `spider_skin`, ... | SPIDER tissue typing | varies |
| ... | (see Thunder docs for full list) | |

---

## What EAF does NOT use from Thunder

- Thunder's task system (`linear_probing`, `knn_classification`, etc.)
- Thunder's Hydra config system
- Thunder's LoRA implementation (`adapters.py`)
- Thunder's `benchmark()` function
- Thunder's `PretrainedModel` abstract class

EAF uses Thunder purely as a **model zoo** and **data loader**, implementing its own training pipeline on top.
