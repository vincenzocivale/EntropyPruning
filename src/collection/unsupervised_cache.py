"""Build per-dataset attention-imitation caches from a frozen, pretrained
foundation model (no Phase-1 fine-tuning / classifier checkpoint).

Used by the unsupervised multi-dataset EAF pipeline: a single
``AttentionForecaster`` is trained on the merged corpus to imitate the
frozen FM's own last-layer CLS->patch attention row.
"""

from pathlib import Path

import h5py
from torch.utils.data import DataLoader, Subset

from src.data.thunder_loaders import build_thunder_loaders
from src.models import ThunderBackboneAdapter, build_classifier

from .extract_features import collect_and_save_dataset


def build_frozen_model(model_name, raw_backbone, device):
    """Wrap a raw FM backbone as a frozen ``LinearProbingClassifier``.

    No Phase-1 checkpoint is loaded, so the backbone is the unmodified
    pretrained foundation model. The classification head is randomly
    initialized and unused (cache building only reads hook-captured
    embeddings/attention, never the logits).
    """
    adapter = ThunderBackboneAdapter(raw_backbone)
    model = build_classifier("linear_probing", raw_backbone, adapter, n_classes=2).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, adapter


def cache_is_valid(cache_path, layers_source, layer_target, n_patches, embed_dim):
    """Check whether an existing HDF5 cache has the expected datasets/shapes
    for all three splits, so a multi-dataset sweep can resume safely."""
    if isinstance(layers_source, int):
        layers_source = [layers_source]
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    try:
        with h5py.File(cache_path, "r") as f:
            for split in ("train", "val", "test"):
                if split not in f:
                    return False
                grp = f[split]
                attn_key = f"attn_layer{layer_target}"
                if "labels" not in grp or attn_key not in grp:
                    return False
                if grp[attn_key].shape[1:] != (n_patches,):
                    return False
                for ls in layers_source:
                    emb_key = f"emb_layer{ls}"
                    if emb_key not in grp:
                        return False
                    if grp[emb_key].shape[1:] != (n_patches, embed_dim):
                        return False
    except OSError:
        return False
    return True


def _subset_loader(loader, max_samples):
    n = min(max_samples, len(loader.dataset))
    return DataLoader(
        Subset(loader.dataset, range(n)),
        batch_size=loader.batch_size, num_workers=loader.num_workers,
        shuffle=False, pin_memory=True,
    )


def build_attention_cache(
    model, adapter, transform, dataset_name, base_data_folder, save_path, device,
    layers_source=None, layer_target=None,
    batch_size=64, num_workers=4, max_samples_per_split=None,
):
    """Extract ``emb_layer{L}`` for each L in ``layers_source`` and
    ``attn_layer{layer_target}`` for one dataset using a shared frozen FM,
    and save them to ``save_path``.

    ``layers_source`` is a list of block indices (e.g. ``[1, 2, 3, 4, 5]``);
    a single int is also accepted for backward compatibility.
    Skips extraction (returns immediately) if a valid cache already exists.
    """
    if layers_source is None:
        layers_source = [2]
    if isinstance(layers_source, int):
        layers_source = [layers_source]
    if layer_target is None:
        layer_target = adapter.n_blocks - 1
    for ls in layers_source:
        assert ls < adapter.n_blocks, f"layers_source value {ls} >= n_blocks {adapter.n_blocks}"
    assert layer_target < adapter.n_blocks, \
        f"layer_target {layer_target} >= n_blocks {adapter.n_blocks}"

    save_path = Path(save_path)
    if cache_is_valid(save_path, layers_source, layer_target, adapter.n_patches, adapter.embed_dim):
        print(f"  Cache valid, skipping: {save_path}")
        return save_path

    train_loader, val_loader, test_loader, _, _ = build_thunder_loaders(
        dataset_name, base_data_folder, transform, batch_size, num_workers, drop_last_train=False,
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if max_samples_per_split is not None:
        loaders = {name: _subset_loader(loader, max_samples_per_split) for name, loader in loaders.items()}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    collect_and_save_dataset(
        model, loaders, device,
        layers_source=layers_source,
        layers_target=[layer_target],
        save_path=save_path,
    )
    return save_path
