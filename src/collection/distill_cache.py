"""Build per-dataset feature caches for Phase 3 (Approach 3) distillation.

``DistilledPrunedBackbone`` only trains the LoRA blocks strictly after
``prune_layer`` (see ``post_prune_lora_targets``); the blocks up to and
including ``prune_layer`` are frozen and identical between the teacher and
the student, and the teacher itself never changes. For a fixed input image
their output is therefore constant for the entire training run, yet
``distill_pruned.py`` used to recompute it from raw pixels on every batch of
every epoch -- once for the teacher's full forward, once more for the
student's frozen prefix.

This module extracts, from a single frozen pretrained foundation model and
in one forward pass per image:

- ``seq_prune``: the raw output of ``blocks[prune_layer]`` (prefix + all
  patch tokens, pre-pruning) -- exactly the tensor the pruning hook in
  ``DistilledPrunedBackbone.forward`` receives, and the input
  ``forward_from_seq`` resumes from.
- ``teacher_cls`` / ``teacher_patches``: the model's own final CLS token and
  patch tokens (after the full, unpruned forward) -- the distillation
  targets.

Saved once per dataset to HDF5 (fp16), mirroring ``unsupervised_cache.py``'s
layout/conventions. Training then reads only these cached tensors: no image
loading and no frozen-backbone forward pass at training time.
"""

from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.data.thunder_loaders import build_thunder_loaders


def distill_cache_is_valid(cache_path, n_patches, embed_dim, num_prefix):
    """Check whether an existing HDF5 cache has the expected datasets/shapes
    for all three splits, so a multi-dataset build can resume safely."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    try:
        with h5py.File(cache_path, "r") as f:
            for split in ("train", "val", "test"):
                if split not in f:
                    return False
                grp = f[split]
                required = ("labels", "seq_prune", "teacher_cls", "teacher_patches")
                if any(key not in grp for key in required):
                    return False
                if grp["seq_prune"].shape[1:] != (num_prefix + n_patches, embed_dim):
                    return False
                if grp["teacher_cls"].shape[1:] != (embed_dim,):
                    return False
                if grp["teacher_patches"].shape[1:] != (n_patches, embed_dim):
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


@torch.no_grad()
def build_distill_cache(
    teacher, adapter, transform, dataset_name, base_data_folder, save_path, device,
    prune_layer, batch_size=64, num_workers=4, max_samples_per_split=None,
):
    """Extract ``seq_prune``/``teacher_cls``/``teacher_patches`` for one
    dataset from a frozen ``teacher`` backbone and save them to
    ``save_path``. Skips extraction if a valid cache already exists.

    Args:
        teacher:    raw timm backbone (frozen, eval mode) from
                    ``get_model_from_name``.
        adapter:    ``ThunderBackboneAdapter`` wrapping ``teacher``.
        prune_layer: block index whose raw output is cached (0-indexed).
    """
    num_prefix = adapter.num_prefix_tokens
    n_patches = adapter.n_patches
    embed_dim = adapter.embed_dim
    assert prune_layer < adapter.n_blocks - 1, \
        f"prune_layer {prune_layer} leaves no blocks to distill (n_blocks={adapter.n_blocks})"

    save_path = Path(save_path)
    if distill_cache_is_valid(save_path, n_patches, embed_dim, num_prefix):
        print(f"  Cache valid, skipping: {save_path}")
        return save_path

    train_loader, val_loader, test_loader, _, _ = build_thunder_loaders(
        dataset_name, base_data_folder, transform, batch_size, num_workers, drop_last_train=False,
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if max_samples_per_split is not None:
        loaders = {name: _subset_loader(loader, max_samples_per_split) for name, loader in loaders.items()}

    teacher.eval()
    captured = {}

    def _capture(module, input, output):
        captured["seq_prune"] = output

    handle = teacher.blocks[prune_layer].register_forward_hook(_capture)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(save_path, "w") as f:
            for split_name, loader in loaders.items():
                print(f"\nCollecting {split_name}...")
                n_total = len(loader.dataset)
                grp = f.create_group(split_name)
                chunk_n = max(1, min(64, n_total))
                ds_label = grp.create_dataset("labels", shape=(n_total,), dtype="i4")
                ds_seq = grp.create_dataset(
                    "seq_prune", shape=(n_total, num_prefix + n_patches, embed_dim),
                    dtype="f2", chunks=(chunk_n, num_prefix + n_patches, embed_dim))
                ds_cls = grp.create_dataset(
                    "teacher_cls", shape=(n_total, embed_dim),
                    dtype="f2", chunks=(chunk_n, embed_dim))
                ds_patches = grp.create_dataset(
                    "teacher_patches", shape=(n_total, n_patches, embed_dim),
                    dtype="f2", chunks=(chunk_n, n_patches, embed_dim))

                ptr = 0
                for imgs, labels in tqdm(loader, desc=split_name):
                    captured.clear()
                    features = teacher.forward_features(imgs.to(device))
                    seq_prune = captured["seq_prune"]
                    B = seq_prune.shape[0]
                    ds_seq[ptr:ptr + B] = seq_prune.half().cpu().numpy()
                    ds_cls[ptr:ptr + B] = features[:, 0].half().cpu().numpy()
                    ds_patches[ptr:ptr + B] = features[:, num_prefix:].half().cpu().numpy()
                    ds_label[ptr:ptr + B] = labels.numpy()
                    ptr += B
                print(f"  {split_name}: {ptr} samples saved")
    finally:
        handle.remove()

    return save_path
