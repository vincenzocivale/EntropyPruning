import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
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

    def read_block(self, start, end):
        """Read rows ``[start, end)`` via one contiguous HDF5 slice per dataset.

        Used by ``BlockShuffleH5Dataset`` to turn ``end - start`` random
        single-row seeks into a single sequential read -- the dominant cost
        on spinning-disk-backed caches under ``shuffle=True``.
        """
        f = self._get_file()
        grp = f[self.split]
        embs = [torch.from_numpy(grp[f"emb_layer{ls}"][start:end]).float() for ls in self.layers_source]
        emb = torch.cat(embs, dim=-1)
        target = torch.from_numpy(grp[f"attn_layer{self.layer_target}"][start:end]).float()
        label = torch.from_numpy(grp["labels"][start:end].astype(np.int64))
        return emb, target, label


class DistillH5Dataset(Dataset):
    """Lazy-loading dataset from a Phase 3 distillation HDF5 cache (see
    ``src.collection.build_distill_cache``).

    Each row holds the cached, *unpruned* sequence at ``blocks[prune_layer]``
    (``seq_prune``) plus the frozen model's own final CLS/patch tokens
    (``teacher_cls``/``teacher_patches``) -- everything ``DistilledPrunedBackbone
    .forward_from_seq`` and the distillation loss need, with no image
    loading or backbone forward pass required at training time.
    """

    def __init__(self, h5_path, split):
        self.h5_path = str(h5_path)
        self.split = split
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
        seq_prune = torch.from_numpy(grp["seq_prune"][idx]).float()
        teacher_cls = torch.from_numpy(grp["teacher_cls"][idx]).float()
        teacher_patches = torch.from_numpy(grp["teacher_patches"][idx]).float()
        return seq_prune, teacher_cls, teacher_patches

    def read_block(self, start, end):
        """Read rows ``[start, end)`` via one contiguous HDF5 slice per
        dataset -- see ``H5ForecastDataset.read_block``; used the same way
        by ``BlockShuffleH5Dataset``."""
        f = self._get_file()
        grp = f[self.split]
        seq_prune = torch.from_numpy(grp["seq_prune"][start:end]).float()
        teacher_cls = torch.from_numpy(grp["teacher_cls"][start:end]).float()
        teacher_patches = torch.from_numpy(grp["teacher_patches"][start:end]).float()
        return seq_prune, teacher_cls, teacher_patches


class MultiDistillH5Dataset(Dataset):
    """Concatenation of per-dataset ``DistillH5Dataset`` caches, for training
    one dataset-agnostic distilled backbone -- see ``MultiH5ForecastDataset``.

    ``__getitem__`` returns ``(seq_prune, teacher_cls, teacher_patches,
    dataset_idx)``, where ``dataset_idx`` indexes into ``self.dataset_names``.
    """

    def __init__(self, cache_paths, split):
        self.dataset_names = list(cache_paths.keys())
        self.datasets = [DistillH5Dataset(path, split) for path in cache_paths.values()]
        lengths = [len(d) for d in self.datasets]
        self._offsets = np.cumsum([0] + lengths)

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, idx):
        ds_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local_idx = idx - int(self._offsets[ds_idx])
        seq_prune, teacher_cls, teacher_patches = self.datasets[ds_idx][local_idx]
        return seq_prune, teacher_cls, teacher_patches, ds_idx


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


class BlockShuffleH5Dataset(IterableDataset):
    """Yields full training batches assembled from contiguous on-disk
    micro-blocks instead of ``batch_size`` independently-shuffled rows.

    Background: ``H5ForecastDataset``/``MultiH5ForecastDataset`` under
    ``DataLoader(..., shuffle=True)`` issue one random single-row HDF5 read
    per sample. On the spinning-disk array backing these caches that costs
    a real seek (~17ms measured here) per row, regardless of how the HDF5
    chunks are laid out -- random access is random access. This class
    keeps the *batch composition* effectively random (so SGD still sees
    i.i.d.-looking minibatches) while making the *disk access pattern*
    mostly sequential:

      1. Each underlying sub-dataset's rows are split into contiguous
         micro-blocks of ``micro_block_size`` rows.
      2. Micro-block order is reshuffled every epoch (call ``set_epoch``).
      3. Each batch is assembled from ``batch_size // micro_block_size``
         micro-blocks -- possibly from different sub-datasets -- each
         fetched with a single contiguous slice read via
         ``H5ForecastDataset.read_block``.
      4. Rows within the assembled batch are shuffled in-memory (free, no
         extra I/O) so per-sample order is still fully random.

    Net effect: ``batch_size`` random seeks/batch become
    ``batch_size // micro_block_size`` sequential reads/batch.

    Wraps any dataset whose sub-datasets implement ``read_block(start, end)``
    returning a tuple of equal-length tensors -- ``H5ForecastDataset``
    (``emb, target, label``) and ``DistillH5Dataset`` (``seq_prune,
    teacher_cls, teacher_patches``) both qualify. Yields a tuple of the same
    arity as ``read_block``, plus a trailing ``ds_idx`` tensor when wrapping
    a multi-dataset (``MultiH5ForecastDataset``/``MultiDistillH5Dataset``,
    detected via a ``.datasets`` attribute) -- matching each one's own
    ``__getitem__`` convention.

    Important: pass ``persistent_workers=False`` to the wrapping
    ``DataLoader``. ``set_epoch`` mutates this object in the main process;
    that update only reaches worker processes if they are freshly forked
    for each epoch's iteration (workers kept alive via
    ``persistent_workers=True`` would keep shuffling with the epoch-0
    seed forever).
    """

    def __init__(self, dataset, batch_size, micro_block_size=32, seed=0, drop_last=False):
        if batch_size % micro_block_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of "
                f"micro_block_size ({micro_block_size})"
            )
        self._is_multi = hasattr(dataset, "datasets")
        self.datasets = dataset.datasets if self._is_multi else [dataset]
        self.batch_size = batch_size
        self.micro_block_size = micro_block_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._microblocks = self._build_microblocks()

    def _build_microblocks(self):
        blocks = []
        for ds_idx, ds in enumerate(self.datasets):
            n = len(ds)
            for start in range(0, n, self.micro_block_size):
                blocks.append((ds_idx, start, min(start + self.micro_block_size, n)))
        return blocks

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        n_per_batch = self.batch_size // self.micro_block_size
        return max(1, len(self._microblocks) // n_per_batch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self._microblocks))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            order = order[worker_info.id::worker_info.num_workers]

        n_per_batch = self.batch_size // self.micro_block_size
        for i in range(0, len(order), n_per_batch):
            group_ids = order[i:i + n_per_batch]
            if len(group_ids) == 0 or (self.drop_last and len(group_ids) < n_per_batch):
                break
            collected, ds_idxs = None, []
            for bi in group_ids:
                ds_idx, start, end = self._microblocks[bi]
                block = self.datasets[ds_idx].read_block(start, end)
                if collected is None:
                    collected = [[] for _ in block]
                for parts, tensor in zip(collected, block):
                    parts.append(tensor)
                if self._is_multi:
                    ds_idxs.append(torch.full((end - start,), ds_idx, dtype=torch.long))

            tensors = [torch.cat(parts, dim=0) for parts in collected]
            perm = torch.randperm(tensors[0].shape[0])
            tensors = [t[perm] for t in tensors]
            if self._is_multi:
                ds_idx_t = torch.cat(ds_idxs, dim=0)
                yield (*tensors, ds_idx_t[perm])
            else:
                yield tuple(tensors)
