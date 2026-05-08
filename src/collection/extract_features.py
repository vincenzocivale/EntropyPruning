import types
import numpy as np
import torch
import h5py
from tqdm.auto import tqdm


def collect_and_save_dataset(model, loaders_dict, device,
                             layers_source, layers_target, save_path):
    """Extract embeddings and CLS attention maps from a trained model and save to HDF5.

    Hooks into each attention layer to capture:
    - Patch embeddings at source layers (spatial tokens only, excluding prefix tokens).
    - CLS-to-patch attention weights at target layers (mean over heads).

    Compatible with any GenericLoRAClassifier wrapping a ThunderBackboneAdapter.
    Dimensions (n_patches, embed_dim) are inferred from model.adapter.

    Args:
        model:        GenericLoRAClassifier (frozen, eval mode). Must have .adapter.
        loaders_dict: {split_name: DataLoader} yielding (imgs, labels) tuples.
        device:       torch device.
        layers_source: list of block indices to extract patch embeddings from.
        layers_target: int or list of block indices to extract CLS attention from.
        save_path:    output HDF5 file path.

    HDF5 layout per split:
        labels          (N,)            int32
        emb_layer{L}    (N, P, D)       float16   P=n_patches, D=embed_dim
        attn_layer{L}   (N, P)          float16
    """
    if isinstance(layers_target, int):
        layers_target = [layers_target]

    n_patches = model.adapter.n_patches
    embed_dim = model.adapter.embed_dim
    num_prefix = model.adapter.num_prefix_tokens

    with h5py.File(save_path, 'w') as f:
        for split_name, loader in loaders_dict.items():
            print(f"\nCollecting {split_name}...")

            n_total = len(loader.dataset)
            grp = f.create_group(split_name)
            ds_label = grp.create_dataset("labels", shape=(n_total,), dtype='i4')
            ds_attn = {
                lt: grp.create_dataset(
                    f"attn_layer{lt}", shape=(n_total, n_patches),
                    dtype='f2', chunks=(128, n_patches))
                for lt in layers_target
            }
            ds_embs = {
                ls: grp.create_dataset(
                    f"emb_layer{ls}", shape=(n_total, n_patches, embed_dim),
                    dtype='f2', chunks=(128, n_patches, embed_dim))
                for ls in layers_source
            }

            orig = {}
            cache = {}

            def make_hook(idx):
                def fwd(self, x, *args, **kwargs):
                    attn_mask = kwargs.get('attn_mask', None)
                    if attn_mask is None and len(args) > 0:
                        attn_mask = args[0]

                    B, N, C = x.shape
                    qkv = self.qkv(x).reshape(
                        B, N, 3, self.num_heads, self.head_dim
                    ).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q, k = self.q_norm(q), self.k_norm(k)

                    attn = (q @ k.transpose(-2, -1) * self.scale)
                    if attn_mask is not None:
                        attn = attn + attn_mask
                    attn = attn.softmax(-1)

                    if idx in layers_source:
                        cache[f"emb_{idx}"] = (
                            x[:, num_prefix:].detach().cpu().half()
                        )
                    if idx in layers_target:
                        cache[f"attn_{idx}"] = (
                            attn[:, :, 0, num_prefix:].mean(1).detach().cpu().half()
                        )

                    x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
                    return self.proj_drop(self.proj(x))
                return fwd

            for i, block in enumerate(model.raw_backbone.blocks):
                if i in layers_source or i in layers_target:
                    orig[i] = block.attn.forward
                    block.attn.forward = types.MethodType(make_hook(i), block.attn)

            FLUSH_EVERY = 128
            buf_labels = []
            buf_attn = {lt: [] for lt in layers_target}
            buf_embs = {ls: [] for ls in layers_source}

            def flush(ptr):
                if not buf_labels:
                    return ptr
                B = sum(len(x) for x in buf_labels)
                ds_label[ptr:ptr + B] = np.concatenate(buf_labels)
                for lt in layers_target:
                    ds_attn[lt][ptr:ptr + B] = torch.cat(buf_attn[lt]).numpy()
                for ls in layers_source:
                    ds_embs[ls][ptr:ptr + B] = torch.cat(buf_embs[ls]).numpy()
                buf_labels.clear()
                for lt in layers_target: buf_attn[lt].clear()
                for ls in layers_source: buf_embs[ls].clear()
                return ptr + B

            ptr = 0
            with torch.no_grad():
                for i_batch, (imgs, labels) in enumerate(tqdm(loader, desc=split_name)):
                    cache.clear()
                    model(imgs.to(device))
                    buf_labels.append(labels.numpy())
                    for lt in layers_target:
                        buf_attn[lt].append(cache[f"attn_{lt}"])
                    for ls in layers_source:
                        buf_embs[ls].append(cache[f"emb_{ls}"])
                    if (i_batch + 1) % FLUSH_EVERY == 0:
                        ptr = flush(ptr)

            ptr = flush(ptr)

            for i, block in enumerate(model.raw_backbone.blocks):
                if i in orig:
                    block.attn.forward = orig[i]

            print(f"  {split_name}: {ptr} samples saved")
