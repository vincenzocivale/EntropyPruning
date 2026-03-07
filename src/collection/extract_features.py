import types
import numpy as np
import torch
import h5py
from tqdm.auto import tqdm


def collect_and_save_dataset(model, loaders_dict, device,
                             layers_source, layer_target, save_path):
    """Extract embeddings and attention maps from a trained model and save to HDF5.

    Hooks into the attention layers to capture:
    - Patch embeddings at each source layer (excluding CLS token)
    - CLS attention weights at the target layer (averaged over heads)

    Args:
        model: Trained UNILoRAClassifier (frozen, eval mode)
        loaders_dict: dict of {split_name: DataLoader}
        device: torch device
        layers_source: list of layer indices to extract embeddings from
        layer_target: layer index to extract attention from
        save_path: path to save the HDF5 file
    """
    with h5py.File(save_path, 'w') as f:
        for split_name, loader in loaders_dict.items():
            print(f"\nCollecting {split_name}...")

            n_total = len(loader.dataset)
            n_patches = 196
            embed_dim = 1024

            grp = f.create_group(split_name)
            ds_label = grp.create_dataset(
                "labels", shape=(n_total,), maxshape=(None,),
                dtype='i4', compression="gzip")
            ds_attn = grp.create_dataset(
                f"attn_layer{layer_target}",
                shape=(n_total, n_patches), maxshape=(None, n_patches),
                dtype='f2', compression="gzip", chunks=(64, n_patches))
            ds_embs = {
                l: grp.create_dataset(
                    f"emb_layer{l}",
                    shape=(n_total, n_patches, embed_dim),
                    maxshape=(None, n_patches, embed_dim),
                    dtype='f2', compression="gzip",
                    chunks=(64, n_patches, embed_dim))
                for l in layers_source
            }

            orig = {}
            cache = {}

            def make_hook(idx):
                def fwd(self, x):
                    B, N, C = x.shape
                    qkv = self.qkv(x).reshape(
                        B, N, 3, self.num_heads, self.head_dim
                    ).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q, k = self.q_norm(q), self.k_norm(k)
                    attn = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
                    if idx in layers_source:
                        cache[f"emb_{idx}"] = x[:, 1:].detach().cpu().half()
                    if idx == layer_target:
                        cache["attn"] = attn[:, :, 0, 1:].mean(1).detach().cpu().half()
                    x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
                    return self.proj_drop(self.proj(x))
                return fwd

            for i, block in enumerate(model.backbone.model.blocks):
                orig[i] = block.attn.forward
                block.attn.forward = types.MethodType(make_hook(i), block.attn)

            FLUSH_EVERY = 16
            buf_labels = []
            buf_attn = []
            buf_embs = {l: [] for l in layers_source}

            def flush(ptr):
                if not buf_labels:
                    return ptr
                B = sum(len(x) for x in buf_labels)
                ds_label[ptr:ptr + B] = np.concatenate(buf_labels)
                ds_attn[ptr:ptr + B] = torch.cat(buf_attn).numpy()
                for l in layers_source:
                    ds_embs[l][ptr:ptr + B] = torch.cat(buf_embs[l]).numpy()
                buf_labels.clear()
                buf_attn.clear()
                for l in layers_source:
                    buf_embs[l].clear()
                return ptr + B

            ptr = 0
            with torch.no_grad():
                for i_batch, (imgs, labels) in enumerate(
                    tqdm(loader, desc=split_name)
                ):
                    cache.clear()
                    model(imgs.to(device))
                    buf_labels.append(labels.numpy())
                    buf_attn.append(cache["attn"])
                    for l in layers_source:
                        buf_embs[l].append(cache[f"emb_{l}"])
                    if (i_batch + 1) % FLUSH_EVERY == 0:
                        ptr = flush(ptr)

            ptr = flush(ptr)

            for i, block in enumerate(model.backbone.model.blocks):
                block.attn.forward = orig[i]

            if ptr < n_total:
                ds_label.resize(ptr, axis=0)
                ds_attn.resize(ptr, axis=0)
                for l in layers_source:
                    ds_embs[l].resize(ptr, axis=0)

            print(f"  {split_name}: {ptr} samples saved")
