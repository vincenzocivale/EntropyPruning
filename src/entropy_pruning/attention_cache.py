from __future__ import annotations

import types
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm


def build_attention_cache(
    model,
    loaders: dict,
    device,
    source_layers: list[int],
    target_layers: list[int],
    save_path: str | Path,
    flush_every: int = 16,
    compression: str | None = "lzf",
    chunk_size: int | None = None,
):
    """
    Cache embeddings from source layers and cls->patch attention from target layers.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, "w") as h5f:
        for split_name, loader in loaders.items():
            n_total = len(loader.dataset)
            split_grp = h5f.create_group(split_name)
            batch_chunk = chunk_size or max(1, min(64, getattr(loader, "batch_size", 64) or 64))

            labels_ds = split_grp.create_dataset(
                "labels",
                shape=(n_total,),
                maxshape=(None,),
                dtype="i4",
                compression=compression,
            )

            emb_datasets = {}
            attn_datasets = {}
            cache = {}
            source_layers_set = set(source_layers)
            target_layers_set = set(target_layers)
            orig_target_fwds = {}
            source_pre_handles = []
            ptr = 0
            labels_buf = []
            emb_buf = {layer: [] for layer in source_layers}
            attn_buf = {layer: [] for layer in target_layers}

            def make_target_hook(idx):
                def fwd(self, x, attn_mask=None, *args, **kwargs):
                    bsz, n_tok, dim = x.shape
                    qkv = self.qkv(x).reshape(bsz, n_tok, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q, k = self.q_norm(q), self.k_norm(k)
                    attn = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)

                    cache[f"attn_{idx}"] = attn[:, :, 0, 1:].mean(1).detach().cpu().half()

                    x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(bsz, n_tok, dim)
                    return self.proj_drop(self.proj(x))

                return fwd

            def make_source_prehook(idx):
                def prehook(module, inputs):
                    x = inputs[0]
                    cache[f"emb_{idx}"] = x[:, 1:].detach().cpu().half()

                return prehook

            def flush_buffers():
                nonlocal ptr
                if not labels_buf:
                    return

                labels_np = np.concatenate(labels_buf, axis=0)
                bsz = labels_np.shape[0]
                labels_ds[ptr : ptr + bsz] = labels_np

                for layer in source_layers:
                    emb_np = torch.cat(emb_buf[layer], dim=0).numpy()
                    emb_datasets[layer][ptr : ptr + bsz] = emb_np
                    emb_buf[layer].clear()

                for layer in target_layers:
                    attn_np = torch.cat(attn_buf[layer], dim=0).numpy()
                    attn_datasets[layer][ptr : ptr + bsz] = attn_np
                    attn_buf[layer].clear()

                labels_buf.clear()
                ptr += bsz

            for i, block in enumerate(model.backbone.model.blocks):
                if i in source_layers_set:
                    source_pre_handles.append(block.attn.register_forward_pre_hook(make_source_prehook(i)))
                if i in target_layers_set:
                    orig_target_fwds[i] = block.attn.forward
                    block.attn.forward = types.MethodType(make_target_hook(i), block.attn)

            with torch.no_grad():
                for i_batch, (imgs, labels) in enumerate(tqdm(loader, desc=f"cache:{split_name}")):
                    cache.clear()
                    _ = model(imgs.to(device))

                    if i_batch == 0:
                        n_patches, embed_dim = cache[f"emb_{source_layers[0]}"][0].shape
                        for layer in source_layers:
                            emb_datasets[layer] = split_grp.create_dataset(
                                f"emb_layer{layer}",
                                shape=(n_total, n_patches, embed_dim),
                                maxshape=(None, n_patches, embed_dim),
                                dtype="f2",
                                compression=compression,
                                chunks=(batch_chunk, n_patches, embed_dim),
                            )
                        for layer in target_layers:
                            attn_datasets[layer] = split_grp.create_dataset(
                                f"attn_layer{layer}",
                                shape=(n_total, n_patches),
                                maxshape=(None, n_patches),
                                dtype="f2",
                                compression=compression,
                                chunks=(batch_chunk, n_patches),
                            )

                    for layer in source_layers:
                        emb_buf[layer].append(cache[f"emb_{layer}"])
                    for layer in target_layers:
                        attn_buf[layer].append(cache[f"attn_{layer}"])
                    labels_buf.append(labels.numpy())

                    if (i_batch + 1) % max(1, flush_every) == 0:
                        flush_buffers()
                        h5f.flush()

                flush_buffers()
                h5f.flush()

            for handle in source_pre_handles:
                handle.remove()
            for i, block in enumerate(model.backbone.model.blocks):
                if i in orig_target_fwds:
                    block.attn.forward = orig_target_fwds[i]

            if ptr < n_total:
                labels_ds.resize(ptr, axis=0)
                for layer in source_layers:
                    emb_datasets[layer].resize(ptr, axis=0)
                for layer in target_layers:
                    attn_datasets[layer].resize(ptr, axis=0)
