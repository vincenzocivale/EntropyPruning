"""LoRA-adapted, forecaster-pruned TITAN for pruned-vs-full embedding distillation.

Mirrors ``src/models/online_tile_eaf.py``'s ``PrunedLoRAEncoder`` (which prunes a
*tile encoder's* patches mid-forward, LoRA-adapts the remaining blocks, and
distills against the frozen full embedding), one level up: this prunes TITAN's
own *tile bag* mid-forward using a frozen WSI-EAF forecaster
(``src/models/wsi/dense_forecaster.py``), then LoRA-adapts TITAN's own
remaining vision-encoder blocks to recover whatever the pruned-vs-full gap
costs. ``prune_layer`` must match the forecaster's own trained source layer
(``--titan-hidden-layer`` at forecaster-training time) -- the forecaster was
only ever shown that exact layer's hidden state.

TITAN's own attention uses ALiBi: a single bias tensor, sized for the *full*
token count, is computed once and threaded through every vision-encoder block
unchanged (see the cached ``vision_transformer.py``'s
``VisionTransformer.forward_features``/``CustomSequential``). Pruning mid-
sequence therefore also requires re-slicing that bias tensor to the kept
tokens for every later block -- unlike the tile-encoder case (plain learned/no
positional bias), simply gathering the token embeddings is not enough on its
own.

Only ``vision_encoder`` gets LoRA (TITAN's separate text tower, used for
captioning/contrastive training, is irrelevant to slide-level tile pruning and
is left untouched). The ``AttentionalPooler`` readout is left frozen too --
only the plain ViT-style ``Attention``/``Mlp`` submodules inside
``vision_encoder.blocks.modules_list`` (targeted by ``qkv``/``proj``/``fc1``/
``fc2``, exactly the tile-level convention) are LoRA-adapted.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel


def _find_block_list(vision_encoder: nn.Module) -> nn.ModuleList:
    """Locate the 6 real ViT blocks generically (same heuristic already proven in
    ``src/wsi_pipeline/wsi_models/titan_attention.py::_find_block_list``, kept as
    an independent copy here since this module has no dependency on that one)."""
    aliases = {"blocks", "modules_list", "attn_blocks", "layers", "layer"}
    candidates = [
        (name, module)
        for name, module in vision_encoder.named_modules()
        if isinstance(module, nn.ModuleList) and name.rsplit(".", 1)[-1] in aliases
    ]
    if not candidates:
        raise RuntimeError("Could not locate TITAN's vision-encoder block list")
    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    return candidates[0][1]


def _gather_pairwise(bias: torch.Tensor, full_index: torch.Tensor) -> torch.Tensor:
    """bias: [B,H,S,S] -> [B,H,K,K], keeping only rows/cols in full_index ([B,K])."""
    batch, heads, _seq_q, seq_k = bias.shape
    keep = full_index.shape[1]
    row_index = full_index.view(batch, 1, keep, 1).expand(batch, heads, keep, seq_k)
    bias = torch.gather(bias, 2, row_index)
    col_index = full_index.view(batch, 1, 1, keep).expand(batch, heads, keep, keep)
    return torch.gather(bias, 3, col_index)


def _make_pruned_blocks_forward(
    modules_list: nn.ModuleList,
    *,
    prune_layer: int,
    forecaster: nn.Module,
    keep_ratio: float,
    num_prefix_tokens: int,
):
    """Build a drop-in replacement for ``CustomSequential.forward`` that prunes the
    token sequence (and re-slices the ALiBi bias to match) right after
    ``modules_list[prune_layer]`` runs, then continues with the remaining blocks
    on the pruned sequence only."""

    def pruned_forward(x: torch.Tensor, attn_mask: torch.Tensor | None, bg_mask=None):
        for module in modules_list[: prune_layer + 1]:
            x = module(x, attn_mask, bg_mask)

        prefix = x[:, :num_prefix_tokens]
        patches = x[:, num_prefix_tokens:]
        batch, n_patches, dim = patches.shape
        keep = max(1, int(round(n_patches * keep_ratio)))
        with torch.no_grad():
            scores = forecaster(patches)  # [B, n_patches], frozen forecaster, no grad
            indices = scores.topk(keep, dim=-1).indices  # [B, keep]
        gather_index = indices.unsqueeze(-1).expand(-1, -1, dim)
        kept_patches = torch.gather(patches, dim=1, index=gather_index)
        x = torch.cat((prefix, kept_patches), dim=1)

        if attn_mask is not None:
            prefix_index = torch.arange(num_prefix_tokens, device=indices.device)
            prefix_index = prefix_index.unsqueeze(0).expand(batch, -1)
            full_index = torch.cat([prefix_index, indices + num_prefix_tokens], dim=1)
            attn_mask = _gather_pairwise(attn_mask, full_index)

        for module in modules_list[prune_layer + 1 :]:
            x = module(x, attn_mask, bg_mask)
        return x

    return pruned_forward


class PrunedLoRATitanEncoder(nn.Module):
    """LoRA-adapted TITAN vision encoder with forecaster-guided tile pruning.

    Distillation target is always the *cached* frozen-TITAN slide embedding
    (already written by ``scripts/wsi_eaf_infer_wsi_fm.py`` into the wsi_eaf
    cache's ``slide_embedding`` for every slide) -- unlike the tile-encoder
    case, there is no need to ever run a second, unpruned forward through
    TITAN during training; the frozen teacher output was already computed
    once, offline, for the entire corpus.
    """

    def __init__(
        self,
        titan_model: nn.Module,
        forecaster: nn.Module,
        *,
        prune_layer: int,
        keep_ratio: float,
        patch_size_level0: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        block_list = _find_block_list(titan_model.vision_encoder)
        if not 0 <= prune_layer < len(block_list) - 1:
            raise ValueError(
                f"prune_layer={prune_layer} must leave at least one block after it "
                f"(got {len(block_list)} blocks total)"
            )
        self.prune_layer = prune_layer
        self.keep_ratio = keep_ratio
        self.patch_size_level0 = patch_size_level0
        self.num_prefix_tokens = 1  # TITAN: CLS only, no register tokens

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.titan = LoraModel(titan_model, config, adapter_name="default")

        self.forecaster = forecaster
        self.forecaster.eval()
        for parameter in self.forecaster.parameters():
            parameter.requires_grad_(False)

    @property
    def raw_titan(self) -> nn.Module:
        # LoraModel injects LoRA layers in-place; `.model` is still the same live
        # Titan object with its original attribute structure (vision_encoder,
        # text_encoder, encode_slide_from_patch_features, ...).
        return self.titan.model

    def train(self, mode: bool = True) -> "PrunedLoRATitanEncoder":
        super().train(mode)
        self.forecaster.eval()  # frozen ranking function must never see dropout
        return self

    @contextmanager
    def _pruning_context(self) -> Iterator[None]:
        block_list = _find_block_list(self.raw_titan.vision_encoder)
        blocks_module = self.raw_titan.vision_encoder.blocks
        original_forward = blocks_module.forward
        blocks_module.forward = _make_pruned_blocks_forward(
            block_list,
            prune_layer=self.prune_layer,
            forecaster=self.forecaster,
            keep_ratio=self.keep_ratio,
            num_prefix_tokens=self.num_prefix_tokens,
        )
        try:
            yield
        finally:
            blocks_module.forward = original_forward

    def forward(self, tile_embeddings: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """tile_embeddings: [N,768] (CONCH v1.5 tile features, un-batched -- matches
        ``Titan.encode_slide_from_patch_features``'s own convention). coords: [N,>=2].

        Returns the pruned+LoRA slide embedding, [768] (post ``vision_encoder.proj``,
        same convention as ``TitanAdapter``/the cached ``slide_embedding``).
        """
        with self._pruning_context():
            embedding = self.raw_titan.encode_slide_from_patch_features(
                tile_embeddings, coords, self.patch_size_level0
            )
        while embedding.dim() > 1 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        return embedding

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only LoRA tensors -- never the full frozen TITAN checkpoint."""
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        parameters = dict(self.named_parameters())
        unknown = sorted(set(state) - set(parameters))
        if unknown:
            raise KeyError(f"Unknown trainable keys: {unknown[:5]}")
        missing = sorted(
            name for name, parameter in parameters.items()
            if parameter.requires_grad and name not in state
        )
        if missing:
            raise KeyError(f"Missing trainable keys: {missing[:5]}")
        with torch.no_grad():
            for name, value in state.items():
                parameter = parameters[name]
                if tuple(parameter.shape) != tuple(value.shape):
                    raise ValueError(f"Shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
