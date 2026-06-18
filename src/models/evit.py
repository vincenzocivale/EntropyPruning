"""EViT token reorganization for timm-style Vision Transformers.

This module adapts the method from
https://github.com/youweiliang/evit to the Thunder foundation models used by
EAF.  The EViT operation itself is kept parameter-free: selected transformer
blocks use the class-token attention after MHSA to keep attentive patch tokens,
optionally fusing the inattentive patches into one extra token before the MLP.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel

from .backbone_adapter import ThunderBackboneAdapter


def complement_indices(idx: torch.Tensor, size: int) -> torch.Tensor:
    """Return per-row indices in ``range(size)`` not present in ``idx``."""
    if idx.ndim != 2:
        raise ValueError("idx must have shape (B, K)")
    marker = torch.ones(idx.shape[0], size, dtype=torch.bool, device=idx.device)
    marker.scatter_(1, idx, False)
    return marker.nonzero(as_tuple=False)[:, 1].reshape(idx.shape[0], size - idx.shape[1])


def parse_evit_drop_locs(value: str | Iterable[int], n_blocks: int) -> List[int]:
    """Parse and validate EViT shrink block indices."""
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("(") or raw.startswith("["):
            raw = raw.strip("()[]")
        locs = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        locs = [int(v) for v in value]
    locs = sorted(set(locs))
    if not locs:
        raise ValueError("EViT needs at least one drop location")
    invalid = [loc for loc in locs if loc < 0 or loc >= n_blocks]
    if invalid:
        raise ValueError(f"EViT drop locations out of range for {n_blocks} blocks: {invalid}")
    return locs


def adjust_evit_keep_rate(
    epoch: int,
    step_in_epoch: int,
    steps_per_epoch: int,
    base_keep_rate: float,
    shrink_start_epoch: int = 10,
    shrink_epochs: int = 0,
) -> float:
    """Linear keep-rate schedule matching EViT's gradual shrinking knob."""
    if shrink_epochs <= 0:
        return base_keep_rate
    progress = (
        (epoch - shrink_start_epoch) * steps_per_epoch + step_in_epoch
    ) / max(1, shrink_epochs * steps_per_epoch)
    progress = min(1.0, max(0.0, progress))
    return 1.0 - (1.0 - base_keep_rate) * progress


class LoRAWithEViTPruning(nn.Module):
    """LoRA classifier with EViT token reorganization.

    EViT itself introduces no parameters.  LoRA is used only as this project's
    downstream adaptation mechanism for large frozen foundation models; the
    token selection and optional fusion follow the original EViT block logic:
    attention, residual add, token reorganization, then MLP.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ThunderBackboneAdapter,
        n_classes: int,
        base_keep_rate: float,
        drop_locs: Sequence[int],
        fuse_token: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0 < base_keep_rate <= 1:
            raise ValueError("base_keep_rate must be in (0, 1]")
        self.adapter = adapter
        self.base_keep_rate = float(base_keep_rate)
        self.current_keep_rate = float(base_keep_rate)
        self.drop_locs = parse_evit_drop_locs(drop_locs, adapter.n_blocks)
        self.fuse_token = fuse_token
        self.num_prefix = adapter.num_prefix_tokens

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=evit_lora_targets(adapter, self.drop_locs),
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )

    @property
    def raw_backbone(self) -> nn.Module:
        return self.backbone.model

    def set_keep_rate(self, keep_rate: float) -> None:
        if not 0 < keep_rate <= 1:
            raise ValueError("keep_rate must be in (0, 1]")
        self.current_keep_rate = float(keep_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        originals = {}
        for block_idx in self.drop_locs:
            block = self.raw_backbone.blocks[block_idx]
            originals[block_idx] = block.forward
            block.forward = self._make_evit_block_forward(block)
        try:
            features = self.backbone(x)
        finally:
            for block_idx, forward in originals.items():
                self.raw_backbone.blocks[block_idx].forward = forward

        if isinstance(features, tuple):
            features = features[0]
        if features.ndim == 3:
            features = features[:, 0]
        return self.head(features)

    def _make_evit_block_forward(self, block: nn.Module):
        def _forward(x: torch.Tensor) -> torch.Tensor:
            attn_out, idx, cls_attn = self._attention_with_cls_scores(block.attn, block.norm1(x))
            x = x + self._drop_path(block, self._layer_scale(block, "ls1", attn_out), branch=1)

            if idx is not None:
                x = self._reorganize_tokens(x, idx, cls_attn)

            mlp_out = block.mlp(block.norm2(x))
            x = x + self._drop_path(block, self._layer_scale(block, "ls2", mlp_out), branch=2)
            return x

        return _forward

    def _attention_with_cls_scores(self, attn: nn.Module, x: torch.Tensor):
        B, N, C = x.shape
        num_heads = int(getattr(attn, "num_heads", 1))
        head_dim = C // num_heads
        scale = getattr(attn, "scale", head_dim ** -0.5)

        qkv = attn.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm"):
            k = attn.k_norm(k)

        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        if hasattr(attn, "attn_drop"):
            attn_weights = attn.attn_drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        out = attn.proj_drop(out) if hasattr(attn, "proj_drop") else out

        n_patches = N - self.num_prefix
        keep_tokens = math.ceil(self.current_keep_rate * n_patches)
        keep_tokens = max(1, min(keep_tokens, n_patches))
        if keep_tokens == n_patches:
            return out, None, None

        cls_attn = attn_weights[:, :, 0, self.num_prefix:].mean(dim=1)
        _, idx = torch.topk(cls_attn, keep_tokens, dim=1, largest=True, sorted=True)
        return out, idx, cls_attn

    def _reorganize_tokens(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        cls_attn: torch.Tensor,
    ) -> torch.Tensor:
        prefix = x[:, :self.num_prefix]
        patches = x[:, self.num_prefix:]
        C = patches.shape[-1]

        kept = patches.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
        if not self.fuse_token:
            return torch.cat([prefix, kept], dim=1)

        compl = complement_indices(idx, patches.shape[1])
        inattentive = patches.gather(1, compl.unsqueeze(-1).expand(-1, -1, C))
        inattentive_attn = cls_attn.gather(1, compl)
        extra_token = torch.sum(inattentive * inattentive_attn.unsqueeze(-1), dim=1, keepdim=True)
        return torch.cat([prefix, kept, extra_token], dim=1)

    @staticmethod
    def _layer_scale(block: nn.Module, name: str, x: torch.Tensor) -> torch.Tensor:
        layer = getattr(block, name, None)
        return layer(x) if layer is not None else x

    @staticmethod
    def _drop_path(block: nn.Module, x: torch.Tensor, branch: int) -> torch.Tensor:
        branch_drop = getattr(block, f"drop_path{branch}", None)
        if branch_drop is not None:
            return branch_drop(x)
        drop_path = getattr(block, "drop_path", None)
        return drop_path(x) if drop_path is not None else x


def evit_lora_targets(adapter: ThunderBackboneAdapter, drop_locs: Sequence[int]) -> list:
    """LoRA targets for blocks whose inputs or MLPs are affected by EViT."""
    first = min(drop_locs)
    targets = []
    for i in range(first, adapter.n_blocks):
        targets += [
            f"blocks.{i}.attn.qkv",
            f"blocks.{i}.attn.proj",
            f"blocks.{i}.mlp.fc1",
            f"blocks.{i}.mlp.fc2",
        ]
    return targets
