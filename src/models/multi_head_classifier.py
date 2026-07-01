"""Multi-head classifiers for simultaneous training on heterogeneous Thunder datasets.

Classes:
    MultiHeadThunderClassifier         — shared backbone + one head per dataset (Phase 1)
    MultiHeadLoRAWithForecasterPruning — same + forecaster-guided token pruning (Phase 3)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel

from .backbone_adapter import ThunderBackboneAdapter


def _make_heads(adapter: ThunderBackboneAdapter, dataset_info: dict, dropout: float) -> nn.ModuleDict:
    return nn.ModuleDict({
        str(idx): nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, info["n_classes"]),
        )
        for idx, info in dataset_info.items()
    })


def _apply_adaptation(backbone: nn.Module, adaptation: str,
                       lora_r: int, lora_alpha: int) -> nn.Module:
    if adaptation == "lora":
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1, bias="none",
        )
        return LoraModel(backbone, cfg, adapter_name="default")
    if adaptation == "linear_probing":
        for p in backbone.parameters():
            p.requires_grad_(False)
    elif adaptation == "bitfit":
        for name, p in backbone.named_parameters():
            p.requires_grad_("bias" in name)
    elif adaptation != "full":
        raise ValueError(f"Unknown adaptation: {adaptation!r}")
    return backbone


class MultiHeadThunderClassifier(nn.Module):
    """Shared backbone + one classification head per Thunder dataset.

    Designed for Phase 1 multi-dataset training.  Forward pass routes each
    sample to its dataset's head based on the ``dataset_indices`` tensor.

    Args:
        backbone:     Raw timm VisionTransformer from get_model_from_name.
        adapter:      ThunderBackboneAdapter for the backbone.
        dataset_info: {dataset_idx: {n_classes, class_names, name}}.
        adaptation:   One of "lora", "linear_probing", "full", "bitfit".
        lora_r, lora_alpha: LoRA params (only used when adaptation="lora").
        dropout:      Head dropout rate.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ThunderBackboneAdapter,
        dataset_info: dict,
        adaptation: str = "lora",
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = adapter
        self._dataset_info = dataset_info
        self._adaptation = adaptation
        self.backbone = _apply_adaptation(backbone, adaptation, lora_r, lora_alpha)
        self.heads = _make_heads(adapter, dataset_info, dropout)

    @property
    def raw_backbone(self) -> nn.Module:
        return self.backbone.model if self._adaptation == "lora" else self.backbone

    @property
    def trainable_backbone_params(self) -> List[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return emb[:, 0] if emb.ndim == 3 else emb

    def forward(self, images: torch.Tensor, dataset_indices: torch.Tensor) -> dict:
        """
        Args:
            images:          (B, C, H, W)
            dataset_indices: (B,) int tensor — dataset index per sample.

        Returns:
            {dataset_idx (int): logits (k, n_classes_for_that_dataset)}
            where k = number of samples in the batch belonging to dataset_idx.
        """
        emb = self._embed(images)
        out: dict = {}
        for idx_val in dataset_indices.unique():
            k = int(idx_val.item())
            mask = dataset_indices == idx_val
            out[k] = self.heads[str(k)](emb[mask])
        return out


class MultiHeadLoRAWithForecasterPruning(nn.Module):
    """Multi-head Phase-3 model: LoRA backbone + forecaster-guided token pruning.

    Identical pruning logic to GenericLoRAWithForecasterPruning, extended to
    route samples to per-dataset heads.

    Args:
        backbone:     Raw timm VisionTransformer from get_model_from_name.
        adapter:      ThunderBackboneAdapter for the backbone.
        dataset_info: {dataset_idx: {n_classes, class_names, name}}.
        forecaster:   Trained AttentionForecaster (must be frozen before passing in).
        prune_layer:  Block index where pruning is applied (0-indexed).
        keep_ratio:   Fraction of spatial patch tokens to keep.
        lora_r, lora_alpha: LoRA params.
        dropout:      Head dropout rate.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ThunderBackboneAdapter,
        dataset_info: dict,
        forecaster: nn.Module,
        prune_layer: int,
        keep_ratio: float,
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = adapter
        self._dataset_info = dataset_info
        self.prune_layer = prune_layer
        self.keep_ratio = keep_ratio
        self.forecaster = forecaster

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1, bias="none",
        )
        self.backbone = LoraModel(backbone, lora_cfg, adapter_name="default")
        self.heads = _make_heads(adapter, dataset_info, dropout)

    @property
    def raw_backbone(self) -> nn.Module:
        return self.backbone.model

    def _pruned_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone with pruning hook at prune_layer; return CLS embedding."""
        num_prefix = self.adapter.num_prefix_tokens
        keep_ratio = self.keep_ratio
        forecaster = self.forecaster
        training = self.training
        raw = self.raw_backbone
        orig_fwd = raw.blocks[self.prune_layer].forward

        def _hook(x):
            x = orig_fwd(x)
            B, _, D = x.shape
            prefix = x[:, :num_prefix]
            patches = x[:, num_prefix:]
            N = patches.shape[1]

            with torch.no_grad():
                scores = forecaster(patches)

            k_keep = max(1, int(N * keep_ratio))
            topk_vals = scores.topk(k_keep, dim=-1).values
            threshold = topk_vals[:, -1:]
            soft_mask = torch.sigmoid((scores - threshold) / 0.05)
            hard_mask = (scores >= threshold).float()
            st_mask = hard_mask - soft_mask.detach() + soft_mask

            topk_idx = scores.topk(k_keep, dim=-1).indices
            if training:
                masked = patches * st_mask.unsqueeze(-1)
                kept = torch.stack([masked[b][topk_idx[b]] for b in range(B)])
            else:
                kept = torch.stack([patches[b][topk_idx[b]] for b in range(B)])
            return torch.cat([prefix, kept], dim=1)

        raw.blocks[self.prune_layer].forward = _hook
        emb = self.backbone(x)
        raw.blocks[self.prune_layer].forward = orig_fwd
        return emb[:, 0] if emb.ndim == 3 else emb

    def forward(self, images: torch.Tensor, dataset_indices: torch.Tensor) -> dict:
        """
        Returns:
            {dataset_idx (int): logits (k, n_classes)} for each unique idx in batch.
        """
        emb = self._pruned_embed(images)
        out: dict = {}
        for idx_val in dataset_indices.unique():
            k = int(idx_val.item())
            mask = dataset_indices == idx_val
            out[k] = self.heads[str(k)](emb[mask])
        return out
