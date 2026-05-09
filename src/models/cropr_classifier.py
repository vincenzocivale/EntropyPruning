"""CropR token pruning wrapper for pre-trained Thunder backbones."""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel

from .backbone_adapter import ThunderBackboneAdapter
from ..baselines.cropr.cropr import Cropr


class CroprClassifier(nn.Module):
    """
    CropR progressive token pruning on top of a pre-trained Thunder backbone.

    One Cropr module is inserted after each transformer block except the last.
    Each Cropr module scores patch tokens via cross-attention with a learnable
    query and drops the lowest-scoring ones.  Position 0 (CLS) is always kept.

    During training returns [main_logits, aux_0, ..., aux_k] so the caller can
    sum cross-entropy losses over all heads.  During eval (model.eval()) returns
    only the final main_logits tensor — compatible with evaluate() from
    src/evaluation/metrics.py.

    Note on register tokens: models with register tokens at positions 1..k
    (e.g. kaiko_vit, dinov2base/large) may have those tokens pruned, because
    the original Cropr implementation protects only position 0.  Prefer models
    with num_prefix_tokens=1 (uni, virchow, hoptimus) for reliable results.

    Args:
        backbone:        raw timm ViT from get_model_from_name.
        adapter:         ThunderBackboneAdapter wrapping the same backbone.
        n_classes:       number of output classes.
        pruning_rate:    tokens removed per block (default 8, as in the paper).
        num_queries:     learnable queries per Cropr cross-attn head (default 1).
        cropr_num_heads: attention heads inside each Cropr module (default 1).
        pre_attn_norm:   LayerNorm before Cropr cross-attn (default False).
        q_proj / k_proj / v_proj: linear projections in Cropr cross-attn.
        mlp:             MLP sub-layer inside each Cropr module (default True).
        mlp_ratio:       hidden-dim multiplier for Cropr MLP (default 4.0).
        freeze_backbone: freeze all backbone weights; only Cropr + head are trained.
        lora_r:          LoRA rank (ignored when freeze_backbone=True).
        lora_alpha:      LoRA alpha (ignored when freeze_backbone=True).
        dropout:         dropout in the classification head (default 0.1).
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ThunderBackboneAdapter,
        n_classes: int,
        pruning_rate: int = 8,
        num_queries: int = 1,
        cropr_num_heads: int = 1,
        pre_attn_norm: bool = False,
        q_proj: bool = False,
        k_proj: bool = False,
        v_proj: bool = False,
        mlp: bool = True,
        mlp_ratio: float = 4.0,
        freeze_backbone: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = adapter
        self._freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)
            self._backbone = backbone
        else:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv", "proj", "fc1", "fc2"],
                lora_dropout=0.1,
                bias="none",
            )
            self._backbone = LoraModel(backbone, lora_cfg, adapter_name="default")

        # One Cropr module per block except the last; first module removes one extra
        # token to match the original paper schedule.
        n_blocks = adapter.n_blocks
        num_cropr = n_blocks - 1
        schedule = [pruning_rate] * num_cropr
        schedule[0] += 1

        num_tokens = adapter.n_patches + adapter.num_prefix_tokens
        remaining = [num_tokens - sum(schedule[: i + 1]) for i in range(num_cropr)]
        print(f"[CroprClassifier] pruning schedule — tokens remaining per step: {remaining}")

        self.cropr_modules = nn.ModuleList([
            Cropr(
                pruning_rate=schedule[i],
                num_queries=num_queries,
                num_classes=n_classes,
                embed_dim=adapter.embed_dim,
                num_heads=cropr_num_heads,
                pre_attn_norm=pre_attn_norm,
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                mlp=mlp,
                mlp_ratio=mlp_ratio,
                training=True,
            )
            for i in range(num_cropr)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def raw_backbone(self) -> nn.Module:
        """Underlying timm ViT, unwrapped from peft when LoRA is active."""
        if isinstance(self._backbone, LoraModel):
            return self._backbone.model
        return self._backbone

    @property
    def trainable_backbone_params(self) -> List[nn.Parameter]:
        if self._freeze_backbone:
            return []
        return [p for p in self._backbone.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        backbone = self.raw_backbone

        # Embedding — skip grad tracking when backbone is frozen
        with torch.set_grad_enabled(not self._freeze_backbone):
            x = backbone.patch_embed(x)
            x = backbone._pos_embed(x)
            x = backbone.patch_drop(x)
            x = backbone.norm_pre(x)

        # Transformer blocks with progressive CropR pruning
        aux_preds = []
        for i, cropr in enumerate(self.cropr_modules):
            with torch.set_grad_enabled(not self._freeze_backbone):
                x = backbone.blocks[i](x)
            x, _, pred = cropr(x, inference=not self.training)
            if pred is not None:
                aux_preds.append(pred)

        # Final block — no pruning
        with torch.set_grad_enabled(not self._freeze_backbone):
            x = backbone.blocks[-1](x)
            x = backbone.norm(x)

        # Pool over spatial patches (exclude prefix tokens)
        num_prefix = backbone.num_prefix_tokens
        if backbone.global_pool == "avg":
            pooled = x[:, num_prefix:].mean(dim=1)
        else:
            pooled = x[:, 0]
        pooled = backbone.fc_norm(pooled)

        final_logits = self.head(pooled)

        if self.training and aux_preds:
            return [final_logits] + aux_preds
        return final_logits
