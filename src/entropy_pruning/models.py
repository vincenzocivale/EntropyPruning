from __future__ import annotations

import types

import timm
import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel


class AttentionForecaster(nn.Module):
    def __init__(self, embed_dim=1024, hidden=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.cls_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        self.self_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=n_heads,
                    dim_feedforward=hidden * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True) for _ in range(n_layers)]
        )
        self.cross_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, patch_embeddings):
        bsz, n_tokens, _ = patch_embeddings.shape
        x = self.input_proj(patch_embeddings)
        for sa in self.self_attn:
            x = sa(x)

        cls = self.cls_query.expand(bsz, -1, -1)
        for ca, norm in zip(self.cross_attn, self.cross_norms):
            cls_out, _ = ca(cls, x, x)
            cls = norm(cls + cls_out)

        x_norm = self.norm(x)
        cls_exp = cls.expand(-1, n_tokens, -1)
        scores = self.score_head(torch.cat([x_norm, cls_exp], dim=-1)).squeeze(-1)
        return scores.softmax(-1)


class UNILoRAClassifier(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.1):
        super().__init__()
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(dropout), nn.Linear(1024, n_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


class UNILoRAWithForecasterPruning(nn.Module):
    def __init__(
        self,
        n_classes: int,
        forecaster: AttentionForecaster,
        prune_layer: int,
        keep_ratio: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(dropout), nn.Linear(1024, n_classes))
        self.forecaster = forecaster
        self.prune_layer = prune_layer
        self.keep_ratio = keep_ratio

    def forward(self, x):
        block = self.backbone.model.blocks[self.prune_layer]
        orig_fwd = block.forward
        block.forward = types.MethodType(self._make_block_hook(orig_fwd), block)
        try:
            out = self.head(self.backbone(x))
        finally:
            block.forward = orig_fwd
        return out

    def _make_block_hook(self, orig_fwd):
        training = self.training
        forecaster = self.forecaster
        keep_ratio = self.keep_ratio

        def block_fwd(block_self, x):
            x = orig_fwd(x)
            bsz, n_tokens, _ = x.shape
            patch_emb = x[:, 1:]
            with torch.no_grad():
                scores = forecaster(patch_emb)

            k_keep = max(1, int((n_tokens - 1) * keep_ratio))
            topk = scores.topk(k_keep, dim=-1)
            topk_idx = topk.indices

            cls_tok = x[:, :1, :]
            patches = x[:, 1:, :]

            if training:
                threshold = topk.values[:, -1:]
                soft_mask = torch.sigmoid((scores - threshold) / 0.05)
                hard_mask = (scores >= threshold).float()
                st_mask = hard_mask - soft_mask.detach() + soft_mask
                patches = patches * st_mask.unsqueeze(-1)

            kept = torch.stack([patches[b][topk_idx[b]] for b in range(bsz)], dim=0)
            return torch.cat([cls_tok, kept], dim=1)

        return block_fwd
