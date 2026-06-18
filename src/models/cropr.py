"""Token Cropr-style pruning modules for EAF.

This is a compact EAF integration of the Cropr idea from
https://github.com/benbergner/cropr, not a vendored copy of the upstream
repository.  Cropr learns lightweight auxiliary cross-attention heads that rank
patch tokens and prune the lowest-scoring ones while training the main task.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel

from .backbone_adapter import ThunderBackboneAdapter


class CroprScorer(nn.Module):
    """Auxiliary cross-attention scorer used to rank patch tokens."""

    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        num_queries: int = 1,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        pre_attn_norm: bool = False,
        q_proj: bool = False,
        k_proj: bool = False,
        v_proj: bool = False,
        mlp: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        self.attn_norm = nn.LayerNorm(embed_dim) if pre_attn_norm else nn.Identity()
        self.q = nn.Linear(embed_dim, embed_dim, bias=False) if q_proj else nn.Identity()
        self.k = nn.Linear(embed_dim, embed_dim, bias=False) if k_proj else nn.Identity()
        self.v = nn.Linear(embed_dim, embed_dim, bias=False) if v_proj else nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim) if num_heads > 1 else nn.Identity()
        self.mlp = (
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            )
            if mlp
            else None
        )
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, n_classes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.queries, std=self.embed_dim ** -0.5)
        linear = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for module in linear:
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _attention(self, patches: torch.Tensor):
        B, N, _ = patches.shape
        x = self.attn_norm(patches)
        q_in = self.queries.expand(B, -1, -1)
        q = self.q(q_in).reshape(B, self.num_queries, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        return q @ k.transpose(-2, -1)

    def score(self, patches: torch.Tensor) -> torch.Tensor:
        attn = self._attention(patches)
        return attn.sum((1, 2))

    def forward(self, patches: torch.Tensor):
        B, N, _ = patches.shape
        x = self.attn_norm(patches)
        q_in = self.queries.expand(B, -1, -1)
        q = self.q(q_in).reshape(B, self.num_queries, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = q @ k.transpose(-2, -1)
        weights = (attn * self.scale).softmax(dim=-1)
        pooled = weights @ v
        pooled = pooled.transpose(1, 2).reshape(B, self.num_queries, self.embed_dim)
        pooled = self.proj(pooled).mean(dim=1)
        if self.mlp is not None:
            pooled = pooled + self.mlp(pooled)
        scores = attn.sum((1, 2))
        return scores, self.head(pooled)


def cropr_pruning_schedule(
    n_patches: int,
    n_modules: int,
    pruning_rate: int | None = None,
    keep_ratio: float | None = None,
) -> List[int]:
    """Return Cropr's constant token-removal schedule.

    Cropr removes a fixed number of tokens after each Cropr module.  ``keep_ratio``
    is accepted only as an EAF convenience for deriving a constant rate when the
    caller does not provide Cropr's native ``pruning_rate``.
    """
    if n_modules < 1:
        return []
    if pruning_rate is None:
        if keep_ratio is None:
            raise ValueError("Either pruning_rate or keep_ratio must be provided")
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        final_keep = max(1, int(n_patches * keep_ratio))
        total_remove = max(0, n_patches - final_keep)
        pruning_rate = max(0, math.ceil(total_remove / n_modules))
    if pruning_rate < 0:
        raise ValueError("pruning_rate must be >= 0")
    return [int(pruning_rate)] * n_modules


def cropr_pruning_layers(n_blocks: int, llf: bool = True) -> List[int]:
    """Transformer block indices after which Cropr modules are applied."""
    last_pruned_block = n_blocks - 3 if llf else n_blocks - 2
    if last_pruned_block < 0:
        return []
    return list(range(last_pruned_block + 1))


class LoRAWithCroprPruning(nn.Module):
    """LoRA classifier with progressive Cropr token pruning.

    Cropr modules are inserted progressively after ViT blocks: through the
    second-to-last block without LLF, or through the third-to-last block with LLF.
    With ``llf=True``, pruned patch tokens are concatenated back before the final
    block, matching Cropr's last-layer fusion idea.  During training the model
    returns ``[main_logits, aux_1, ...]`` so the caller can apply the same
    classification loss to all outputs.  During eval it returns only the main
    logits.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ThunderBackboneAdapter,
        n_classes: int,
        keep_ratio: float,
        pruning_rate: int | None = None,
        llf: bool = True,
        num_queries: int = 1,
        num_heads: int = 1,
        pre_attn_norm: bool = False,
        q_proj: bool = False,
        k_proj: bool = False,
        v_proj: bool = False,
        mlp: bool = True,
        mlp_ratio: float = 4.0,
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = adapter
        self.keep_ratio = keep_ratio
        self.pruning_rate = pruning_rate
        self.llf = llf
        self.num_prefix = adapter.num_prefix_tokens

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=cropr_lora_targets(adapter),
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )

        self.prune_layers = cropr_pruning_layers(adapter.n_blocks, llf=llf)
        schedule = cropr_pruning_schedule(
            adapter.n_patches,
            len(self.prune_layers),
            pruning_rate=pruning_rate,
            keep_ratio=keep_ratio,
        )
        self.cropr_schedule = schedule
        self.cropr = nn.ModuleList(
            [
                CroprScorer(
                    embed_dim=adapter.embed_dim,
                    n_classes=n_classes,
                    num_queries=num_queries,
                    num_heads=num_heads,
                    pre_attn_norm=pre_attn_norm,
                    q_proj=q_proj,
                    k_proj=k_proj,
                    v_proj=v_proj,
                    mlp=mlp,
                    mlp_ratio=mlp_ratio,
                )
                for _ in schedule
            ]
        )

    @property
    def raw_backbone(self) -> nn.Module:
        return self.backbone.model

    def forward(self, x: torch.Tensor):
        aux_logits: List[torch.Tensor] = []
        pruned_tokens: List[torch.Tensor] = []
        handles = []

        for module_idx, block_idx in enumerate(self.prune_layers):
            scorer = self.cropr[module_idx]
            pruning_rate = self.cropr_schedule[module_idx]
            handles.append(
                self.raw_backbone.blocks[block_idx].register_forward_hook(
                    self._make_prune_hook(scorer, pruning_rate, aux_logits, pruned_tokens)
                )
            )

        if self.llf and pruned_tokens:
            raise RuntimeError("Unexpected stale Cropr pruned token cache.")
        if self.llf:
            last_block = self.raw_backbone.blocks[-1]
            handles.append(last_block.register_forward_pre_hook(self._make_llf_hook(pruned_tokens)))

        try:
            features = self.backbone(x)
        finally:
            for handle in handles:
                handle.remove()

        if features.ndim == 3:
            features = features[:, 0]
        logits = self.head(features)
        if self.training and aux_logits:
            return [logits] + aux_logits
        return logits

    def _make_prune_hook(
        self,
        scorer: CroprScorer,
        pruning_rate: int,
        aux_logits: List[torch.Tensor],
        pruned_tokens: List[torch.Tensor],
    ):
        num_prefix = self.num_prefix

        def _hook(module, inputs, output):
            prefix = output[:, :num_prefix]
            patches = output[:, num_prefix:]
            if patches.shape[1] <= 1 or pruning_rate <= 0:
                if self.training:
                    _, pred = scorer(patches.detach())
                    aux_logits.append(pred)
                return output

            if self.training:
                scores, pred = scorer(patches.detach())
                aux_logits.append(pred)
            else:
                with torch.no_grad():
                    scores = scorer.score(patches)

            k_keep = max(1, patches.shape[1] - pruning_rate)
            k_keep = min(k_keep, patches.shape[1])
            idx = torch.argsort(scores, dim=1, descending=True)
            keep_idx = idx[:, :k_keep]
            drop_idx = idx[:, k_keep:]
            kept = patches.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
            if self.llf and drop_idx.shape[1] > 0:
                dropped = patches.gather(
                    1, drop_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
                )
                pruned_tokens.append(dropped)
            return torch.cat([prefix, kept], dim=1)

        return _hook

    @staticmethod
    def _make_llf_hook(pruned_tokens: Sequence[torch.Tensor]):
        def _hook(module, inputs):
            if not pruned_tokens:
                return inputs
            x = inputs[0]
            return (torch.cat([x] + list(pruned_tokens), dim=1),) + tuple(inputs[1:])

        return _hook


def cropr_lora_targets(adapter: ThunderBackboneAdapter) -> list:
    """LoRA targets for blocks affected by Cropr pruning."""
    targets = []
    for i in range(1, adapter.n_blocks):
        targets += [
            f"blocks.{i}.attn.qkv",
            f"blocks.{i}.attn.proj",
            f"blocks.{i}.mlp.fc1",
            f"blocks.{i}.mlp.fc2",
        ]
    return targets
