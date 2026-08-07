"""Online attention distillation utilities for tile-level EAF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class FrozenTimmAttentionTeacher(nn.Module):
    """Extract source tokens and final CLS-to-patch attention online.

    The wrapped tile encoder is always frozen and kept in evaluation mode. The
    class registers lightweight pre-forward hooks on two timm attention modules
    and recomputes only the target layer's QK attention after the normal forward.
    No tile embeddings or attention maps are materialized on disk.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter,
        source_layer: int,
        target_layer: int,
        target_normalization: str = "patch",
    ) -> None:
        super().__init__()
        if source_layer < 0 or source_layer >= adapter.n_blocks:
            raise ValueError(f"source_layer={source_layer} outside encoder blocks")
        if target_layer < 0 or target_layer >= adapter.n_blocks:
            raise ValueError(f"target_layer={target_layer} outside encoder blocks")
        if target_normalization not in {"patch", "none"}:
            raise ValueError("target_normalization must be 'patch' or 'none'")

        self.backbone = backbone
        self.adapter = adapter
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.target_normalization = target_normalization
        self.num_prefix_tokens = int(adapter.num_prefix_tokens)
        self._source_tokens: torch.Tensor | None = None
        self._target_tokens: torch.Tensor | None = None

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        source_attn = adapter.get_attn_module(source_layer)
        target_attn = adapter.get_attn_module(target_layer)
        self._target_attn = target_attn
        self._handles = [
            source_attn.register_forward_pre_hook(self._capture_source),
            target_attn.register_forward_pre_hook(self._capture_target),
        ]

    def train(self, mode: bool = True):
        """Keep the teacher deterministic even when the trainer enters train mode."""
        super().train(False)
        self.backbone.eval()
        return self

    def _capture_source(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        self._source_tokens = inputs[0]

    def _capture_target(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        self._target_tokens = inputs[0]

    def _cls_patch_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_module = self._target_attn
        batch, n_tokens, channels = tokens.shape
        num_heads = int(attn_module.num_heads)
        head_dim = int(getattr(attn_module, "head_dim", channels // num_heads))
        qkv = attn_module.qkv(tokens).reshape(
            batch, n_tokens, 3, num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        q_norm = getattr(attn_module, "q_norm", nn.Identity())
        k_norm = getattr(attn_module, "k_norm", nn.Identity())
        q, k = q_norm(q), k_norm(k)
        scale = float(getattr(attn_module, "scale", head_dim ** -0.5))

        cls_scores = q[:, :, :1] @ k.transpose(-2, -1)
        cls_attention = (cls_scores * scale).softmax(dim=-1)
        patch_attention = cls_attention[
            :, :, 0, self.num_prefix_tokens:
        ].mean(dim=1)
        if self.target_normalization == "patch":
            patch_attention = patch_attention / patch_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        return patch_attention.float()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._source_tokens = None
        self._target_tokens = None
        self.backbone.eval()
        self.backbone(images)
        if self._source_tokens is None or self._target_tokens is None:
            raise RuntimeError("Failed to capture timm attention inputs")

        source = self._source_tokens[:, self.num_prefix_tokens:].detach()
        target = self._cls_patch_attention(self._target_tokens).detach()
        if source.shape[:2] != target.shape:
            raise RuntimeError(
                "Source patch tokens and target attention disagree: "
                f"source={tuple(source.shape)}, target={tuple(target.shape)}"
            )
        return source, target

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def spearman_correlation(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-sample Spearman correlation without scipy."""
    predicted_rank = predicted.argsort(dim=-1).argsort(dim=-1).float()
    target_rank = target.argsort(dim=-1).argsort(dim=-1).float()
    predicted_rank -= predicted_rank.mean(dim=-1, keepdim=True)
    target_rank -= target_rank.mean(dim=-1, keepdim=True)
    numerator = (predicted_rank * target_rank).sum(dim=-1)
    denominator = torch.sqrt(
        predicted_rank.square().sum(dim=-1)
        * target_rank.square().sum(dim=-1)
    )
    return numerator / denominator.clamp_min(1e-8)


def topk_recall(
    predicted: torch.Tensor, target: torch.Tensor, fraction: float
) -> torch.Tensor:
    """Fraction of teacher top-k patches recovered by the prediction."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, int(round(predicted.shape[-1] * fraction)))
    predicted_idx = predicted.topk(k, dim=-1).indices
    target_idx = target.topk(k, dim=-1).indices
    matches = (
        predicted_idx.unsqueeze(-1) == target_idx.unsqueeze(-2)
    ).any(dim=-1)
    return matches.float().mean(dim=-1)


def load_forecaster_checkpoint(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Load raw or metadata-wrapped EAF checkpoints."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in ("forecaster_state_dict", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                metadata = {k: v for k, v in payload.items() if k != key}
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}")
    state_dict = {
        key.removeprefix("module."): value for key, value in payload.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return metadata
