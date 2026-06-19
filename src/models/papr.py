"""PaPr-style training-free patch pruning for timm ViT backbones.

This module ports the small part of PaPr needed by EAF: use a frozen
pretrained ConvNet to produce a patch significance map, prune low-scoring
spatial tokens immediately after ViT patch/position embedding, and run the
existing classifier without retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_adapter import ThunderBackboneAdapter


GridSize = Tuple[int, int]


def _infer_grid_size(n_patches: int, grid_size: Optional[GridSize] = None) -> GridSize:
    if grid_size is not None:
        return int(grid_size[0]), int(grid_size[1])
    side = int(n_patches ** 0.5)
    if side * side != n_patches:
        raise ValueError(
            f"Cannot infer a square patch grid from {n_patches} patches. "
            "Pass grid_size explicitly for non-square inputs."
        )
    return side, side


def papr_scores_from_features(
    conv_features: torch.Tensor,
    grid_size: GridSize,
    align_corners: bool = True,
) -> torch.Tensor:
    """Return PaPr patch scores from ConvNet feature maps.

    PaPr averages the proposal ConvNet feature map over channels and upsamples
    it to the ViT token grid. The returned tensor has shape ``(B, H * W)``.
    """
    if conv_features.ndim != 4:
        raise ValueError(
            f"Expected proposal features with shape (B, C, H, W), got "
            f"{tuple(conv_features.shape)}"
        )
    discriminative = conv_features.mean(dim=1, keepdim=True)
    scores = F.interpolate(
        discriminative,
        size=grid_size,
        mode="bicubic",
        align_corners=align_corners,
    )
    return scores.flatten(1)


def apply_papr_to_tokens(
    tokens: torch.Tensor,
    conv_features: torch.Tensor,
    keep_ratio: float,
    num_prefix_tokens: int = 1,
    grid_size: Optional[GridSize] = None,
    align_corners: bool = True,
    return_indices: bool = False,
):
    """Prune ViT tokens using the PaPr proposal map.

    Prefix tokens are always preserved. ``keep_ratio`` is applied to spatial
    patch tokens, which keeps the behavior well-defined for Thunder backbones
    with CLS plus register tokens.
    """
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, N, D), got {tokens.shape}")

    batch, n_tokens, dim = tokens.shape
    if not 0 <= num_prefix_tokens < n_tokens:
        raise ValueError(
            f"num_prefix_tokens={num_prefix_tokens} is incompatible with "
            f"{n_tokens} tokens"
        )

    n_patches = n_tokens - num_prefix_tokens
    grid_size = _infer_grid_size(n_patches, grid_size)
    if grid_size[0] * grid_size[1] != n_patches:
        raise ValueError(
            f"grid_size={grid_size} covers {grid_size[0] * grid_size[1]} patches, "
            f"but tokens contain {n_patches} spatial patches"
        )

    scores = papr_scores_from_features(conv_features, grid_size, align_corners)
    if scores.shape != (batch, n_patches):
        raise ValueError(
            f"Proposal scores have shape {tuple(scores.shape)}, expected "
            f"{(batch, n_patches)}"
        )

    k_keep = max(1, int(n_patches * keep_ratio))
    patch_indices = scores.argsort(dim=1, descending=True)[:, :k_keep]
    prefix = tokens[:, :num_prefix_tokens]
    patches = tokens[:, num_prefix_tokens:]
    kept = patches.gather(1, patch_indices.unsqueeze(-1).expand(-1, -1, dim))
    pruned = torch.cat([prefix, kept], dim=1)
    if return_indices:
        return pruned, patch_indices
    return pruned


class TorchvisionResNetFeatures(nn.Module):
    """Feature extractor matching PaPr's ResNet proposal path."""

    _CHANNELS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }

    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        import torchvision.models as tv_models

        if name not in self._CHANNELS:
            raise ValueError(
                f"Unsupported torchvision ResNet proposal '{name}'. "
                f"Choose from {sorted(self._CHANNELS)}."
            )

        weights = None
        if pretrained and weights_path is None:
            enum_name = {
                "resnet18": "ResNet18_Weights",
                "resnet34": "ResNet34_Weights",
                "resnet50": "ResNet50_Weights",
                "resnet101": "ResNet101_Weights",
                "resnet152": "ResNet152_Weights",
            }[name]
            weights = getattr(tv_models, enum_name).DEFAULT

        model = getattr(tv_models, name)(weights=weights)
        if weights_path is not None:
            _load_checkpoint_state(model, weights_path)

        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )
        self.out_channels = self._CHANNELS[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class TimmFeatureProposal(nn.Module):
    """Feature extractor for timm ConvNets, including MobileOne when available."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        import timm

        self.model = timm.create_model(
            name,
            pretrained=(pretrained and weights_path is None),
            features_only=True,
            out_indices=(-1,),
        )
        if weights_path is not None:
            _load_checkpoint_state(self.model, weights_path)
        channels = self.model.feature_info.channels()
        self.out_channels = int(channels[-1]) if channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return features[-1] if isinstance(features, (list, tuple)) else features


def _load_checkpoint_state(model: nn.Module, weights_path: str) -> None:
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {weights_path}")
    state = {
        k.removeprefix("module."): v
        for k, v in checkpoint.items()
        if torch.is_tensor(v)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and unexpected:
        raise RuntimeError(
            f"Could not load proposal weights from {weights_path}: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def build_papr_proposal(
    proposal_model: str = "mobileone_s0",
    pretrained: bool = True,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """Build a frozen proposal ConvNet for PaPr.

    ``mobileone_s0`` is the PaPr ViT default. ``resnet*`` names use torchvision
    and match the heavier proposal choices exposed by the PaPr scripts. Other
    names are delegated to timm with ``features_only=True``.

    Note that the original PaPr repository loads Apple MobileOne checkpoints
    with ``stage0/stage1/...`` keys. timm's MobileOne uses a different module
    layout, so local ``--proposal-weights`` for MobileOne must be compatible
    with the local timm model.
    """
    if weights_path is not None and not Path(weights_path).exists():
        raise FileNotFoundError(f"Proposal weights not found: {weights_path}")

    if proposal_model in TorchvisionResNetFeatures._CHANNELS:
        proposal = TorchvisionResNetFeatures(proposal_model, pretrained, weights_path)
    else:
        proposal = TimmFeatureProposal(proposal_model, pretrained, weights_path)

    proposal.eval()
    for param in proposal.parameters():
        param.requires_grad_(False)
    return proposal


class PaPrPrunedClassifier(nn.Module):
    """Inference-only PaPr wrapper around an EAF Phase 1 classifier."""

    def __init__(
        self,
        classifier: nn.Module,
        adapter: ThunderBackboneAdapter,
        proposal: nn.Module,
        keep_ratio: float,
        align_corners: bool = True,
    ):
        super().__init__()
        if not hasattr(classifier, "raw_backbone") or not hasattr(classifier, "head"):
            raise TypeError("classifier must expose raw_backbone and head attributes")
        self.classifier = classifier
        self.adapter = adapter
        self.proposal = proposal
        self.keep_ratio = keep_ratio
        self.align_corners = align_corners
        self.num_prefix_tokens = adapter.num_prefix_tokens
        self.grid_size = self._adapter_grid_size(adapter)

        for param in self.proposal.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _adapter_grid_size(adapter: ThunderBackboneAdapter) -> GridSize:
        patch_embed = adapter.model.patch_embed
        grid_size = getattr(patch_embed, "grid_size", None)
        if grid_size is not None:
            if isinstance(grid_size, int):
                return grid_size, grid_size
            return int(grid_size[0]), int(grid_size[1])
        return _infer_grid_size(adapter.n_patches)

    @property
    def raw_backbone(self) -> nn.Module:
        return self.classifier.raw_backbone

    @property
    def head(self) -> nn.Module:
        return self.classifier.head

    @torch.no_grad()
    def extract_conv_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.proposal(images)

    def forward_features(self, images: torch.Tensor, return_tokens: bool = False):
        raw = self.raw_backbone
        proposal_features = self.extract_conv_features(images)
        x = raw.patch_embed(images)

        if hasattr(raw, "_pos_embed"):
            x = raw._pos_embed(x)
        else:
            cls_token = getattr(raw, "cls_token", None)
            if cls_token is not None:
                cls = cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls, x), dim=1)
            pos_embed = getattr(raw, "pos_embed", None)
            if pos_embed is not None:
                x = x + pos_embed
            pos_drop = getattr(raw, "pos_drop", None)
            if pos_drop is not None:
                x = pos_drop(x)

        patch_drop = getattr(raw, "patch_drop", None)
        if patch_drop is not None:
            x = patch_drop(x)
        norm_pre = getattr(raw, "norm_pre", None)
        if norm_pre is not None:
            x = norm_pre(x)

        x, kept_indices = apply_papr_to_tokens(
            x,
            proposal_features,
            keep_ratio=self.keep_ratio,
            num_prefix_tokens=self.num_prefix_tokens,
            grid_size=self.grid_size,
            align_corners=self.align_corners,
            return_indices=True,
        )

        for block in raw.blocks:
            x = block(x)

        norm = getattr(raw, "norm", None)
        if norm is not None:
            x = norm(x)

        if not return_tokens:
            return x
        return {"features": x, "kept_indices": kept_indices}

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.raw_backbone
        forward_head = getattr(raw, "forward_head", None)
        if forward_head is not None:
            try:
                pooled = forward_head(features, pre_logits=True)
            except TypeError:
                pooled = forward_head(features)
        elif features.ndim == 3:
            pooled = features[:, 0]
        else:
            pooled = features

        if pooled.ndim == 3:
            pooled = pooled[:, 0]
        return pooled

    def forward(self, images: torch.Tensor, return_tokens: bool = False):
        features = self.forward_features(images, return_tokens=return_tokens)
        if return_tokens:
            token_info = features
            features = token_info["features"]
        pooled = self.pool_features(features)
        logits = self.head(pooled)
        if return_tokens:
            token_info["logits"] = logits
            return token_info
        return logits
