from typing import Any

import torch.nn as nn


class ThunderBackboneAdapter:
    """
    Wraps a timm-based backbone from Thunder's get_model_from_name and exposes
    uniform attributes and accessors for EAF's 3-phase pipeline.

    Supports timm-based ViT models (standard Block with Attention), including
    models exposed directly and wrappers used by THUNDER. In particular,
    CONCH v1.5 is registered as ``titan`` and exposes its ViT as ``model.trunk``;
    CONCH v1 exposes the timm trunk as ``model.visual.trunk``.

    Raises NotImplementedError for backbones whose transformer cannot be
    resolved to a standard timm VisionTransformer.

    Args:
        model: raw backbone from get_model_from_name (first element of the tuple).
        transform: optional preprocessing transform, the second element of
            get_model_from_name's return tuple. ``patch_embed.num_patches`` is a
            build-time value baked in from the model's *configured* img_size
            (e.g. CONCH v1.5/titan is configured at 224, giving a stale 196) and
            is silently wrong whenever the model's own transform actually runs
            at a different resolution (CONCH v1.5 resizes to 448, a real 28x28=784
            grid via its Conv2d patch_embed, which has no fixed spatial output size).
            When ``transform`` is given, ``n_patches`` is recomputed from the
            transform's real crop size and the patch_embed's true patch size
            instead of trusting the stale config value.

    Attributes:
        embed_dim (int):          Token embedding dimension.
        n_blocks (int):           Total number of transformer blocks.
        n_patches (int):          Spatial patch count (excludes CLS + register tokens).
        num_prefix_tokens (int):  Number of CLS + register tokens prepended to the sequence.
    """

    def __init__(self, model: nn.Module, transform: Any = None):
        core_model = self._resolve_timm(model)
        if core_model is None:
            raise NotImplementedError(
                f"ThunderBackboneAdapter: '{type(model).__name__}' does not expose "
                "a supported timm VisionTransformer at the model root, .trunk, "
                "or .visual.trunk. HuggingFace-native models such as phikon and "
                "hibou are not yet supported."
            )
        self.wrapper = model
        self.model = core_model
        self.embed_dim: int = core_model.embed_dim
        self.n_blocks: int = len(core_model.blocks)
        self.num_prefix_tokens: int = core_model.num_prefix_tokens
        self.n_patches: int = core_model.patch_embed.num_patches
        self.input_size: int | None = None
        if transform is not None:
            crop_size = self._infer_crop_size(transform)
            self.input_size = crop_size
            patch_size = self._infer_patch_size(core_model)
            if crop_size is not None and patch_size is not None and patch_size > 0:
                real_side = crop_size // patch_size
                self.n_patches = real_side * real_side

    @staticmethod
    def _infer_crop_size(transform: Any) -> int | None:
        """Best-effort read of a torchvision Compose's Resize/CenterCrop target size."""
        from torchvision import transforms as T

        steps = getattr(transform, "transforms", None)
        if steps is None:
            return None
        for step in reversed(steps):
            if isinstance(step, (T.CenterCrop, T.Resize)):
                size = step.size
                if isinstance(size, (tuple, list)):
                    return int(size[0])
                if isinstance(size, int):
                    return size
        return None

    @staticmethod
    def _infer_patch_size(core_model: nn.Module) -> int | None:
        patch_size = getattr(core_model.patch_embed, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            return int(patch_size[0])
        if isinstance(patch_size, int):
            return patch_size
        return None

    @staticmethod
    def _detect_timm(model: nn.Module) -> bool:
        return (
            hasattr(model, "blocks")
            and hasattr(model, "embed_dim")
            and hasattr(model, "patch_embed")
            and hasattr(model.patch_embed, "num_patches")
            and hasattr(model, "num_prefix_tokens")
        )

    @classmethod
    def _resolve_timm(cls, model: nn.Module) -> nn.Module | None:
        candidates = [model, getattr(model, "trunk", None)]
        visual = getattr(model, "visual", None)
        if visual is not None:
            candidates.append(getattr(visual, "trunk", None))
        for candidate in candidates:
            if isinstance(candidate, nn.Module) and cls._detect_timm(candidate):
                return candidate
        return None

    def get_blocks(self):
        """Returns the list of transformer blocks (timm: model.blocks)."""
        return self.model.blocks

    def get_attn_module(self, block_idx: int) -> nn.Module:
        """Returns the attention module at block_idx. Validates standard timm layout."""
        block = self.model.blocks[block_idx]
        if not hasattr(block, "attn"):
            raise ValueError(
                f"Block {block_idx} ({type(block).__name__}) has no .attn attribute. "
                "Only standard timm Block with Attention is supported."
            )
        attn = block.attn
        if not hasattr(attn, "qkv"):
            raise ValueError(
                f"Attention at block {block_idx} ({type(attn).__name__}) has no .qkv. "
                "Expected standard timm Attention."
            )
        return attn
