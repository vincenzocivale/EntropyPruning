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

    Attributes:
        embed_dim (int):          Token embedding dimension.
        n_blocks (int):           Total number of transformer blocks.
        n_patches (int):          Spatial patch count (excludes CLS + register tokens).
        num_prefix_tokens (int):  Number of CLS + register tokens prepended to the sequence.
    """

    def __init__(self, model: nn.Module):
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
