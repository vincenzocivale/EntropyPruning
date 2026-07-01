import torch.nn as nn


class ThunderBackboneAdapter:
    """
    Wraps a timm-based backbone from Thunder's get_model_from_name and exposes
    uniform attributes and accessors for EAF's 3-phase pipeline.

    Supports timm-based ViT models (standard Block with Attention):
      uni, uni2h, hoptimus0, hoptimus1, virchow, virchow2, h0mini,
      kaiko_vit*, dinov2base, dinov2large.
    Raises NotImplementedError for HuggingFace-based models (phikon, hibou).

    Args:
        model: raw backbone from get_model_from_name (first element of the tuple).

    Attributes:
        embed_dim (int):          Token embedding dimension.
        n_blocks (int):           Total number of transformer blocks.
        n_patches (int):          Spatial patch count (excludes CLS + register tokens).
        num_prefix_tokens (int):  Number of CLS + register tokens prepended to the sequence.
    """

    def __init__(self, model: nn.Module):
        if not self._detect_timm(model):
            raise NotImplementedError(
                f"ThunderBackboneAdapter: '{type(model).__name__}' is not a supported "
                "timm VisionTransformer. Supported: uni, uni2h, hoptimus0/1, "
                "virchow, virchow2, h0mini, kaiko_vit*, dinov2base, dinov2large. "
                "HuggingFace-based models (phikon, hibou) are not yet supported."
            )
        self.model = model
        self.embed_dim: int = model.embed_dim
        self.n_blocks: int = len(model.blocks)
        self.num_prefix_tokens: int = model.num_prefix_tokens
        self.n_patches: int = model.patch_embed.num_patches

    @staticmethod
    def _detect_timm(model: nn.Module) -> bool:
        return (
            hasattr(model, "blocks")
            and hasattr(model, "embed_dim")
            and hasattr(model, "patch_embed")
            and hasattr(model.patch_embed, "num_patches")
            and hasattr(model, "num_prefix_tokens")
        )

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
