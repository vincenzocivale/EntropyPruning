import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel

from .backbone_adapter import BackboneAdapter


class GenericLoRAWithForecasterPruning(nn.Module):
    """
    Forecaster-guided token pruning model for TRIDENT backbones.

    Runs the full backbone up to prune_layer, scores spatial patch tokens with a
    frozen AttentionForecaster, keeps the top keep_ratio fraction, then runs the
    remaining blocks. Prefix tokens (CLS + register tokens) are always preserved.

    Pruning is implemented via a persistent forward hook registered in __init__.

    Args:
        backbone:    raw timm backbone model.
        adapter:     BackboneAdapter for backbone.
        n_classes:   number of output classes.
        forecaster:  trained AttentionForecaster (must be frozen before passing in).
        prune_layer: block index where pruning is applied (0-indexed).
        keep_ratio:  fraction of spatial patch tokens to keep (e.g. 0.1 = top 10%).
        lora_r, lora_alpha: LoRA parameters.
        dropout:     classifier head dropout.

    Note:
        Load Phase 1 checkpoint with strict=False — peft key prefix differs from
        a plain model, and the forecaster keys are new.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: BackboneAdapter,
        n_classes: int,
        forecaster: nn.Module,
        prune_layer: int,
        keep_ratio: float,
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = adapter
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )
        self.forecaster = forecaster
        self.prune_layer = prune_layer
        self.keep_ratio = keep_ratio

        # Register persistent forward hook on the prune layer
        self._register_pruning_hook()

    @property
    def raw_backbone(self) -> nn.Module:
        """The underlying timm VisionTransformer (unwrapped from peft)."""
        return self.backbone.model

    def _register_pruning_hook(self):
        """Register a forward hook on the prune layer block to apply pruning."""
        prune_block = self.raw_backbone.blocks[self.prune_layer]

        def pruning_hook(module, input, output):
            """
            Apply forecaster-guided token pruning to the block output.

            Args:
                module: the block (TransformerBlock)
                input: unused
                output: (B, N_total, D) tensor from the block forward

            Returns:
                pruned_output: (B, N_kept, D) with CLS/prefix tokens always kept
            """
            x = output
            B, N_total, D = x.shape
            num_prefix = self.adapter.num_prefix_tokens

            # Split prefix and patches
            prefix = x[:, :num_prefix, :]  # CLS + register tokens — always kept
            patches = x[:, num_prefix:, :]  # spatial patches — subject to pruning
            N_patches = patches.shape[1]

            # Score patches with frozen forecaster
            with torch.no_grad():
                scores = self.forecaster(patches)  # (B, N_patches)

            # Select top-k patches
            k_keep = max(1, int(N_patches * self.keep_ratio))
            topk_vals, topk_idx = scores.topk(k_keep, dim=-1)

            # Straight-through estimator during training for gradient flow
            if self.training:
                # Soft mask via sigmoid, hard mask via threshold
                threshold = topk_vals[:, -1:]  # (B, 1)
                soft_mask = torch.sigmoid((scores - threshold) / 0.05)  # (B, N_patches)
                hard_mask = (scores >= threshold).float()  # (B, N_patches)
                # Straight-through: use hard forward, soft gradient
                st_mask = hard_mask - soft_mask.detach() + soft_mask
                masked_patches = patches * st_mask.unsqueeze(-1)
                kept = torch.stack(
                    [masked_patches[b][topk_idx[b]] for b in range(B)], dim=0
                )
            else:
                # Inference: hard selection only
                kept = torch.stack(
                    [patches[b][topk_idx[b]] for b in range(B)], dim=0
                )

            # Concatenate prefix (always kept) with pruned patches
            return torch.cat([prefix, kept], dim=1)

        prune_block.register_forward_hook(pruning_hook)

    def forward(self, x):
        """
        Forward pass with pruning applied via registered hook.

        Args:
            x: input tensor (B, 3, H, W)

        Returns:
            logits: (B, n_classes)
        """
        # Backbone forward (with pruning applied by hook at prune_layer).
        # Use forward_features to get the full token sequence; the top-level
        # forward() would apply pooling/head and return (B, D) instead.
        features = self.raw_backbone.forward_features(x)  # (B, N_kept, D)

        # Extract CLS token and pass through classification head
        cls_token = features[:, 0, :]  # (B, D)
        logits = self.head(cls_token)  # (B, n_classes)

        return logits

    def get_cls_embedding(self, x):
        """
        Get CLS embedding without classification head.

        Useful for distillation where we need the CLS embedding before head.

        Args:
            x: input tensor (B, 3, H, W)

        Returns:
            cls_embedding: (B, D)
        """
        features = self.raw_backbone.forward_features(x)  # (B, N_kept, D)
        cls_embedding = features[:, 0, :]  # (B, D)
        return cls_embedding
