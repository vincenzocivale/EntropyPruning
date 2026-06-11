import torch
import torch.nn as nn

from .backbone_adapter import ThunderBackboneAdapter
from .lora_utils import wrap_lora


class GenericLoRAWithForecasterPruning(nn.Module):
    """
    Forecaster-guided token pruning model for any timm-based Thunder backbone.

    Runs the full backbone up to prune_layer, scores spatial patch tokens with a
    frozen AttentionForecaster, keeps the top keep_ratio fraction, then runs the
    remaining blocks. Prefix tokens (CLS + register tokens) are always preserved.

    Args:
        backbone:    raw timm model from thunder's get_model_from_name.
        adapter:     ThunderBackboneAdapter for backbone.
        n_classes:   number of output classes.
        forecaster:  trained AttentionForecaster (must be frozen before passing in).
        prune_layer: block index where pruning is applied (0-indexed).
        keep_ratio:  fraction of spatial patch tokens to keep (e.g. 0.1 = top 10%).
        lora_r, lora_alpha: LoRA parameters (must match Phase 1 for warm-start).
        dropout:     classifier head dropout.

    Note:
        ``backbone`` and ``head`` use the same LoRA config and architecture as
        ``GenericLoRAClassifier`` (Phase 1), so a Phase-1 ``adapted_state_dict()`` can be
        loaded directly into ``self.backbone`` / ``self.head`` to warm-start training
        (see ``load_lora_adapted_weights`` and ``scripts/finetune_pruned.py``).
    """

    def __init__(self, backbone: nn.Module, adapter: ThunderBackboneAdapter,
                 n_classes: int, forecaster: nn.Module,
                 prune_layer: int, keep_ratio: float,
                 lora_r: int = 8, lora_alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.adapter = adapter
        self.backbone = wrap_lora(backbone, lora_r=lora_r, lora_alpha=lora_alpha)
        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )
        self.forecaster = forecaster
        self.prune_layer = prune_layer
        self.keep_ratio = keep_ratio

    @property
    def raw_backbone(self) -> nn.Module:
        """The underlying timm VisionTransformer (unwrapped from peft)."""
        return self.backbone.model

    def forward(self, x):
        hook = self._make_block_hook()(self.prune_layer)
        orig_fwd = self.raw_backbone.blocks[self.prune_layer].forward
        self.raw_backbone.blocks[self.prune_layer].forward = hook
        out = self.head(self.backbone(x))
        self.raw_backbone.blocks[self.prune_layer].forward = orig_fwd
        return out

    def _make_block_hook(self):
        num_prefix = self.adapter.num_prefix_tokens

        def make_hook(idx):
            orig_fwd = self.raw_backbone.blocks[idx].forward
            training = self.training
            forecaster = self.forecaster
            keep_ratio = self.keep_ratio

            def block_fwd(x):
                x = orig_fwd(x)
                B, _, D = x.shape
                prefix = x[:, :num_prefix, :]       # CLS + register tokens — always kept
                patches = x[:, num_prefix:, :]      # spatial patches — subject to pruning
                N = patches.shape[1]

                with torch.no_grad():
                    scores = forecaster(patches)     # (B, N)

                k_keep = max(1, int(N * keep_ratio))
                topk_vals = scores.topk(k_keep, dim=-1).values
                threshold = topk_vals[:, -1:]
                soft_mask = torch.sigmoid((scores - threshold) / 0.05)
                hard_mask = (scores >= threshold).float()
                st_mask = hard_mask - soft_mask.detach() + soft_mask

                topk_idx = scores.topk(k_keep, dim=-1).indices
                if training:
                    masked_patches = patches * st_mask.unsqueeze(-1)
                    kept = torch.stack(
                        [masked_patches[b][topk_idx[b]] for b in range(B)]
                    )
                else:
                    kept = torch.stack(
                        [patches[b][topk_idx[b]] for b in range(B)]
                    )
                return torch.cat([prefix, kept], dim=1)
            return block_fwd
        return make_hook
