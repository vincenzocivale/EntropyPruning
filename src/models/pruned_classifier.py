import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel

from .backbone_adapter import ThunderBackboneAdapter


class FrozenPrunedLinearProbe(nn.Module):
    """Frozen ViT encoder + frozen EAF forecaster + trainable linear head.

    Applies EAF-guided token pruning at inference and training time. The
    backbone and forecaster are never updated; only ``head`` is optimised.

    When ``layers_source`` contains more than one index the patch embeddings
    from each source block are captured via forward hooks and concatenated
    along the feature dimension before being fed to the forecaster. Pruning
    is applied after the highest-indexed source block.

    Args:
        backbone:      raw timm ViT (will be frozen).
        adapter:       ThunderBackboneAdapter wrapping ``backbone``.
        forecaster:    AttentionForecaster (will be frozen). Its input
                       dimension must equal ``embed_dim * len(layers_source)``.
        n_classes:     number of output classes.
        layers_source: block indices to capture (e.g. ``[1, 2, 3, 4, 5]``).
                       A single int is accepted for convenience.
        keep_ratio:    fraction of spatial patch tokens to retain (0 < r ≤ 1).
    """

    def __init__(self, backbone, adapter, forecaster, n_classes,
                 layers_source, keep_ratio=0.5):
        super().__init__()
        if isinstance(layers_source, int):
            layers_source = [layers_source]
        self.layers_source = sorted(layers_source)
        self.prune_layer = max(self.layers_source)
        self.keep_ratio = keep_ratio
        self.num_prefix = adapter.num_prefix_tokens

        self.backbone = backbone
        self.adapter = adapter
        self.forecaster = forecaster
        self.head = nn.Linear(adapter.embed_dim, n_classes)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.forecaster.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        """Keep backbone and forecaster in eval mode at all times."""
        super().train(mode)
        self.backbone.eval()
        self.forecaster.eval()
        return self

    def forward(self, x):
        captured = {}
        handles = []
        num_prefix = self.num_prefix

        for ls in self.layers_source[:-1]:
            def _cap(module, input, output, _ls=ls):
                captured[_ls] = output[:, num_prefix:].clone()
            handles.append(self.backbone.blocks[ls].register_forward_hook(_cap))

        forecaster = self.forecaster
        keep_ratio = self.keep_ratio
        layers_source = self.layers_source

        def _prune(module, input, output):
            prefix = output[:, :num_prefix]
            patches = output[:, num_prefix:]
            embs = [captured[ls] for ls in layers_source[:-1]] + [patches]
            emb_cat = torch.cat(embs, dim=-1)
            with torch.no_grad():
                scores = forecaster(emb_cat)
            k = max(1, int(patches.shape[1] * keep_ratio))
            idx = scores.topk(k, dim=-1).indices
            kept = patches.gather(1, idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
            return torch.cat([prefix, kept], dim=1)

        handles.append(self.backbone.blocks[self.prune_layer].register_forward_hook(_prune))

        try:
            features = self.backbone.forward_features(x)
        finally:
            for h in handles:
                h.remove()

        return self.head(features[:, 0])


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
        lora_r, lora_alpha: LoRA parameters.
        dropout:     classifier head dropout.

    Note:
        Load Phase 1 checkpoint with strict=False — peft key prefix differs from
        a plain model, and the forecaster keys are new.
    """

    def __init__(self, backbone: nn.Module, adapter: ThunderBackboneAdapter,
                 n_classes: int, forecaster: nn.Module,
                 prune_layer: int, keep_ratio: float,
                 lora_r: int = 8, lora_alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.adapter = adapter
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1, bias="none",
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
