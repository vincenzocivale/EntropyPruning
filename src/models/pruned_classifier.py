import torch
import torch.nn as nn
import timm
from peft import LoraConfig
from peft.tuners.lora import LoraModel


class UNILoRAWithForecasterPruning(nn.Module):
    def __init__(self, n_classes, forecaster, prune_layer, keep_ratio,
                 dropout=0.1):
        super().__init__()
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, dynamic_img_size=True,
        )
        lora_config = LoraConfig(
            r=8, lora_alpha=32,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1, bias="none",
        )
        self.backbone = LoraModel(backbone, lora_config, adapter_name="default")
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
        )
        self.forecaster = forecaster
        self.prune_layer = prune_layer
        self.keep_ratio = keep_ratio

    def forward(self, x):
        make_block_hook = self._make_block_hook()
        orig_fwd = self.backbone.model.blocks[self.prune_layer].forward
        self.backbone.model.blocks[self.prune_layer].forward = \
            make_block_hook(self.prune_layer)
        out = self.head(self.backbone(x))
        self.backbone.model.blocks[self.prune_layer].forward = orig_fwd
        return out

    def _make_block_hook(self):
        def make_block_hook(idx):
            orig_fwd = self.backbone.model.blocks[idx].forward
            training = self.training
            forecaster = self.forecaster
            keep_ratio = self.keep_ratio

            def block_fwd(x):
                x = orig_fwd(x)
                B, N, D = x.shape
                patch_emb = x[:, 1:]
                with torch.no_grad():
                    scores = forecaster(patch_emb)
                k_keep = max(1, int((N - 1) * keep_ratio))
                topk_vals = scores.topk(k_keep, dim=-1).values
                threshold = topk_vals[:, -1:]
                soft_mask = torch.sigmoid((scores - threshold) / 0.05)
                hard_mask = (scores >= threshold).float()
                st_mask = hard_mask - soft_mask.detach() + soft_mask
                cls_tok = x[:, :1, :]
                patches = x[:, 1:, :]
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
                return torch.cat([cls_tok, kept], dim=1)
            return block_fwd
        return make_block_hook
