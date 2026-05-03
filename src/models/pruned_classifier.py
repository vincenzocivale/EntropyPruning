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
        self._install_pruning_hook()

    def _install_pruning_hook(self):
        block = self.backbone.model.blocks[self.prune_layer]
        orig_fwd = block.forward

        def pruned_fwd(x):
            x = orig_fwd(x)
            B, N, D = x.shape
            patch_emb = x[:, 1:]
            with torch.no_grad():
                scores = self.forecaster(patch_emb)
            k_keep = max(1, int((N - 1) * self.keep_ratio))
            # single topk call — reuse values for threshold and indices for gather
            topk = scores.topk(k_keep, dim=-1)
            threshold = topk.values[:, -1:]
            topk_idx = topk.indices
            soft_mask = torch.sigmoid((scores - threshold) / 0.05)
            hard_mask = (scores >= threshold).float()
            st_mask = hard_mask - soft_mask.detach() + soft_mask
            cls_tok = x[:, :1, :]
            patches = x[:, 1:, :]
            # vectorised gather — avoids Python loop over batch dim
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)
            if self.training:
                masked_patches = patches * st_mask.unsqueeze(-1)
                kept = torch.gather(masked_patches, 1, idx_expanded)
            else:
                kept = torch.gather(patches, 1, idx_expanded)
            return torch.cat([cls_tok, kept], dim=1)

        block.forward = pruned_fwd

    def forward(self, x):
        return self.head(self.backbone(x))
