from pathlib import Path

import torch
import torch.nn as nn
from peft.tuners.lora import LoraModel

from .backbone_adapter import ThunderBackboneAdapter
from .lora_utils import wrap_lora


class GenericLoRAClassifier(nn.Module):
    """Phase 1: full-backbone LoRA fine-tuning + classification head on full sequences.

    Matches the paper's Stage 1: LoRA adapters on qkv/proj/fc1/fc2 of every Transformer
    block, trained end-to-end with cross-entropy on full (un-pruned) token sequences.
    The resulting LoRA-adapted backbone's final-block CLS->patch attention becomes the
    Phase-2 distillation teacher (see ``load_lora_adapted_weights`` and
    ``scripts/train_forecaster.py``), and its LoRA adapters + head warm-start Phase 3's
    ``GenericLoRAWithForecasterPruning`` (see ``scripts/finetune_pruned.py``).

    Args:
        backbone:    raw timm model from thunder's get_model_from_name.
        adapter:     ThunderBackboneAdapter for backbone.
        n_classes:   number of output classes.
        lora_r, lora_alpha: LoRA parameters (must match Phase 3 for warm-start).
        dropout:     classifier head dropout.
    """

    def __init__(self, backbone: nn.Module, adapter: ThunderBackboneAdapter,
                 n_classes: int, lora_r: int = 8, lora_alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.adapter = adapter
        self.backbone = wrap_lora(backbone, lora_r=lora_r, lora_alpha=lora_alpha)
        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )

    @property
    def raw_backbone(self) -> nn.Module:
        """The underlying timm VisionTransformer (unwrapped from peft)."""
        return self.backbone.model

    @property
    def trainable_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, x):
        return self.head(self.backbone(x))

    def adapted_state_dict(self) -> dict:
        """LoRA adapters + head, used to warm-start Phase 3 and to build the Phase-2 teacher."""
        return {"backbone": self.backbone.state_dict(), "head": self.head.state_dict()}

    def load_adapted_state_dict(self, sd: dict) -> None:
        self.backbone.load_state_dict(sd["backbone"])
        self.head.load_state_dict(sd["head"])


def load_lora_adapted_weights(model: nn.Module, ckpt_path, map_location=None) -> None:
    """Load a Phase-1 ``adapted_state_dict`` into ``model.backbone`` / ``model.head``.

    Used by Phase 3 to warm-start ``GenericLoRAWithForecasterPruning`` from Phase 1's
    converged full-network LoRA adapters + head (paper's Stage1 -> Stage3 warm start:
    LoRA adapters already converged on the full-sequence task only need to adapt to the
    pruning-induced sequence shortening, instead of learning both from scratch).
    """
    state = torch.load(Path(ckpt_path), map_location=map_location)
    model.backbone.load_state_dict(state["backbone"])
    model.head.load_state_dict(state["head"])


def load_lora_adapted_backbone(raw_backbone: nn.Module, lora_r: int = 8, lora_alpha: int = 32,
                                ckpt_path=None, map_location=None) -> LoraModel:
    """Wrap a raw backbone with LoRA and load a Phase-1 checkpoint's adapter weights.

    Used by Phase 2 to build the distillation teacher: a frozen, LoRA-adapted backbone
    whose final-block CLS->patch attention reflects the task-adapted (Stage-1-equivalent)
    model. The head is ignored.
    """
    lora_backbone = wrap_lora(raw_backbone, lora_r=lora_r, lora_alpha=lora_alpha)
    if ckpt_path is not None:
        state = torch.load(Path(ckpt_path), map_location=map_location)
        lora_backbone.load_state_dict(state["backbone"])
    return lora_backbone
