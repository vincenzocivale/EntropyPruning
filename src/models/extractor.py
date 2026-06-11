import torch.nn as nn

from .backbone_adapter import ThunderBackboneAdapter


class FrozenBackbone(nn.Module):
    """Frozen backbone used only for Phase-2 teacher extraction.

    The forecaster distills the final-block CLS->patch attention of this backbone. By
    default ``backbone`` is the Phase-1 LoRA-adapted backbone (a ``peft.LoraModel``, see
    ``load_lora_adapted_backbone`` and ``scripts/train_phase1_lora.py``); if no Phase-1
    checkpoint is available the pretrained backbone is used as-is. This thin wrapper
    exposes exactly what ``collect_and_save_dataset`` needs — ``.adapter``,
    ``.raw_backbone`` (for per-block attention hooks) and a ``forward`` that runs the
    backbone so the hooks fire — while keeping every weight frozen.

    Args:
        backbone: raw timm model from ``get_model_from_name``, or a ``peft.LoraModel``
                  wrapping it (optionally Phase-1 adapted).
        adapter:  ``ThunderBackboneAdapter`` wrapping the same underlying timm model.
    """

    def __init__(self, backbone: nn.Module, adapter: ThunderBackboneAdapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @property
    def raw_backbone(self) -> nn.Module:
        """The underlying timm VisionTransformer (unwrapped from peft, if wrapped)."""
        return getattr(self.backbone, "model", self.backbone)

    def forward(self, x):
        return self.backbone(x)
