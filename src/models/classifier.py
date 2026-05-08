"""
Adaptation strategies for the EAF base classifier (Phase 1).

All strategies share the same interface via BaseClassifier:
    - model.raw_backbone  → underlying timm VisionTransformer (for block-level hooks)
    - model.trainable_backbone_params → params that receive the backbone learning rate
    - model.adapter       → ThunderBackboneAdapter (embed_dim, n_patches, etc.)
    - model.head          → classification head (LayerNorm → Dropout → Linear)
    - model(x)            → class logits

Available strategies:

    LinearProbingClassifier  — backbone frozen, only head trained
    LoRAClassifier           — low-rank adapters injected into backbone + head trained
    FullFinetuneClassifier   — all backbone weights + head trained
    BitFitClassifier         — only bias terms in backbone + head trained

Factory:

    build_classifier(strategy, backbone, adapter, n_classes, **kwargs)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora import LoraModel

from .backbone_adapter import ThunderBackboneAdapter


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseClassifier(nn.Module, ABC):
    """Abstract base for all EAF adaptation strategies.

    Concrete subclasses must set ``self.adapter`` and ``self.head``, and
    implement ``raw_backbone`` and ``trainable_backbone_params``.
    """

    adapter: ThunderBackboneAdapter
    head: nn.Module

    @property
    @abstractmethod
    def raw_backbone(self) -> nn.Module:
        """The underlying timm VisionTransformer.

        Used by ``collect_and_save_dataset`` and
        ``GenericLoRAWithForecasterPruning`` to access individual blocks for
        forward hooks.  Always returns the unwrapped timm model, regardless of
        whether a peft adapter has been applied on top.
        """

    @property
    @abstractmethod
    def trainable_backbone_params(self) -> List[nn.Parameter]:
        """Backbone parameters that should receive gradients during training.

        Return an empty list for strategies that freeze the backbone entirely
        (e.g. linear probing).  The training scripts use this to construct the
        optimizer with a separate (typically smaller) backbone learning rate.
        """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits of shape (B, n_classes)."""


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class LinearProbingClassifier(BaseClassifier):
    """Frozen backbone + trained linear classification head.

    The backbone is frozen entirely — no gradients flow through it during
    training.  Only the LayerNorm → Dropout → Linear head receives updates.
    This is the lightest adaptation strategy in terms of compute and memory,
    and works well when the pretrained features are already highly informative.

    Args:
        backbone:  raw timm model from ``get_model_from_name``.
        adapter:   ``ThunderBackboneAdapter`` wrapping the same backbone.
        n_classes: number of output classes.
        dropout:   dropout rate in the classification head (default 0.1).
    """

    def __init__(self, backbone: nn.Module, adapter: ThunderBackboneAdapter,
                 n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )

    @property
    def raw_backbone(self) -> nn.Module:
        return self.backbone

    @property
    def trainable_backbone_params(self) -> List[nn.Parameter]:
        return []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self.backbone(x)
        return self.head(emb)


class LoRAClassifier(BaseClassifier):
    """LoRA-adapted backbone + trained classification head.

    Injects low-rank adapter matrices into the QKV projection, output
    projection, and both FFN layers of each transformer block.  Only the LoRA
    parameters and the classification head receive gradients.  The base
    pretrained weights remain frozen.

    Args:
        backbone:   raw timm model from ``get_model_from_name``.
        adapter:    ``ThunderBackboneAdapter`` wrapping the same backbone.
        n_classes:  number of output classes.
        lora_r:     LoRA rank (default 8).
        lora_alpha: LoRA scaling factor; effective scale = lora_alpha / lora_r
                    (default 32).
        dropout:    dropout rate in the classification head (default 0.1).

    Note:
        After construction, ``self.backbone`` is a peft ``LoraModel``.
        ``self.raw_backbone`` (= ``self.backbone.model``) gives access to the
        original timm blocks for forward hooks.
    """

    def __init__(self, backbone: nn.Module, adapter: ThunderBackboneAdapter,
                 n_classes: int, lora_r: int = 8, lora_alpha: int = 32,
                 dropout: float = 0.1):
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

    @property
    def raw_backbone(self) -> nn.Module:
        return self.backbone.model  # unwrap peft wrapper

    @property
    def trainable_backbone_params(self) -> List[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class FullFinetuneClassifier(BaseClassifier):
    """Fully fine-tuned backbone + classification head.

    All backbone weights receive gradients.  This gives the most expressive
    adaptation but requires the most GPU memory and is susceptible to
    catastrophic forgetting without careful learning rate scheduling.
    Use a small backbone learning rate (e.g. 1e-5 to 1e-6).

    Args:
        backbone:  raw timm model from ``get_model_from_name``.
        adapter:   ``ThunderBackboneAdapter`` wrapping the same backbone.
        n_classes: number of output classes.
        dropout:   dropout rate in the classification head (default 0.1).
    """

    def __init__(self, backbone: nn.Module, adapter: ThunderBackboneAdapter,
                 n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )

    @property
    def raw_backbone(self) -> nn.Module:
        return self.backbone

    @property
    def trainable_backbone_params(self) -> List[nn.Parameter]:
        return list(self.backbone.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class BitFitClassifier(BaseClassifier):
    """BitFit adaptation: only bias terms in the backbone are trained.

    All non-bias backbone parameters are frozen.  Only bias vectors (in
    attention projections, FFN layers, and layer norms) receive gradients,
    alongside the full classification head.  This is a middle ground between
    linear probing (no backbone params) and LoRA (rank-decomposed updates).

    Reference: Ben Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning
    for Transformer-based Masked Language-models", ACL 2022.

    Args:
        backbone:  raw timm model from ``get_model_from_name``.
        adapter:   ``ThunderBackboneAdapter`` wrapping the same backbone.
        n_classes: number of output classes.
        dropout:   dropout rate in the classification head (default 0.1).
    """

    def __init__(self, backbone: nn.Module, adapter: ThunderBackboneAdapter,
                 n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        for name, param in self.backbone.named_parameters():
            param.requires_grad_("bias" in name)
        self.head = nn.Sequential(
            nn.LayerNorm(adapter.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(adapter.embed_dim, n_classes),
        )

    @property
    def raw_backbone(self) -> nn.Module:
        return self.backbone

    @property
    def trainable_backbone_params(self) -> List[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Maps CLI strategy names to their constructor classes.
_STRATEGY_MAP = {
    "linear_probing": LinearProbingClassifier,
    "lora":           LoRAClassifier,
    "full":           FullFinetuneClassifier,
    "bitfit":         BitFitClassifier,
}

STRATEGIES = list(_STRATEGY_MAP.keys())


def build_classifier(
    strategy: str,
    backbone: nn.Module,
    adapter: ThunderBackboneAdapter,
    n_classes: int,
    **kwargs,
) -> BaseClassifier:
    """Instantiate a classifier using the requested adaptation strategy.

    Args:
        strategy:  one of ``"linear_probing"``, ``"lora"``, ``"full"``,
                   ``"bitfit"``.
        backbone:  raw timm model from ``get_model_from_name``.
        adapter:   ``ThunderBackboneAdapter`` for the same backbone.
        n_classes: number of output classes.
        **kwargs:  forwarded to the strategy constructor
                   (e.g. ``lora_r``, ``lora_alpha``, ``dropout``).

    Returns:
        A ``BaseClassifier`` subclass instance, not yet moved to a device.

    Raises:
        ValueError: if ``strategy`` is not one of the supported values.

    Examples::

        model = build_classifier("lora", backbone, adapter, n_classes, lora_r=16)
        model = build_classifier("linear_probing", backbone, adapter, n_classes)
    """
    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown adaptation strategy '{strategy}'. "
            f"Choose from: {STRATEGIES}"
        )
    return _STRATEGY_MAP[strategy](backbone, adapter, n_classes, **kwargs)
