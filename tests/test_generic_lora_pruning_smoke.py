"""Fast, CPU-only smoke tests for GenericLoRAWithForecasterPruning (Phase 3,
Approach 2), checking the post-prune_layer LoRA scoping fix: blocks at or
before prune_layer must stay plain nn.Linear (frozen, pretrained), only
blocks strictly after prune_layer get LoRA-adapted.
"""

import torch
from peft.tuners.lora.layer import Linear as LoraLinear

from src.models import (
    AttentionForecaster,
    GenericLoRAWithForecasterPruning,
    ThunderBackboneAdapter,
)
from tests._fakes import FakeViT

D, N_BLOCKS, N_PATCHES, NUM_PREFIX, PRUNE_LAYER, N_CLASSES, B = 16, 6, 20, 1, 2, 3, 4


def _build_model():
    backbone = FakeViT(D, N_BLOCKS, N_PATCHES, NUM_PREFIX)
    adapter = ThunderBackboneAdapter(backbone)
    forecaster = AttentionForecaster(embed_dim=D, hidden=8, n_heads=2, n_layers=1)
    model = GenericLoRAWithForecasterPruning(
        backbone=backbone, adapter=adapter, n_classes=N_CLASSES, forecaster=forecaster,
        prune_layer=PRUNE_LAYER, keep_ratio=0.5, lora_r=4, lora_alpha=8,
    )
    return model


def test_lora_scoped_to_post_prune_blocks():
    model = _build_model()
    for i, blk in enumerate(model.raw_backbone.blocks):
        assert isinstance(blk.attn.qkv, LoraLinear) == (i > PRUNE_LAYER)
        assert isinstance(blk.attn.proj, LoraLinear) == (i > PRUNE_LAYER)
        assert isinstance(blk.mlp.fc1, LoraLinear) == (i > PRUNE_LAYER)


def test_prune_layer_block_itself_has_no_lora():
    """The prune_layer block runs on the full (unpruned) sequence in both the
    student and any unpruned baseline, so it has nothing to compensate for."""
    model = _build_model()
    assert not isinstance(model.raw_backbone.blocks[PRUNE_LAYER].attn.qkv, LoraLinear)


def test_trainable_backbone_params_exclude_pre_and_at_prune_blocks():
    model = _build_model()
    trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert trainable, "expected some trainable LoRA parameters"
    assert all(
        f"blocks.{i}." not in n
        for n in trainable
        for i in range(PRUNE_LAYER + 1)
    )


def test_forward_shape_and_grad_flows_to_lora_and_head():
    model = _build_model()
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)
    logits = model(x)
    assert logits.shape == (B, N_CLASSES)

    logits.sum().backward()
    backbone_grads = [
        p.grad is not None for p in model.backbone.parameters() if p.requires_grad
    ]
    assert backbone_grads and all(backbone_grads)
    assert all(p.grad is not None for p in model.head.parameters())
