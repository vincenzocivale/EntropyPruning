"""Fast CPU-only smoke tests for the Cropr pruning integration."""

import torch
from peft.tuners.lora.layer import Linear as LoraLinear

from src.models import (
    LoRAWithCroprPruning,
    ThunderBackboneAdapter,
    cropr_pruning_schedule,
)
from tests._fakes import FakeViT


D, N_BLOCKS, N_PATCHES, NUM_PREFIX, N_CLASSES, B = 16, 6, 20, 1, 3, 4


def _build_model(keep_ratio=0.5, llf=True):
    backbone = FakeViT(D, N_BLOCKS, N_PATCHES, NUM_PREFIX)
    adapter = ThunderBackboneAdapter(backbone)
    return LoRAWithCroprPruning(
        backbone=backbone,
        adapter=adapter,
        n_classes=N_CLASSES,
        keep_ratio=keep_ratio,
        llf=llf,
        num_queries=1,
        num_heads=2,
        lora_r=4,
        lora_alpha=8,
    )


def test_cropr_schedule_reaches_requested_keep_ratio():
    schedule = cropr_pruning_schedule(N_PATCHES, n_modules=4, pruning_rate=3)
    assert schedule == [3, 3, 3, 3]


def test_cropr_schedule_can_derive_constant_rate_from_keep_ratio():
    schedule = cropr_pruning_schedule(N_PATCHES, n_modules=5, keep_ratio=0.5)
    assert schedule == [2, 2, 2, 2, 2]


def test_cropr_forward_returns_aux_logits_only_in_training():
    model = _build_model()
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)

    model.train()
    outputs = model(x)
    assert isinstance(outputs, list)
    assert outputs[0].shape == (B, N_CLASSES)
    assert all(aux.shape == (B, N_CLASSES) for aux in outputs[1:])
    assert len(outputs) == 1 + len(model.prune_layers)

    model.eval()
    logits = model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (B, N_CLASSES)


def test_cropr_backward_updates_lora_head_and_aux_modules():
    model = _build_model()
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)
    outputs = model(x)
    loss = sum(out.pow(2).mean() for out in outputs)
    loss.backward()

    assert all(p.grad is not None for p in model.head.parameters())
    assert any(p.grad is not None for p in model.cropr.parameters())
    assert all(
        p.grad is not None
        for p in model.backbone.parameters()
        if p.requires_grad
    )


def test_cropr_lora_scoped_after_prune_start():
    model = _build_model()
    for i, blk in enumerate(model.raw_backbone.blocks):
        assert isinstance(blk.attn.qkv, LoraLinear) == (i > 0)


def test_cropr_without_llf_keeps_final_token_count():
    model = _build_model(keep_ratio=0.5, llf=False)
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)
    shapes = []

    def _capture(module, inputs, output):
        shapes.append(output.shape[1])

    handle = model.raw_backbone.blocks[-1].register_forward_hook(_capture)
    try:
        logits = model.eval()(x)
    finally:
        handle.remove()

    assert logits.shape == (B, N_CLASSES)
    assert shapes[-1] == NUM_PREFIX + int(N_PATCHES * model.keep_ratio)
