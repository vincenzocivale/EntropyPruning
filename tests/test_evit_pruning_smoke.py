"""Fast CPU-only smoke tests for the EViT pruning integration."""

import torch
from peft.tuners.lora.layer import Linear as LoraLinear

from src.models import (
    LoRAWithEViTPruning,
    ThunderBackboneAdapter,
    adjust_evit_keep_rate,
    complement_indices,
    parse_evit_drop_locs,
)
from tests._fakes import FakeViT


D, N_BLOCKS, N_PATCHES, NUM_PREFIX, N_CLASSES, B = 16, 6, 20, 1, 3, 4


def _build_model(base_keep_rate=0.5, drop_locs=(2, 4), fuse_token=True):
    backbone = FakeViT(D, N_BLOCKS, N_PATCHES, NUM_PREFIX)
    adapter = ThunderBackboneAdapter(backbone)
    return LoRAWithEViTPruning(
        backbone=backbone,
        adapter=adapter,
        n_classes=N_CLASSES,
        base_keep_rate=base_keep_rate,
        drop_locs=drop_locs,
        fuse_token=fuse_token,
        lora_r=4,
        lora_alpha=8,
    )


def test_parse_evit_drop_locs_accepts_comma_and_tuple_like_strings():
    assert parse_evit_drop_locs("3,6,9", 12) == [3, 6, 9]
    assert parse_evit_drop_locs("(3, 6, 9)", 12) == [3, 6, 9]


def test_adjust_evit_keep_rate_matches_fixed_and_gradual_modes():
    assert adjust_evit_keep_rate(0, 0, 10, 0.7, shrink_epochs=0) == 0.7
    assert adjust_evit_keep_rate(0, 0, 10, 0.7, shrink_start_epoch=0, shrink_epochs=1) == 1.0
    assert adjust_evit_keep_rate(1, 0, 10, 0.7, shrink_start_epoch=0, shrink_epochs=1) == 0.7


def test_complement_indices_returns_rowwise_missing_indices():
    idx = torch.tensor([[0, 2], [1, 3]])
    comp = complement_indices(idx, 4)
    assert torch.equal(comp, torch.tensor([[1, 3], [0, 2]]))


def test_evit_forward_returns_logits_and_reduces_tokens_inside_drop_blocks():
    model = _build_model(base_keep_rate=0.5, drop_locs=(2,), fuse_token=False)
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)
    shapes = []

    def _capture(module, inputs, output):
        shapes.append(output.shape[1])

    handle = model.raw_backbone.blocks[2].register_forward_hook(_capture)
    try:
        logits = model.eval()(x)
    finally:
        handle.remove()

    assert logits.shape == (B, N_CLASSES)
    assert shapes[-1] == NUM_PREFIX + int(N_PATCHES * model.base_keep_rate)


def test_evit_fuse_token_keeps_one_extra_inattentive_token():
    model = _build_model(base_keep_rate=0.5, drop_locs=(2,), fuse_token=True)
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)
    shapes = []

    def _capture(module, inputs, output):
        shapes.append(output.shape[1])

    handle = model.raw_backbone.blocks[2].register_forward_hook(_capture)
    try:
        logits = model.eval()(x)
    finally:
        handle.remove()

    assert logits.shape == (B, N_CLASSES)
    assert shapes[-1] == NUM_PREFIX + int(N_PATCHES * model.base_keep_rate) + 1


def test_evit_backward_updates_lora_and_head():
    model = _build_model()
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)
    loss = model(x).pow(2).mean()
    loss.backward()

    assert all(p.grad is not None for p in model.head.parameters())
    assert all(
        p.grad is not None
        for p in model.backbone.parameters()
        if p.requires_grad
    )


def test_evit_lora_scoped_from_first_drop_location_onward():
    model = _build_model(drop_locs=(2, 4))
    for i, blk in enumerate(model.raw_backbone.blocks):
        assert isinstance(blk.attn.qkv, LoraLinear) == (i >= 2)
