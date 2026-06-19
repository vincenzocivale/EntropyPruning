"""Fast, CPU-only smoke tests for DistilledPrunedBackbone (Phase 3, Approach 3).

Uses a tiny fake timm-like ViT (no real weights/dataset) to check that LoRA is
scoped to post-prune_layer blocks only, the forward/backward pass works, and
merge_and_unload() produces a clean backbone -- without depending on GPU,
thunder checkpoints, or downloaded datasets.
"""

import torch
import torch.nn as nn
from peft.tuners.lora.layer import Linear as LoraLinear

from src.models import (
    AttentionForecaster,
    DistilledPrunedBackbone,
    ThunderBackboneAdapter,
    post_prune_lora_targets,
)
from tests._fakes import FakeViT

D, N_BLOCKS, N_PATCHES, NUM_PREFIX, PRUNE_LAYER, B = 16, 6, 20, 1, 2, 4


def _build_student():
    backbone = FakeViT(D, N_BLOCKS, N_PATCHES, NUM_PREFIX)
    adapter = ThunderBackboneAdapter(backbone)
    forecaster = AttentionForecaster(embed_dim=D, hidden=8, n_heads=2, n_layers=1)
    student = DistilledPrunedBackbone(
        backbone=backbone, adapter=adapter, forecaster=forecaster,
        prune_layer=PRUNE_LAYER, keep_ratio=0.5, lora_r=4, lora_alpha=8,
    )
    return student, adapter


def test_post_prune_lora_targets_excludes_early_blocks():
    adapter = ThunderBackboneAdapter(FakeViT(D, N_BLOCKS, N_PATCHES, NUM_PREFIX))
    targets = post_prune_lora_targets(adapter, PRUNE_LAYER)
    assert len(targets) == (N_BLOCKS - PRUNE_LAYER - 1) * 4
    assert all(f"blocks.{i}." not in t for t in targets for i in range(PRUNE_LAYER + 1))


def test_lora_scoped_to_post_prune_blocks():
    student, _ = _build_student()
    for i, blk in enumerate(student.raw_backbone.blocks):
        assert isinstance(blk.attn.qkv, LoraLinear) == (i > PRUNE_LAYER)


def test_trainable_params_are_post_prune_lora_only():
    student, _ = _build_student()
    trainable = [n for n, p in student.backbone.named_parameters() if p.requires_grad]
    assert trainable, "expected some trainable LoRA parameters"
    assert all(
        f"blocks.{i}." not in n for n in trainable for i in range(PRUNE_LAYER + 1)
    )


def test_forward_backward_shapes_and_grad():
    student, _ = _build_student()
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)
    out = student(x)
    assert out.shape == (B, D)

    out.pow(2).mean().backward()
    assert all(
        p.grad is not None
        for p in student.backbone.parameters()
        if p.requires_grad
    )


def test_forward_can_return_kept_tokens_and_original_indices():
    student, _ = _build_student()
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)
    out = student(x, return_tokens=True)
    k = int(N_PATCHES * student.keep_ratio)

    assert out["cls"].shape == (B, D)
    assert out["tokens"].shape == (B, k, D)
    assert out["features"].shape == (B, NUM_PREFIX + k, D)
    assert out["kept_indices"].shape == (B, k)
    assert out["kept_indices"].min() >= 0
    assert out["kept_indices"].max() < N_PATCHES


def test_merge_and_unload_removes_lora():
    student, _ = _build_student()
    merged = student.backbone.merge_and_unload()
    assert isinstance(merged.blocks[PRUNE_LAYER + 1].attn.qkv, nn.Linear)
    assert not isinstance(merged.blocks[PRUNE_LAYER + 1].attn.qkv, LoraLinear)
    assert not any("lora_A" in k or "lora_B" in k for k in merged.state_dict())


def test_forward_from_seq_matches_forward_on_equivalent_input():
    """forward_from_seq, given the exact sequence forward()'s pruning hook
    would see, must reproduce forward(..., return_tokens=True) bit-for-bit --
    it is the same computation, just resumed from a cached intermediate
    instead of re-deriving it via a live hook."""
    student, _ = _build_student()
    student.eval()
    x = torch.randn(B, NUM_PREFIX + N_PATCHES, D)

    expected = student(x, return_tokens=True)

    captured = {}
    handle = student.raw_backbone.blocks[PRUNE_LAYER].register_forward_hook(
        lambda module, input, output: captured.setdefault("seq", output)
    )
    student.raw_backbone.forward_features(x)
    handle.remove()

    actual = student.forward_from_seq(captured["seq"], return_tokens=True)

    assert torch.equal(actual["kept_indices"], expected["kept_indices"])
    assert torch.allclose(actual["cls"], expected["cls"], atol=1e-6)
    assert torch.allclose(actual["tokens"], expected["tokens"], atol=1e-6)
    assert torch.allclose(actual["features"], expected["features"], atol=1e-6)


def test_forward_from_seq_runs_only_post_prune_blocks_and_keeps_grad():
    student, _ = _build_student()
    seq = torch.randn(B, NUM_PREFIX + N_PATCHES, D, requires_grad=True)

    out = student.forward_from_seq(seq, return_tokens=True)
    k = int(N_PATCHES * student.keep_ratio)

    assert out["cls"].shape == (B, D)
    assert out["tokens"].shape == (B, k, D)
    assert out["kept_indices"].shape == (B, k)

    out["cls"].pow(2).mean().backward()
    assert seq.grad is not None
    assert all(
        p.grad is not None
        for p in student.backbone.parameters()
        if p.requires_grad
    )
