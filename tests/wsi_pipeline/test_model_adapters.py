import pytest

from src.wsi_pipeline.model_adapters import (
    HookedViTTileTeacherAdapter,
    TileTeacherFinalOutput,
    TileTeacherOutput,
    TitanWSITeacherAdapter,
)


def test_hooked_vit_tile_teacher_adapter_shapes() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    adapter = HookedViTTileTeacherAdapter.from_timm(
        "vit_tiny_patch16_224", pretrained=False, revision="test"
    )
    assert adapter.name == "vit_tiny_patch16_224"

    images = torch.randn(2, 3, 224, 224)
    output = adapter.extract(images, early_layer=1)

    assert isinstance(output, TileTeacherOutput)
    n_patches = (224 // 16) ** 2
    assert output.early_tokens.shape == (2, n_patches, 192)
    assert output.final_attention.shape == (2, n_patches)
    assert output.tile_embeddings.shape == (2, 192)
    assert output.early_tokens.dtype == torch.float16
    assert output.final_attention.dtype == torch.float16
    # Attention over patches (post-softmax) sums to ~1 per image.
    sums = output.final_attention.float().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-2)


def test_extract_final_matches_combined_extract() -> None:
    """extract_final (the production cache-building path) must reproduce exactly what
    the combined extract() reports for final_attention/tile_embeddings -- proving the
    split into extract_final/extract_early didn't change either quantity's value."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    adapter = HookedViTTileTeacherAdapter.from_timm(
        "vit_tiny_patch16_224", pretrained=False, revision="test"
    )
    images = torch.randn(2, 3, 224, 224)
    combined = adapter.extract(images, early_layer=1)
    final_only = adapter.extract_final(images)

    assert isinstance(final_only, TileTeacherFinalOutput)
    torch.testing.assert_close(final_only.final_attention, combined.final_attention)
    torch.testing.assert_close(final_only.tile_embeddings, combined.tile_embeddings)


def test_extract_final_matches_native_embedding_and_full_attention_reference() -> None:
    """CLS-row optimization must preserve the native model output and full-matrix target."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    adapter = HookedViTTileTeacherAdapter.from_timm(
        "vit_tiny_patch16_224", pretrained=False, revision="test"
    )
    images = torch.randn(2, 3, 224, 224)
    observed = {}
    attention_module = adapter._blocks[-1].attn

    def capture(_module, inputs):
        observed["x"] = inputs[0].detach()

    handle = attention_module.register_forward_pre_hook(capture)
    with torch.inference_mode():
        native = adapter._forward_fn(images)
    handle.remove()
    extracted = adapter.extract_final(images)

    with torch.inference_mode():
        x = observed["x"]
        batch, tokens, _channels = x.shape
        qkv = attention_module.qkv(x).reshape(
            batch, tokens, 3, attention_module.num_heads, attention_module.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, _v = qkv.unbind(0)
        q = attention_module.q_norm(q)
        k = attention_module.k_norm(k)
        full = ((q @ k.transpose(-2, -1)) * attention_module.scale).float().softmax(-1)
        reference = full[:, :, 0, adapter.num_prefix_tokens:].mean(1)
        reference = reference / reference.sum(-1, keepdim=True)

    torch.testing.assert_close(
        extracted.final_attention.float(), reference.float(), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        extracted.tile_embeddings.float(), native.float(), atol=2e-3, rtol=2e-3
    )


def test_extract_early_matches_combined_extract() -> None:
    """extract_early (the online, training-time path) must reproduce exactly what the
    combined extract() reports for early_tokens at the same layer -- proving the
    early-exit optimization doesn't change the captured value, only skips wasted
    compute after it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    adapter = HookedViTTileTeacherAdapter.from_timm(
        "vit_tiny_patch16_224", pretrained=False, revision="test"
    )
    images = torch.randn(2, 3, 224, 224)
    combined = adapter.extract(images, early_layer=1)
    early_only = adapter.extract_early(images, early_layer=1)

    torch.testing.assert_close(early_only.float(), combined.early_tokens.float(), atol=2e-3, rtol=2e-3)


def test_extract_early_rejects_out_of_range_layer() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch

    adapter = HookedViTTileTeacherAdapter.from_timm(
        "vit_tiny_patch16_224", pretrained=False, revision="test"
    )
    with pytest.raises(ValueError):
        adapter.extract_early(torch.randn(1, 3, 224, 224), early_layer=999)


def test_extract_early_restores_block_after_early_exit() -> None:
    """The _EarlyExit sentinel used to abort the forward pass must never leak out of
    extract_early, and must never leave a dangling forward_hook on the block."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    adapter = HookedViTTileTeacherAdapter.from_timm(
        "vit_tiny_patch16_224", pretrained=False, revision="test"
    )
    block = adapter._blocks[1]
    n_hooks_before = len(block._forward_hooks)
    adapter.extract_early(torch.randn(1, 3, 224, 224), early_layer=1)
    assert len(block._forward_hooks) == n_hooks_before


def test_hooked_vit_tile_teacher_adapter_rejects_out_of_range_layer() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")

    adapter = HookedViTTileTeacherAdapter.from_timm(
        "vit_tiny_patch16_224", pretrained=False, revision="test"
    )
    torch = pytest.importorskip("torch")
    images = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError):
        adapter.extract(images, early_layer=999)


def test_hooked_vit_tile_teacher_adapter_restores_original_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    adapter = HookedViTTileTeacherAdapter.from_timm(
        "vit_tiny_patch16_224", pretrained=False, revision="test"
    )
    original_forwards = [block.attn.forward for block in adapter._blocks]
    adapter.extract(torch.randn(1, 3, 224, 224), early_layer=0)
    restored_forwards = [block.attn.forward for block in adapter._blocks]
    assert [f.__func__ if hasattr(f, "__func__") else f for f in original_forwards] == [
        f.__func__ if hasattr(f, "__func__") else f for f in restored_forwards
    ]


def _titan_adapter_stub(tile_score_key: str = "global_to_tiles_mass_share") -> TitanWSITeacherAdapter:
    # Exercise the pure tile-score reduction logic without downloading the real
    # (gated) TITAN checkpoint: bypass __init__, which only sets plain attributes
    # that _tile_scores reads.
    adapter = object.__new__(TitanWSITeacherAdapter)
    adapter.tile_score_key = tile_score_key
    return adapter


def test_titan_tile_scores_reduces_layers_and_heads() -> None:
    torch = pytest.importorskip("torch")
    adapter = _titan_adapter_stub()
    # [n_layers=2, heads=3, n_tiles=4]
    attention = {"global_to_tiles_mass_share": torch.rand(2, 3, 4)}
    scores = adapter._tile_scores(attention, n_tiles=4)
    assert scores.shape == (4,)


def test_titan_tile_scores_falls_back_to_1to1_token_map() -> None:
    torch = pytest.importorskip("torch")
    adapter = _titan_adapter_stub(tile_score_key="missing_key")
    attention = {"global_to_tokens": torch.rand(2, 3, 5)}  # last dim == n_tiles
    scores = adapter._tile_scores(attention, n_tiles=5)
    assert scores.shape == (5,)


def test_titan_tile_scores_raises_when_nothing_matches() -> None:
    torch = pytest.importorskip("torch")
    adapter = _titan_adapter_stub(tile_score_key="missing_key")
    attention = {"global_to_tokens": torch.rand(2, 3, 5)}  # last dim != n_tiles
    with pytest.raises(KeyError):
        adapter._tile_scores(attention, n_tiles=999)
