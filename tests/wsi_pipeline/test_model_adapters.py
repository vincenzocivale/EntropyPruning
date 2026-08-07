import pytest

from src.wsi_pipeline.model_adapters import (
    HookedViTTileTeacherAdapter,
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
