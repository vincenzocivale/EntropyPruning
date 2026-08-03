from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.wsi_pipeline.wsi_models.titan_attention import (
    TitanAttentionCaptureConfig,
    capture_titan_attention,
    infer_titan_tile_to_token,
)


class ExplicitAttention(nn.Module):
    def __init__(self, dim: int = 8, heads: int = 2) -> None:
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.scale = (dim // heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(output)


class SDPAAttention(nn.Module):
    def __init__(self, dim: int = 8, heads: int = 2) -> None:
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return output.transpose(1, 2).reshape(batch, tokens, dim)


class FakeVisionModel(nn.Module):
    def __init__(self, attention_type: type[nn.Module]) -> None:
        super().__init__()
        self.attn_blocks = nn.ModuleList([attention_type(), attention_type()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.attn_blocks:
            x = block(x)
        return x[:, 0]


def _run_capture(attention_type: type[nn.Module]):
    torch.manual_seed(3)
    model = FakeVisionModel(attention_type).eval()
    features = torch.randn(5, 8)
    cls = torch.zeros(1, 8)
    sequence = torch.cat([cls, features], dim=0).unsqueeze(0)
    coords = torch.arange(10).reshape(5, 2)
    embedding, result = capture_titan_attention(
        model,
        lambda: model(sequence),
        coords=coords,
        patch_size_level0=512,
        config=TitanAttentionCaptureConfig(
            modes=("global_to_tokens", "received", "rollout", "full"),
            full_layers=(-1,),
            max_full_attention_tokens=32,
        ),
    )
    return embedding, result


def test_captures_explicit_post_softmax_attention() -> None:
    embedding, result = _run_capture(ExplicitAttention)
    assert embedding.shape == (1, 8)
    assert result.attention["global_to_tokens"].shape == (2, 2, 6)
    assert result.attention["received_by_tokens"].shape == (2, 2, 6)
    assert result.attention["rollout_global_to_tokens"].shape == (6,)
    assert result.attention["full_layer_001"].shape == (2, 6, 6)
    assert result.attention["global_to_tiles_broadcast"].shape == (2, 2, 5)
    assert result.auxiliary["tile_to_token"].tolist() == [1, 2, 3, 4, 5]
    assert "softmax_exact" in result.metadata["attention_capture_backends"]


def test_captures_sdpa_attention_from_real_qk() -> None:
    _, result = _run_capture(SDPAAttention)
    assert result.attention["global_to_tokens"].shape == (2, 2, 6)
    assert "sdpa_qk_exact" in result.metadata["attention_capture_backends"]
    full = result.attention["full_layer_001"].float()
    torch.testing.assert_close(full.sum(dim=-1), torch.ones_like(full.sum(dim=-1)), atol=2e-3, rtol=2e-3)


def test_dense_grid_mapping_is_validated_against_token_count() -> None:
    class PatchEmbed(nn.Module):
        patch_size = (2, 2)

    model = nn.Module()
    model.patch_embed = PatchEmbed()
    coords = torch.tensor(
        [
            [0, 0],
            [512, 0],
            [1024, 0],
            [1536, 0],
            [0, 512],
            [512, 512],
            [1024, 512],
            [1536, 512],
        ]
    )
    mapping, metadata = infer_titan_tile_to_token(
        coords,
        token_count=3,  # one prefix + 1x2 spatial tokens
        patch_size_level0=512,
        model=model,
    )
    assert mapping is not None
    assert mapping.tolist() == [1, 1, 2, 2, 1, 1, 2, 2]
    assert metadata["tile_token_mapping"] == "dense_grid_ceil_validated"
