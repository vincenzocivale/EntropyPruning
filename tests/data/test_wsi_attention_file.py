from __future__ import annotations

import pytest
import torch

from src.data.wsi.attention_file import reduce_attention_tensor


def test_reduce_attention_tensor_infers_unique_tile_axis() -> None:
    tensor = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    observed = reduce_attention_tensor(tensor, n_tiles=5, reduction="mean")
    expected = tensor.mean(dim=(0, 1))
    assert torch.allclose(observed, expected)


def test_reduce_attention_tensor_selects_layer_before_head_reduction() -> None:
    tensor = torch.arange(4 * 2 * 5, dtype=torch.float32).reshape(4, 2, 5)
    observed = reduce_attention_tensor(
        tensor,
        n_tiles=5,
        selections={0: -1},
        reduction="mean",
    )
    assert torch.allclose(observed, tensor[-1].mean(dim=0))


def test_reduce_attention_tensor_rejects_ambiguous_tile_axis() -> None:
    tensor = torch.zeros(5, 2, 5)
    with pytest.raises(ValueError, match="unique tile axis"):
        reduce_attention_tensor(tensor, n_tiles=5)


def test_reduce_attention_tensor_explicitly_drops_cls_token() -> None:
    # [layers, heads, query_tokens, key_tokens], with CLS at token index 0.
    tensor = torch.arange(2 * 3 * 6 * 6, dtype=torch.float32).reshape(2, 3, 6, 6)
    observed = reduce_attention_tensor(
        tensor,
        n_tiles=5,
        tile_axis=3,
        tile_slice_start=1,
        selections={0: -1, 2: 0},
        reduction="mean",
    )
    assert torch.allclose(observed, tensor[-1, :, 0, 1:].mean(dim=0))


def test_longer_token_axis_requires_explicit_slice() -> None:
    with pytest.raises(ValueError, match="special tokens"):
        reduce_attention_tensor(torch.zeros(2, 6), n_tiles=5, tile_axis=1)
