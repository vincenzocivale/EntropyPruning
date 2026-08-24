"""ALiBi attention-bias construction, matching TITAN's own vision tower.

TITAN's ViT blocks use no learned/absolute positional embedding at all
(``pos_encode_type='alibi'`` in the cached ``MahmoodLab/TITAN``
``vision_transformer.py::VisionTransformer.get_alibi``): every self-attention
score is biased by ``-slope_h * euclidean_distance(tile_i, tile_j)``, a fixed,
per-head, distance-decaying penalty computed straight from real tile
coordinates -- not learned, not content-dependent.

A WSI-EAF forecaster that consumes TITAN's own intermediate hidden state (see
``src/wsi_pipeline/wsi_models/titan_attention.py``'s ``hidden_layers`` capture)
is consuming data that was produced *under* this bias. Its own remaining
self-attention blocks need the same bias to have any chance of reproducing
TITAN's final attention: plain permutation-invariant attention has no way to
recover a purely spatial component of the target from content alone, no
matter how it's trained. See ``dense_forecaster.py`` for the block that uses
this.

Slope computation is a direct port of TITAN's own ``get_slopes`` (recursive,
handles non-power-of-2 head counts by borrowing every other slope from the
next power of 2) -- kept bit-for-bit equivalent rather than reapproximated, so
a forecaster's own bias uses exactly the same per-head decay rates the teacher
does.
"""

from __future__ import annotations

import math

import torch


def _alibi_slopes(num_heads: int) -> list[float]:
    def slopes_pow2(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return slopes_pow2(num_heads)
    nearest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base_slopes = slopes_pow2(nearest_power_of_2)
    if nearest_power_of_2 == num_heads:
        return base_slopes
    extra_slopes = slopes_pow2(2 * nearest_power_of_2)[0::2][: num_heads - nearest_power_of_2]
    return base_slopes + extra_slopes


def _infer_tile_pitch(xy: torch.Tensor) -> float:
    """Estimate the tile grid spacing (in the same units as ``xy``) directly from the
    coordinates themselves, instead of trusting a hardcoded/assumed patch size.

    TITAN's own preprocessing (``preprocess_features`` in the cached
    ``vision_transformer.py``) always floor-divides raw level-0 pixel coordinates by
    ``patch_size_lv0`` before computing ALiBi distances, so its self-attention only
    ever sees grid distances on the order of 1 (adjacent tiles) to a few hundred
    (opposite corners of a large slide). This forecaster's WSIBag only carries raw
    coordinates (HISTAI tiles are laid out on a regular grid with ~512px pitch), so
    skipping this normalization means ``torch.cdist`` returns distances ~500x larger
    than what TITAN itself ever produces -- multiplied by ALiBi's slopes, that is
    enough to blow the attention logits into a regime where softmax saturates and
    backprop through it produces exploding gradients (observed: grad_norm up to
    ~4.5e4 against a clip threshold of 1.0). The minimum positive gap between
    distinct coordinate values on either axis is exactly the tile pitch for a
    regular grid, so this needs no dataset-specific constant.
    """
    if xy.shape[0] < 2:
        return 1.0
    candidates: list[float] = []
    for axis in range(2):
        values = torch.unique(xy[:, axis])
        if values.numel() < 2:
            continue
        diffs = torch.diff(torch.sort(values).values)
        positive = diffs[diffs > 0]
        if positive.numel() > 0:
            candidates.append(float(positive.min().item()))
    return min(candidates) if candidates else 1.0


def build_alibi_bias(
    coords: torch.Tensor,
    num_heads: int,
    *,
    prefix_tokens: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive per-head attention bias from real tile coordinates.

    Args:
        coords: ``[N, >=2]`` tile coordinates (only the first two columns,
            (x, y), are used).
        num_heads: number of attention heads to produce one slope per.
        prefix_tokens: reserve this many leading rows/cols with zero bias
            (e.g. for a CLS token the caller prepends), matching TITAN's own
            ``all_bias[:, :, 1:, 1:] = bias_matrix`` handling of its class
            token.
        dtype: dtype of the returned bias tensor.

    Returns:
        ``[num_heads, N + prefix_tokens, N + prefix_tokens]``, directly usable
        as an additive ``attn_mask`` for ``F.scaled_dot_product_attention``.

    O(N^2) memory (``num_heads * N * N`` floats) -- the same scaling limit
    TITAN's own ``get_alibi`` has, since this is the identical computation.
    For very large tile bags (HISTAI slides run up to ~20k tiles), this can be
    the dominant memory cost; that is inherent to ALiBi's design, not specific
    to this forecaster.
    """
    if coords.ndim != 2 or coords.shape[-1] < 2:
        raise ValueError(f"coords must be [N,>=2], got {tuple(coords.shape)}")
    xy = coords[:, :2].to(torch.float32)
    xy = xy / _infer_tile_pitch(xy)
    dists = torch.cdist(xy, xy)
    slopes = torch.tensor(_alibi_slopes(num_heads), dtype=torch.float32, device=xy.device).view(num_heads, 1, 1)
    bias = -dists.unsqueeze(0) * slopes  # [num_heads, N, N]
    if prefix_tokens:
        n = bias.shape[-1] + prefix_tokens
        padded = torch.zeros(num_heads, n, n, dtype=bias.dtype, device=bias.device)
        padded[:, prefix_tokens:, prefix_tokens:] = bias
        bias = padded
    return bias.to(dtype)
