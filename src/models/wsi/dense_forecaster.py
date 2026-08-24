"""WSI-EAF dense-attention forecaster: standard multi-head self-attention over
the whole tile bag, mirroring TITAN's own block design at a smaller scale.

Replaces the earlier `WSILandmarkForecaster` (ISAB/landmark-bottleneck design,
see `landmark_forecaster.py`), which turned out to have a self-inflicted
instability class: its 128 learned landmark tokens started numerically
identical to each other (a symmetric initialization), which either collapsed
permanently (small init) or, once escaped via a large init, could grow
unboundedly during training (nothing renormalized them), producing grad_norm
spikes into the hundreds of thousands and stalling training indefinitely on
the full corpus despite working fine on every smaller-scale check.

Standard dense self-attention has no equivalent failure mode: the "queries"
are the tiles themselves, which are never identical to begin with (they come
from genuinely different tile embeddings), so there is no symmetric
initialization to collapse or escape. TITAN's own vision_encoder blocks are
exactly this: plain qkv+proj attention, 768-dim, 12 heads, only ~7.1M params
per block -- proven stable in production on slides with >20k tiles. This
module copies that block design (standard nn.TransformerEncoderLayer,
pre-norm) at a smaller width/head-count than TITAN's own, trained from
scratch to distill TITAN's real attention rather than reusing its weights.

O(N^2) attention cost is real but was likely never the actual blocker here --
PyTorch's scaled_dot_product_attention dispatches to a memory-efficient/flash
backend on modern GPUs, so even N~20k tiles does not require materializing a
dense N x N score matrix in practice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alibi import build_alibi_bias


class WSIDenseForecaster(nn.Module):
    """Standard pre-norm transformer-encoder forecaster of per-tile WSI-FM attention.

    Args:
        embed_dim: dimensionality of the tile encoder's final embedding
            (768 for CONCH v1.5's attentional-pooler output).
        hidden: internal model width (smaller than TITAN's own 768 by design,
            for a cheaper-than-TITAN's-own-layers forecaster).
        n_heads: attention heads.
        n_layers: number of standard self-attention blocks.
        dropout: dropout probability throughout.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=n_heads,
                    dim_feedforward=hidden * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.score_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        """tile_embeddings: [B, N, embed_dim] (typically B=1, N up to ~20k).

        Returns per-tile logits [B, N] (unnormalized; apply softmax/log_softmax
        outside for a distribution).
        """
        x = self.input_norm(self.input_proj(tile_embeddings))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.score_head(x).squeeze(-1)


class _ALiBiSelfAttention(nn.Module):
    """Plain qkv+proj self-attention with an additive ALiBi bias baked into SDPA,
    mirroring TITAN's own ``vision_transformer.Attention`` (its ``pos_encode='alibi'``
    branch) rather than routing through ``nn.MultiheadAttention``/``TransformerEncoderLayer``,
    neither of which take a per-head additive bias as cleanly."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, alibi_bias: torch.Tensor) -> torch.Tensor:
        batch, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, n_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        bias = alibi_bias.unsqueeze(0).to(dtype=q.dtype)  # [1, H, N, N] broadcasts over batch
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(batch, n_tokens, dim)
        return self.proj(out)


class _ALiBiBlock(nn.Module):
    """Pre-norm attention+MLP block, structurally identical to TITAN's own ``Block``
    (norm -> attn -> residual, norm -> mlp -> residual; no LayerScale/DropPath, which
    TITAN's checkpoint also has disabled by default)."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _ALiBiSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, alibi_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), alibi_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class WSIDenseForecasterALiBi(nn.Module):
    """``WSIDenseForecaster`` variant that adds TITAN-matching ALiBi spatial bias.

    Meant specifically for the "intermediate WSI-FM representation" input path
    (TITAN's own hidden state at an early vision-encoder block, captured via
    ``TitanAttentionCaptureConfig.hidden_layers`` /
    ``src/wsi_pipeline/wsi_models/titan_attention.py``) rather than the tile
    encoder's context-free final embedding: that input was itself produced
    under TITAN's ALiBi bias, so the forecaster's own remaining self-attention
    blocks need the same bias to have a chance of reproducing the final-layer
    target, which a plain permutation-invariant ``WSIDenseForecaster`` cannot
    do from content alone (see ``alibi.py``'s docstring). Depth/width are
    meant to be sized to whatever TITAN depth remains after the chosen source
    layer (e.g. 2-3 blocks for a mid-tower source, given TITAN itself has only
    6), not to TITAN's own 768-dim/12-head scale.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([_ALiBiBlock(hidden, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.score_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )
        self.n_heads = n_heads

    def forward(self, tile_embeddings: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """tile_embeddings: [B, N, embed_dim]. coords: [B, N, >=2] or [N, >=2],
        same tile order as tile_embeddings (B=1 only -- ALiBi bias is built per-call
        from real coordinates, not shared/batched across slides).

        Returns per-tile logits [B, N] (unnormalized).
        """
        if tile_embeddings.shape[0] != 1:
            raise ValueError(f"WSIDenseForecasterALiBi only supports batch=1, got {tile_embeddings.shape[0]}")
        if coords.ndim == 3:
            coords = coords[0]
        x = self.input_norm(self.input_proj(tile_embeddings))
        bias = build_alibi_bias(coords, self.n_heads, dtype=x.dtype).to(device=x.device)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.norm(x)
        return self.score_head(x).squeeze(-1)
