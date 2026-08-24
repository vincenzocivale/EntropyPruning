"""WSI-EAF landmark forecaster: O(N*K) cross-tile attention distillation.

Distills TITAN's real final-layer cross-tile attention from the tile encoder's
*final* embeddings, without ever materializing dense O(N^2) attention over the
tile bag. Every tile cross-attends only to a small, fixed number of learned
landmark tokens (K << N), mirroring the Set-Transformer / Induced Set
Attention Block (ISAB) pattern: landmarks first summarize the whole bag
(tile -> landmark, O(N*K)), refine among themselves (O(K^2), K is small), then
broadcast context back to every tile (landmark -> tile, O(N*K)).

See docs/offline_eaf_pipeline.md and the WSI-EAF attention-signal investigation
notes for why this exists: isolated per-tile statistics (centroid distance,
embedding norm, prototype prevalence, intra-tile heterogeneity, spatial-
neighbor dissimilarity) and even flexible case-disjoint kNN in the full
embedding space carry *zero* recoverable signal about TITAN's cross-tile
attention -- the signal only emerges from learned cross-tile interaction, at
which point O(N^2) becomes the actual cost problem for large tile bags (some
HISTAI slides have >20k tiles). This module trades exactness for sub-quadratic
cost while keeping real learned cross-tile mixing, the one ingredient shown to
matter.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WSILandmarkForecaster(nn.Module):
    """ISAB-style forecaster of per-tile WSI-FM attention.

    Args:
        embed_dim: dimensionality of the tile encoder's final embedding
            (768 for CONCH v1.5's attentional-pooler output).
        hidden: internal model width.
        n_heads: attention heads per block.
        n_layers: number of (tile->landmark, landmark self-attn,
            landmark->tile) blocks.
        n_landmarks: K, the number of learned landmark tokens. Controls the
            O(N*K) cost; independent of bag size N.
        dropout: dropout probability throughout.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_landmarks: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_landmarks = n_landmarks
        self.input_proj = nn.Linear(embed_dim, hidden)
        # LayerNorm right after the projection, before any attention touches it:
        # without it K/V enter tile_to_landmark at the raw Linear output scale
        # (empirically ~0.6 std for CONCH v1.5 embeddings), an arbitrary, untuned
        # magnitude to be doing dot-product attention against.
        self.input_norm = nn.LayerNorm(hidden)
        # A standard small-init (e.g. 0.02, transformer-embedding-style) landmark
        # scale is *catastrophically* wrong here: with bag sizes N up to ~20k,
        # log(N) up to ~9.9, and MultiheadAttention's own 1/sqrt(head_dim) score
        # scaling, small-init landmark queries against normalized keys produce
        # logits with std orders of magnitude below what's needed to move the
        # softmax off uniform (verified empirically: entropy_ratio == 1.0000 to
        # 4 decimals, i.e. every landmark's cross-attention output collapses to
        # the exact same tile-bag mean, and -- because all 128 landmarks are then
        # numerically indistinguishable -- they receive identical gradients and
        # NEVER differentiate, however long training runs). This std is chosen to
        # sit comfortably past that collapse point (entropy_ratio ~0.5, landmark
        # pairwise cosine similarity ~0.7 on a real ~6k-tile slide, vs ~1.0/~1.0
        # at std=0.02) while still leaving genuine room for learned attention to
        # sharpen further during training.
        self.landmarks = nn.Parameter(torch.randn(1, n_landmarks, hidden) * 16.0)
        # Fixed (non-learnable) target scale the raw parameter above gets rescaled to
        # on every forward pass -- see the renormalization in forward(). Without this,
        # nothing bounds how large `self.landmarks` can grow during training: gradient
        # descent can push its magnitude up indefinitely (a real, observed failure mode
        # here -- grad_norm climbing from ~0.1 to 10^5-10^6 over a few hundred/thousand
        # real optimizer steps, a classic runaway feedback loop: larger landmarks ->
        # larger Q magnitude into tile_to_landmark -> more extreme/saturated softmax ->
        # more extreme outputs -> larger loss gradients -> even larger landmark update).
        # Renormalizing every forward pass keeps the *escape-collapse* property (see the
        # docstring above `self.landmarks`) permanent instead of just true at init.
        self._landmark_scale = 16.0

        self.tile_to_landmark = nn.ModuleList(
            [nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True) for _ in range(n_layers)]
        )
        self.landmark_norms_in = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.landmark_self_attn = nn.ModuleList(
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
        self.landmark_to_tile = nn.ModuleList(
            [nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True) for _ in range(n_layers)]
        )
        self.tile_norms_out = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        self.norm = nn.LayerNorm(hidden)
        self.score_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        """tile_embeddings: [B, N, embed_dim] (typically B=1, N up to ~20k).

        Returns per-tile logits [B, N] (unnormalized; apply softmax/log_softmax
        outside for a distribution, matching AttentionForecaster's contract).
        """
        batch, _n_tiles, _dim = tile_embeddings.shape
        x = self.input_norm(self.input_proj(tile_embeddings))
        raw_landmarks = self.landmarks.expand(batch, -1, -1)
        # Per-vector mean-center + rescale to a fixed std, every forward pass -- caps
        # landmark magnitude regardless of how the underlying parameter drifts during
        # training (see comment on self._landmark_scale). Preserves direction/relative
        # shape (what makes the 128 landmarks different from each other), just not
        # whatever raw magnitude gradient descent currently has them at.
        landmarks = (raw_landmarks - raw_landmarks.mean(-1, keepdim=True)) / (
            raw_landmarks.std(-1, keepdim=True) + 1e-6
        ) * self._landmark_scale

        blocks = zip(
            self.tile_to_landmark,
            self.landmark_norms_in,
            self.landmark_self_attn,
            self.landmark_to_tile,
            self.tile_norms_out,
        )
        for t2l, lm_norm, lm_self_attn, l2t, tile_norm in blocks:
            # Landmarks summarize the whole bag: O(N*K).
            lm_out, _ = t2l(landmarks, x, x)
            landmarks = lm_norm(landmarks + lm_out)
            # Landmarks refine among themselves: O(K^2), K is small so this is cheap
            # regardless of bag size.
            landmarks = lm_self_attn(landmarks)
            # Every tile reads back the (now bag-contextualized) landmarks: O(N*K).
            tile_out, _ = l2t(x, landmarks, landmarks)
            x = tile_norm(x + tile_out)

        x = self.norm(x)
        return self.score_head(x).squeeze(-1)
