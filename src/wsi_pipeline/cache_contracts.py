"""Versioned contracts for frozen EAF teacher caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

CACHE_SCHEMA_VERSION = 2
# v1 -> v2 (2026-08-07): `early_tokens` [N, T, D] is no longer part of the permanent
# Tile-EAF cache (it was >99.8% of on-disk bytes: ~1.53 MiB/tile vs. ~3.1 KiB/tile for
# everything else combined, at the projected full-HISTAI+GTEx+HEST scale that is a
# many-TB-to-low-PB difference). EAF Tile training now recomputes it ONLINE via
# `HookedViTTileTeacherAdapter.extract_early` -- a cheap early-exit partial forward
# through blocks 0..early_layer only, never the full encoder -- against tile pixels
# re-read from the (never-deleted) raw WSI. See
# `src/wsi_pipeline/compact_cache_dataset.py` and docs/offline_eaf_pipeline.md. Only
# `coords`, `final_attention`, `tile_embeddings` are written/required now.


def _stable_id(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# Fixed, non-configurable descriptions of what `early_layer`/`attention_reduction`
# actually mean, pinned to `HookedViTTileTeacherAdapter.extract` (see its docstring)
# and to `src.models.online_tile_eaf.OnlineAttentionTeacher`, which the cache must
# match bit-for-bit. These are documentation baked into every cache file's metadata,
# not knobs: changing the extraction code without updating these strings is a bug.
EARLY_LAYER_SEMANTICS = (
    "0-based transformer block index; value = the OUTPUT of that block "
    "(after both its attention and MLP residual branches, i.e. the hidden state as "
    "handed to block[early_layer + 1]), CLS/register prefix tokens stripped. "
    "NOT the block's attention-submodule *input* (norm1(x), i.e. effectively the "
    "previous block's output after normalization) -- an earlier, superseded version "
    "of this pipeline captured that instead."
)
ATTENTION_SEMANTICS = (
    "CLS-to-patch self-attention (post-softmax) at the model's LAST transformer block "
    "(index n_blocks-1), i.e. the query is the CLS/prefix-0 token and keys are the "
    "patch tokens; CLS/register prefix columns are dropped before storage."
)
ATTENTION_HEAD_AGGREGATION = (
    "mean over attention heads, then L1-renormalized over the patch axis so each row "
    "sums to 1.0 (renormalization is required because dropping the CLS/prefix column "
    "after softmax leaves each row summing to <1)."
)


@dataclass(frozen=True)
class TileCacheSpec:
    """Identity + documentation for one permanent Tile-EAF/WSI-EAF cache directory.

    Only `coords`, `final_attention`, `tile_embeddings` are physically stored (see
    `CACHE_SCHEMA_VERSION` v2 note above). `early_layer`/`early_layer_semantics` do not
    affect those stored bytes at all -- they record which layer EAF Tile training is
    expected to recompute online via `extract_early`, so a reader of a cache directory
    can always tell what early-layer choice it was produced/intended for, without that
    choice forcing a cache rebuild (a compact cache is cheap enough that rebuilding for
    a layer experiment would be wasteful busywork, unlike the old early_tokens-bearing
    cache where every field genuinely determined the on-disk bytes).
    """

    tile_encoder: str
    model_revision: str = "unknown"
    early_layer: int = 2
    input_mag: int = 20
    patch_size: int = 512
    stride: int = 512
    input_mpp: float = 0.5
    dtype: str = "float16"
    attention_reduction: str = "cls_mean_heads_l1norm"
    dataset: str = "unknown"
    early_layer_semantics: str = EARLY_LAYER_SEMANTICS
    attention_semantics: str = ATTENTION_SEMANTICS
    attention_head_aggregation: str = ATTENTION_HEAD_AGGREGATION
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def cache_id(self) -> str:
        # `early_layer` is excluded: it is a training-time online-recompute parameter
        # (see class docstring), not something that determines the stored bytes, so
        # changing it must never force an unnecessary, expensive full-forward rebuild
        # of the (unrelated) cached final_attention/tile_embeddings.
        payload = asdict(self)
        payload.pop("early_layer", None)
        return _stable_id(payload)

    def metadata(self) -> dict:
        return asdict(self) | {"cache_id": self.cache_id, "kind": "tile_eaf"}


@dataclass(frozen=True)
class WSICacheSpec:
    tile_encoder: str
    wsi_encoder: str
    tile_model_revision: str = "unknown"
    wsi_model_revision: str = "unknown"
    input_mag: int = 20
    patch_size: int = 512
    dtype: str = "float16"
    tile_score_policy: str = "teacher_default"
    store_raw_attention: bool = False
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def cache_id(self) -> str:
        return _stable_id(asdict(self))

    def metadata(self) -> dict:
        return asdict(self) | {"cache_id": self.cache_id, "kind": "wsi_eaf"}
