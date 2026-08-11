"""Model-agnostic contracts used by offline EAF teacher extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class TileTeacherOutput:
    """Frozen tile-encoder outputs for one image batch (early + final, one forward).

    Used for numeric-equivalence testing and one-off inspection. Production code paths
    use the split methods below instead: ``extract_final`` (offline cache building --
    ``early_tokens`` is deliberately NOT part of the permanent cache, see
    ``docs/offline_eaf_pipeline.md`` -- storage cost) and ``extract_early`` (an
    efficient, early-exit partial forward run ONLINE during EAF Tile training, never
    persisted).
    """

    early_tokens: Any
    final_attention: Any
    tile_embeddings: Any


@dataclass
class TileTeacherFinalOutput:
    """The two quantities that ARE written to the permanent Tile-EAF/WSI-EAF cache."""

    final_attention: Any
    tile_embeddings: Any


class TileTeacherAdapter(Protocol):
    """Adapter implemented by CONCH/timm/UNI/Virchow-style tile teachers."""

    name: str
    revision: str

    def extract_final(self, images: Any) -> TileTeacherFinalOutput:
        """Full forward; return the final attention target and final embedding."""
        ...

    def extract_early(self, images: Any, *, early_layer: int) -> Any:
        """Cheap partial forward (blocks 0..early_layer only); return early tokens.

        Never cached -- called online, every EAF Tile training step, against tile
        pixels re-read from the WSI (see ``src/wsi_pipeline/compact_cache_dataset.py``).
        """
        ...


@dataclass
class WSITeacherOutput:
    """Frozen slide-model outputs for one complete WSI bag."""

    tile_scores: Any
    wsi_embedding: Any
    raw_attention: Any | None = None


class WSITeacherAdapter(Protocol):
    """Adapter implemented by TITAN/GigaPath/ABMIL/etc. slide teachers."""

    name: str
    revision: str

    def extract(self, tile_embeddings: Any, coords: Any) -> WSITeacherOutput:
        """Return one canonical score per tile and the full-WSI embedding."""
        ...


# ---------------------------------------------------------------------------
# Concrete adapters
#
# These wrap already-working, previously script-only extraction logic behind
# the two protocols above, so there is exactly one place that knows how to run
# each frozen teacher for offline cache creation.
# ---------------------------------------------------------------------------


class _EarlyExit(Exception):
    """Raised from inside a block's forward_hook to abort the rest of the encoder's
    forward pass (remaining blocks, final norm, pooling head) without running it.
    Used by ``HookedViTTileTeacherAdapter.extract_early`` -- never escapes that
    method, always caught immediately around the ``self._forward_fn(images)`` call."""


class HookedViTTileTeacherAdapter:
    """Generic tile teacher for CONCH/timm/UNI-style ``timm.layers.Attention`` ViTs.

    Captures, in one forward pass per image batch, exactly the three quantities the
    offline Tile-EAF/WSI-EAF cache needs, with semantics pinned to match
    ``src.models.online_tile_eaf.OnlineAttentionTeacher`` (the teacher actually driving
    the live online Tile-EAF trainer, ``scripts/train_wsi_tile_eaf_online.py``) bit for
    bit, not merely "a" reasonable early-layer/attention definition:

    - ``early_tokens``: the **output of transformer block ``early_layer`` itself**
      (0-based; after that block's attention *and* MLP residual branches — i.e. the
      hidden state as it is handed to block ``early_layer + 1``), patch tokens only
      (CLS/register prefix stripped). Captured via a plain ``register_forward_hook`` on
      the block, mirroring ``OnlineAttentionTeacher._source_hook`` exactly. This is
      deliberately *not* the block's attention-submodule *input* (which would be
      ``norm1(block_input)``, i.e. effectively the *previous* block's output after
      normalization) — an earlier version of this adapter captured that instead, which
      silently disagreed with the online teacher for the same ``early_layer`` value.
    - ``final_attention``: CLS-to-patch self-attention at the model's **last transformer
      block**, averaged over heads and then L1-renormalized over the patch axis so each
      row sums to 1 (``TileCacheSpec.attention_reduction == "cls_mean_heads_l1norm"``),
      exactly matching ``OnlineAttentionTeacher._target_attention``. Renormalization
      matters: softmax attention including the CLS/register prefix does not sum to 1
      once those prefix columns are dropped, so skipping it (as an earlier version of
      this adapter did) leaves a teacher distribution EAF Tile was never trained
      against upstream.
    - ``tile_embeddings``: the encoder's own final pooled output exactly as produced by
      calling the model (``model(images)`` for a plain timm ViT, or ``conch(images)``
      for CONCH v1.5 — see ``from_conch``). For CONCH v1.5 this is the 768-d output of
      its attentional pooler + LayerNorm, not a raw CLS token; no extra CLS-slicing is
      applied unless the model itself returns a token sequence.

    Block discovery reuses ``src.wsi_pipeline.tile_encoders.hooks.find_transformer_blocks``
    so CONCH v1.5's ``trunk.blocks`` and plain timm ViTs (UNI, UNI2, Virchow, H-optimus,
    ...) are both supported as long as their attention module exposes the standard
    ``qkv``/``scale``/``attn_drop``/``proj``/``proj_drop`` interface (optionally
    ``q_norm``/``k_norm``). Anything else raises immediately instead of silently
    capturing the wrong tensor. Prefix-token count is auto-resolved per
    ``resolve_num_prefix_tokens`` (CLS-only for CONCH v1.5 and UNI-family; override via
    ``num_prefix_tokens`` if a future encoder uses register tokens too).
    """

    def __init__(
        self,
        model: Any,
        *,
        name: str = "vit",
        revision: str = "unknown",
        device: Any = None,
        forward_fn: Any = None,
        transform: Any = None,
        input_size: int | None = None,
        num_prefix_tokens: int | None = None,
    ) -> None:
        from .tile_encoders.hooks import find_transformer_blocks, resolve_num_prefix_tokens

        self.model = model.eval()
        if device is not None:
            self.model = self.model.to(device)
        self.name = name
        self.revision = revision
        self._blocks = find_transformer_blocks(self.model)
        self.num_prefix_tokens = (
            num_prefix_tokens
            if num_prefix_tokens is not None
            else resolve_num_prefix_tokens(self.model, self._blocks)
        )
        # Default forward path is a plain classifier-style `model(images)`; CONCH's
        # wrapper needs `conch(images)` too, but with a different constructor path
        # (see `from_conch`) since it is not itself a timm model.
        self._forward_fn = forward_fn or (lambda images: self.model(images))
        self.transform = transform
        self.input_size = input_size

    def to(self, device: Any) -> "HookedViTTileTeacherAdapter":
        self.model = self.model.to(device)
        return self

    @classmethod
    def from_timm(
        cls,
        model_name: str,
        *,
        pretrained: bool = True,
        revision: str = "unknown",
        device: Any = None,
        input_size: int = 224,
    ) -> "HookedViTTileTeacherAdapter":
        import timm
        from torchvision import transforms

        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        return cls(
            model,
            name=model_name,
            revision=revision,
            device=device,
            transform=transform,
            input_size=input_size,
        )

    @classmethod
    def from_conch(
        cls,
        *,
        model_id: str = "MahmoodLab/TITAN",
        token: str | None = None,
        revision: str = "unknown",
        device: Any = None,
    ) -> "HookedViTTileTeacherAdapter":
        """Load CONCH v1.5 through TITAN's ``return_conch()`` accessor.

        The returned object is ``EncoderWithAttentionalPooler`` (``conch.trunk`` is the
        24-block ViT-L/16, input 448x448, ``num_prefix_tokens == 1``). It exposes a
        plain ``forward(x)`` — there is no ``encode_image(...)`` method on this
        checkpoint despite that being a common CONCH/CLIP-style API name; calling it
        raises ``AttributeError`` immediately rather than silently doing nothing.
        """
        from transformers import AutoModel

        titan = AutoModel.from_pretrained(model_id, trust_remote_code=True, token=token)
        conch, transform = titan.return_conch()
        input_size = _infer_crop_size(transform, default=448)
        return cls(
            conch,
            name="conch_v15",
            revision=revision,
            device=device,
            forward_fn=lambda images: conch(images),
            transform=transform,
            input_size=input_size,
        )

    def _make_final_attn_hook(self, cache: dict[str, Any], final_layer: int):
        """Capture only the CLS attention row without replacing model attention.

        The former implementation replaced the complete attention forward and
        materialized ``[B,H,N,N]`` logits in fp32.  CONCH normally uses fused SDPA;
        keeping the original forward both preserves the encoder's native pooled
        embedding and avoids the multi-GiB temporary.  A pre-hook computes the only
        quantity EAF needs: ``[B,H,1,N]`` for the CLS query.
        """
        prefix = self.num_prefix_tokens

        def hook(attn_module, inputs):
            import torch

            x = inputs[0]
            for required in ("qkv", "scale", "num_heads", "head_dim"):
                if not hasattr(attn_module, required):
                    raise RuntimeError(
                        f"Attention module at block {final_layer} is missing `{required}`; "
                        "HookedViTTileTeacherAdapter only supports timm-style Attention"
                    )
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(
                B, N, 3, attn_module.num_heads, attn_module.head_dim
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q_norm = getattr(attn_module, "q_norm", None)
            k_norm = getattr(attn_module, "k_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            if k_norm is not None:
                k = k_norm(k)
            # Softmax rows are independent, so computing only query zero is
            # numerically equivalent to slicing query zero from the full matrix.
            cls_logits = ((q[:, :, :1] @ k.transpose(-2, -1)) * attn_module.scale).float()
            cls_attention = cls_logits.softmax(-1)
            target = cls_attention[:, :, 0, prefix:].mean(1)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            # Keep it device-local until the model forward finishes; copying here
            # would synchronize CUDA before the original fused attention runs.
            cache["final_attention"] = target.detach()

        return hook

    def extract_final(self, images: Any) -> TileTeacherFinalOutput:
        """Full forward: the final CLS-to-patch attention target + final pooled
        embedding -- the only two quantities written to the permanent cache. Does
        NOT touch/allocate ``early_tokens`` at all (no hook installed for it), unlike
        the combined ``extract()`` used only for testing/inspection."""
        import torch

        blocks = self._blocks
        final_layer = len(blocks) - 1
        cache: dict[str, Any] = {}
        attention_handle = blocks[final_layer].attn.register_forward_pre_hook(
            self._make_final_attn_hook(cache, final_layer)
        )
        try:
            with torch.inference_mode():
                final_output = self._forward_fn(images)
        finally:
            attention_handle.remove()

        if "final_attention" not in cache:
            raise RuntimeError(
                f"Tile-teacher final-attention hook did not fire; final_layer={final_layer}, "
                f"n_blocks={len(blocks)}"
            )
        if isinstance(final_output, (tuple, list)):
            final_output = final_output[0]
        if final_output.ndim == 3:
            final_output = final_output[:, 0, :]
        tile_embeddings = final_output.detach().to("cpu", dtype=torch.float16)
        return TileTeacherFinalOutput(
            final_attention=cache["final_attention"].to("cpu", dtype=torch.float16),
            tile_embeddings=tile_embeddings,
        )

    def extract_early(self, images: Any, *, early_layer: int) -> Any:
        """Cheap partial forward: run through block ``early_layer`` only, then abort
        before any later block, the final norm, or the pooling head ever execute --
        an early exit raised from inside the block's own ``forward_hook``, which
        propagates up through the encoder's call stack. Meant to be called online,
        every EAF Tile training step, instead of reading a cached ``early_tokens``
        array (which is no longer persisted -- see docs/offline_eaf_pipeline.md).

        Returned tensor stays on its compute device/dtype (no forced CPU/fp16 cast,
        unlike extract_final): this is a live teacher signal feeding straight into a
        training step, not a value being written to disk.
        """
        import torch

        blocks = self._blocks
        if not 0 <= early_layer < len(blocks):
            raise ValueError(f"early_layer={early_layer} out of range for {len(blocks)} blocks")
        prefix = self.num_prefix_tokens
        cache: dict[str, torch.Tensor] = {}

        def early_exit_hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Block {early_layer} returned unsupported type {type(output)!r}; "
                    "HookedViTTileTeacherAdapter only supports plain-tensor block outputs"
                )
            cache["early_tokens"] = output[:, prefix:].detach()
            raise _EarlyExit()

        handle = blocks[early_layer].register_forward_hook(early_exit_hook)
        try:
            with torch.inference_mode():
                try:
                    self._forward_fn(images)
                except _EarlyExit:
                    pass
        finally:
            handle.remove()

        if "early_tokens" not in cache:
            raise RuntimeError(
                f"Tile-teacher early-exit hook did not fire; early_layer={early_layer}, "
                f"n_blocks={len(blocks)}"
            )
        return cache["early_tokens"]

    def extract(self, images: Any, *, early_layer: int) -> TileTeacherOutput:
        """Combined early+final extraction in one full forward. Used only for
        numeric-equivalence testing (proving ``extract_early``/``extract_final``
        individually reproduce the same values as a single combined pass) and ad-hoc
        inspection -- production code paths call the split methods above instead."""
        import torch

        blocks = self._blocks
        final_layer = len(blocks) - 1
        if not 0 <= early_layer <= final_layer:
            raise ValueError(f"early_layer={early_layer} out of range for {len(blocks)} blocks")
        prefix = self.num_prefix_tokens

        cache: dict[str, Any] = {}

        def source_hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Block {early_layer} returned unsupported type {type(output)!r}; "
                    "HookedViTTileTeacherAdapter only supports plain-tensor block outputs"
                )
            cache["early_tokens"] = output[:, prefix:].detach().to("cpu", dtype=torch.float16)

        source_handle = blocks[early_layer].register_forward_hook(source_hook)
        attention_handle = blocks[final_layer].attn.register_forward_pre_hook(
            self._make_final_attn_hook(cache, final_layer)
        )
        try:
            with torch.inference_mode():
                final_output = self._forward_fn(images)
        finally:
            source_handle.remove()
            attention_handle.remove()

        if "early_tokens" not in cache or "final_attention" not in cache:
            raise RuntimeError(
                "Tile-teacher hook did not fire for the requested layer(s); "
                f"early_layer={early_layer}, final_layer={final_layer}, n_blocks={len(blocks)}"
            )

        if isinstance(final_output, (tuple, list)):
            final_output = final_output[0]
        if final_output.ndim == 3:
            final_output = final_output[:, 0, :]
        tile_embeddings = final_output.detach().to("cpu", dtype=torch.float16)

        early = cache["early_tokens"]
        target = cache["final_attention"].to("cpu", dtype=torch.float16)
        if early.shape[:2] != target.shape:
            raise RuntimeError(
                f"early_tokens/final_attention token-count mismatch: "
                f"{tuple(early.shape)} vs {tuple(target.shape)}"
            )

        return TileTeacherOutput(
            early_tokens=early,
            final_attention=target,
            tile_embeddings=tile_embeddings,
        )


def _infer_crop_size(transform: Any, *, default: int) -> int:
    """Best-effort read of a torchvision Compose's Resize/CenterCrop target size."""
    from torchvision import transforms as T

    steps = getattr(transform, "transforms", None)
    if steps is None:
        return default
    for step in reversed(steps):
        if isinstance(step, (T.CenterCrop, T.Resize)):
            size = step.size
            if isinstance(size, (tuple, list)):
                return int(size[0])
            if isinstance(size, int):
                return size
    return default


class TitanWSITeacherAdapter:
    """WSI teacher wrapping the existing ``TitanAdapter`` (real post-softmax attention).

    ``tile_scores`` defaults to ``global_to_tiles_mass_share``: TITAN's global/CLS-token
    attention to every other vision token, mass-normalized and mapped from TITAN's own
    token grid down to one value per input tile (see
    ``src/wsi_pipeline/wsi_models/titan_attention.py``). This is exactly the kind of
    "documented importance projection" the WSI-EAF cache contract asks for, reusing the
    already-implemented, tested TITAN attention capture rather than duplicating it.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        tile_score_key: str = "global_to_tiles_mass_share",
        global_token_index: int = 0,
        patch_size_level0: int = 512,
        revision: str = "unknown",
        device: Any = None,
        store_raw_attention: bool = False,
    ) -> None:
        from .wsi_models.titan import TitanAdapter

        modes: tuple[str, ...] = ("global_to_tokens", "received", "rollout")
        if store_raw_attention:
            modes = modes + ("full",)
        self._adapter = TitanAdapter(
            token=token, attention_modes=modes, global_token_index=global_token_index
        ).eval()
        if device is not None:
            self._adapter.to(device)
        self.name = TitanAdapter.name
        self.revision = revision
        self.tile_score_key = tile_score_key
        self.patch_size_level0 = patch_size_level0
        self.store_raw_attention = store_raw_attention

    def extract(self, tile_embeddings: Any, coords: Any) -> WSITeacherOutput:
        import torch

        result = self._adapter.encode(tile_embeddings, coords, self.patch_size_level0)
        n_tiles = tile_embeddings.shape[0]
        scores = self._tile_scores(result.attention, n_tiles=n_tiles)
        embedding = result.slide_embedding
        while embedding.ndim > 1 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        raw_attention = None
        if self.store_raw_attention:
            full_keys = sorted(k for k in result.attention if k.startswith("full_layer_"))
            if full_keys:
                raw_attention = result.attention[full_keys[-1]].to(torch.float16)
        return WSITeacherOutput(
            tile_scores=scores.to(torch.float16),
            wsi_embedding=embedding.to(torch.float16),
            raw_attention=raw_attention,
        )

    def _tile_scores(self, attention: Any, *, n_tiles: int) -> Any:
        key = self.tile_score_key
        if key not in attention:
            # Fall back to the raw token-level tensor only if TITAN's vision tokens
            # already map 1:1 onto input tiles (no tile-to-token reduction needed).
            if "global_to_tokens" in attention and attention["global_to_tokens"].shape[-1] == n_tiles:
                key = "global_to_tokens"
            else:
                raise KeyError(
                    f"TITAN attention key {self.tile_score_key!r} unavailable and no 1:1 "
                    f"token/tile fallback applies; available keys={sorted(attention)}"
                )
        value = attention[key]
        while value.ndim > 1:
            value = value.mean(dim=0)
        return value


class FeatherWSITeacherAdapter:
    """WSI teacher wrapping the existing FEATHER (gated ABMIL) adapter.

    FEATHER's instance-attention softmax probabilities are already a single
    canonical importance value per tile, so no reduction/projection is needed.
    """

    def __init__(
        self,
        *,
        model_id: str = "MahmoodLab/abmil.base.conch_v15.pc108-24k",
        token: str | None = None,
        revision: str = "unknown",
        device: Any = None,
    ) -> None:
        from .wsi_models.feather import FeatherABMILAdapter

        self._adapter = FeatherABMILAdapter(model_id=model_id, token=token).eval()
        if device is not None:
            self._adapter.to(device)
        self.name = model_id
        self.revision = revision

    def extract(self, tile_embeddings: Any, coords: Any) -> WSITeacherOutput:
        result = self._adapter.encode(tile_embeddings, coords, patch_size_level0=0)
        scores = result.attention["probs"]
        while scores.ndim > 1:
            scores = scores.squeeze(0)
        embedding = result.slide_embedding
        while embedding.ndim > 1 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        return WSITeacherOutput(tile_scores=scores.half(), wsi_embedding=embedding.half())


class GigaPathWSITeacherAdapter:
    """WSI teacher wrapping Prov-GigaPath, which exposes no native attention.

    ``tile_scores`` falls back to a documented, deterministic cosine-similarity
    between each tile embedding and the resulting slide embedding. This is an
    explicit importance projection, not a fabricated attention value; adapters for
    models with native attention (TITAN, FEATHER) never use this path.
    """

    def __init__(self, *, model_path: str = "hf_hub:prov-gigapath/prov-gigapath", revision: str = "unknown", device: Any = None) -> None:
        from .wsi_models.gigapath import ProvGigaPathAdapter

        self._adapter = ProvGigaPathAdapter(model_path=model_path).eval()
        if device is not None:
            self._adapter.to(device)
        self.name = "prov-gigapath"
        self.revision = revision

    def extract(self, tile_embeddings: Any, coords: Any) -> WSITeacherOutput:
        import torch
        import torch.nn.functional as F

        result = self._adapter.encode(tile_embeddings, coords, patch_size_level0=0)
        embedding = result.slide_embedding
        while embedding.ndim > 1 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        tiles = torch.as_tensor(tile_embeddings, dtype=torch.float32)
        slide = embedding.to(torch.float32)
        if tiles.shape[-1] != slide.shape[-1]:
            raise RuntimeError(
                "GigaPath tile_scores fallback requires matching tile/slide embedding "
                f"dims; got tile={tiles.shape[-1]} slide={slide.shape[-1]}."
            )
        scores = F.cosine_similarity(tiles, slide.unsqueeze(0).expand_as(tiles), dim=-1)
        return WSITeacherOutput(tile_scores=scores.to(torch.float16), wsi_embedding=embedding.to(torch.float16))
