"""Model-agnostic contracts used by offline EAF teacher extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class TileTeacherOutput:
    """Frozen tile-encoder outputs for one image batch."""

    early_tokens: Any
    final_attention: Any
    tile_embeddings: Any


class TileTeacherAdapter(Protocol):
    """Adapter implemented by CONCH/timm/UNI/Virchow-style tile teachers."""

    name: str
    revision: str

    def extract(self, images: Any, *, early_layer: int) -> TileTeacherOutput:
        """Return early tokens, final attention target and final embedding."""
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


class HookedViTTileTeacherAdapter:
    """Generic tile teacher for CONCH/timm/UNI-style ``timm.layers.Attention`` ViTs.

    Captures, in one forward pass per image batch:

    - ``early_tokens``: patch tokens (CLS excluded) at the requested early block —
      the same quantity ``src/collection/extract_features.py::collect_and_save_dataset``
      captures for the EntropyPruning classifier, generalized into a reusable adapter
      and applied here to WSI tile crops instead of standalone images.
    - ``final_attention``: CLS-to-patch attention at the model's last block, averaged
      over heads (``TileCacheSpec.attention_reduction == "cls_mean_heads"``).
    - ``tile_embeddings``: the final block's CLS/pooled output.

    Block discovery reuses ``src.wsi_pipeline.tile_encoders.hooks.find_transformer_blocks``,
    the same helper the existing ``ConchV15MultiLayerEncoder``/``TimmViTMultiLayerEncoder``
    use, so CONCH v1.5's ``visual.trunk.blocks`` and plain timm ViTs (UNI, UNI2, Virchow,
    H-optimus, ...) are both supported as long as their attention module exposes the
    standard ``qkv``/``scale``/``attn_drop``/``proj``/``proj_drop`` interface (optionally
    ``q_norm``/``k_norm``). Anything else raises immediately instead of silently
    capturing the wrong tensor.
    """

    def __init__(
        self,
        model: Any,
        *,
        name: str = "vit",
        revision: str = "unknown",
        device: Any = None,
        forward_fn: Any = None,
    ) -> None:
        from .tile_encoders.hooks import find_transformer_blocks

        self.model = model.eval()
        if device is not None:
            self.model = self.model.to(device)
        self.name = name
        self.revision = revision
        self._blocks = find_transformer_blocks(self.model)
        # Default forward path is a plain classifier-style `model(images)`; CONCH's
        # CLIP-style wrapper needs `model.encode_image(...)` instead (see `from_conch`).
        self._forward_fn = forward_fn or (lambda images: self.model(images))

    @classmethod
    def from_timm(
        cls,
        model_name: str,
        *,
        pretrained: bool = True,
        revision: str = "unknown",
        device: Any = None,
    ) -> "HookedViTTileTeacherAdapter":
        import timm

        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        return cls(model, name=model_name, revision=revision, device=device)

    @classmethod
    def from_conch(
        cls,
        *,
        model_id: str = "MahmoodLab/TITAN",
        token: str | None = None,
        revision: str = "unknown",
        device: Any = None,
    ) -> "HookedViTTileTeacherAdapter":
        from transformers import AutoModel

        titan = AutoModel.from_pretrained(model_id, trust_remote_code=True, token=token)
        conch, _transform = titan.return_conch()
        return cls(
            conch,
            name="conch_v15",
            revision=revision,
            device=device,
            forward_fn=lambda images: conch.encode_image(images, proj_contrast=False, normalize=False),
        )

    def extract(self, images: Any, *, early_layer: int) -> TileTeacherOutput:
        import types

        import torch

        blocks = self._blocks
        final_layer = len(blocks) - 1
        if not 0 <= early_layer <= final_layer:
            raise ValueError(f"early_layer={early_layer} out of range for {len(blocks)} blocks")

        cache: dict[str, torch.Tensor] = {}

        def make_hook(idx: int):
            def fwd(attn_module, x, *args, **kwargs):
                del args, kwargs
                for required in ("qkv", "scale", "attn_drop", "proj", "proj_drop", "num_heads", "head_dim"):
                    if not hasattr(attn_module, required):
                        raise RuntimeError(
                            f"Attention module at block {idx} is missing `{required}`; "
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
                attn = (q @ k.transpose(-2, -1)) * attn_module.scale
                attn = attn.softmax(-1)
                if idx == early_layer:
                    cache["early_tokens"] = x[:, 1:].detach().to("cpu", dtype=torch.float16)
                if idx == final_layer:
                    cache["final_attention"] = attn[:, :, 0, 1:].mean(1).detach().to("cpu", dtype=torch.float16)
                x = (attn_module.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
                return attn_module.proj_drop(attn_module.proj(x))

            return fwd

        originals: dict[int, Any] = {}
        for idx in {early_layer, final_layer}:
            block = blocks[idx]
            originals[idx] = block.attn.forward
            block.attn.forward = types.MethodType(make_hook(idx), block.attn)
        try:
            with torch.inference_mode():
                final_output = self._forward_fn(images)
        finally:
            for idx, original in originals.items():
                blocks[idx].attn.forward = original

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

        return TileTeacherOutput(
            early_tokens=cache["early_tokens"],
            final_attention=cache["final_attention"],
            tile_embeddings=tile_embeddings,
        )


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
