from __future__ import annotations

import torch

from .base import WSIModelAdapter, WSIModelOutput


class ProvGigaPathAdapter(WSIModelAdapter):
    """Prov-GigaPath slide encoder adapter for its native 1536-d tile features."""

    name = "prov-gigapath"
    required_feature_key = "final"
    required_feature_dim = 1536
    native_attention = False

    def __init__(self, *, model_path: str = "hf_hub:prov-gigapath/prov-gigapath") -> None:
        try:
            from gigapath.slide_encoder import create_model
        except ImportError as exc:
            raise RuntimeError("Install the official prov-gigapath package to use this adapter") from exc
        self.model = create_model(model_path, "gigapath_slide_enc12l768d", 1536)

    def to(self, device: torch.device) -> "ProvGigaPathAdapter":
        self.model = self.model.to(device)
        return self

    def eval(self) -> "ProvGigaPathAdapter":
        self.model.eval()
        return self

    def encode(self, features: torch.Tensor, coords: torch.Tensor, patch_size_level0: int) -> WSIModelOutput:
        del patch_size_level0
        if features.shape[1] != self.required_feature_dim:
            raise ValueError(f"Prov-GigaPath expects D=1536, got {features.shape[1]}")
        with torch.inference_mode():
            embedding = self.model(features, coords)
        if isinstance(embedding, (tuple, list)):
            embedding = embedding[0]
        return WSIModelOutput(
            slide_embedding=embedding,
            metadata={"attention_kind": "unavailable_in_public_api", "attention_is_native": False},
        )
