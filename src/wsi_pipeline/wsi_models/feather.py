from __future__ import annotations

import torch

from .base import WSIModelAdapter, WSIModelOutput


class FeatherABMILAdapter(WSIModelAdapter):
    """FEATHER ABMIL adapter loaded directly from its Hugging Face checkpoint."""

    native_attention = True
    required_feature_key = "final"

    def __init__(
        self,
        *,
        model_id: str = "MahmoodLab/abmil.base.conch_v15.pc108-24k",
        token: str | None = None,
        required_feature_dim: int = 768,
    ) -> None:
        from transformers import AutoModel

        self.name = model_id
        self.required_feature_dim = required_feature_dim
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token,
        )

    def to(self, device: torch.device) -> "FeatherABMILAdapter":
        self.model = self.model.to(device)
        return self

    def eval(self) -> "FeatherABMILAdapter":
        self.model.eval()
        return self

    def encode(self, features: torch.Tensor, coords: torch.Tensor, patch_size_level0: int) -> WSIModelOutput:
        del coords, patch_size_level0
        if features.ndim != 2:
            raise ValueError(f"FEATHER expects [N,D], got {tuple(features.shape)}")
        if self.required_feature_dim is not None and features.shape[1] != self.required_feature_dim:
            raise ValueError(f"FEATHER expects D={self.required_feature_dim}, got D={features.shape[1]}")
        with torch.inference_mode():
            _results, log_dict = self.model(
                features.unsqueeze(0),
                return_attention=True,
                return_slide_feats=True,
            )
        raw = log_dict["attention"]
        while raw.ndim > 1:
            raw = raw.squeeze(0)
        probs = torch.softmax(raw.float(), dim=-1)
        slide_embedding = log_dict["slide_feats"]
        while slide_embedding.ndim > 1 and slide_embedding.shape[0] == 1:
            slide_embedding = slide_embedding.squeeze(0)
        return WSIModelOutput(
            slide_embedding=slide_embedding,
            attention={"logits": raw, "probs": probs},
            metadata={"attention_kind": "gated_abmil_instance_attention", "attention_is_native": True},
        )
