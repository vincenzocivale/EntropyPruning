from __future__ import annotations

import torch

from .base import WSIModelAdapter, WSIModelOutput
from .titan_attention import TitanAttentionCaptureConfig, capture_titan_attention


class TitanAdapter(WSIModelAdapter):
    """TITAN slide embedding and real post-softmax self-attention adapter.

    Attention is captured from the live gated model during the official
    ``encode_slide_from_patch_features`` call. The adapter does not replace or
    approximate the model forward. It supports both fused SDPA and explicit ViT
    softmax implementations and fails if neither path is observed.
    """

    name = "MahmoodLab/TITAN"
    required_feature_key = "final"
    required_feature_dim = 768
    native_attention = True

    def __init__(
        self,
        *,
        token: str | None = None,
        attention_modes: tuple[str, ...] = ("global_to_tokens", "received", "rollout"),
        global_token_index: int = 0,
        full_attention_layers: tuple[int, ...] = (-1,),
        max_full_attention_tokens: int = 2048,
        max_rollout_tokens: int = 4096,
        strict_attention_capture: bool = True,
        revision: str | None = None,
    ) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            self.name,
            trust_remote_code=True,
            token=token,
            revision=revision,
        )
        self.capture_config = TitanAttentionCaptureConfig(
            modes=attention_modes,
            global_token_index=global_token_index,
            full_layers=full_attention_layers,
            max_full_attention_tokens=max_full_attention_tokens,
            max_rollout_tokens=max_rollout_tokens,
            strict=strict_attention_capture,
        )

    def to(self, device: torch.device) -> "TitanAdapter":
        self.model = self.model.to(device)
        return self

    def eval(self) -> "TitanAdapter":
        self.model.eval()
        return self

    def encode(self, features: torch.Tensor, coords: torch.Tensor, patch_size_level0: int) -> WSIModelOutput:
        if features.shape[1] != self.required_feature_dim:
            raise ValueError(f"TITAN expects D=768, got {features.shape[1]}")
        if self.model.training:
            raise RuntimeError("TITAN attention extraction requires model.eval()")

        def forward() -> torch.Tensor:
            return self.model.encode_slide_from_patch_features(features, coords, patch_size_level0)

        with torch.inference_mode():
            embedding, captured = capture_titan_attention(
                self.model,
                forward,
                coords=coords,
                patch_size_level0=patch_size_level0,
                config=self.capture_config,
            )
        return WSIModelOutput(
            slide_embedding=embedding,
            attention=captured.attention,
            auxiliary=captured.auxiliary,
            metadata=captured.metadata,
        )
