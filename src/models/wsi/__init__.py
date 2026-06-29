"""WSI-level models."""

from src.models.wsi.tile_attention_forecaster import (
    WSITileAttentionForecaster,
    wsi_attention_kl_loss,
)

__all__ = ["WSITileAttentionForecaster", "wsi_attention_kl_loss"]
