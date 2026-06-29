import torch
from torch.utils.data import DataLoader

from src.data.wsi import InMemoryWSIBagDataset, WSIBag, collate_padded_wsi_bags
from src.models.wsi import WSITileAttentionForecaster
from src.training.wsi import (
    evaluate_wsi_attention_forecasting_epoch,
    train_wsi_attention_forecasting_epoch,
)


def _make_learnable_bag(
    slide_id: str,
    n_tiles: int,
    feature_dim: int,
    generator: torch.Generator,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim, generator=generator)

    # Synthetic target: tiles with larger projection on this fixed direction
    # should receive larger WSI-level attention. The forecaster should be able
    # to learn this ranking from tile_features alone.
    target_direction = torch.linspace(-1.0, 1.0, feature_dim)
    target_logits = tile_features @ target_direction
    attention = torch.softmax(target_logits, dim=0)

    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=torch.zeros(n_tiles, 2, dtype=torch.long),
        label=1,
        attention=attention,
    )


def test_wsi_attention_forecaster_reduces_loss_on_synthetic_learnable_target() -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(123)

    feature_dim = 8
    bags = [
        _make_learnable_bag(
            slide_id=f"slide_{index:03d}",
            n_tiles=5 + (index % 4),
            feature_dim=feature_dim,
            generator=generator,
        )
        for index in range(12)
    ]

    loader = DataLoader(
        InMemoryWSIBagDataset(bags),
        batch_size=4,
        shuffle=True,
        collate_fn=collate_padded_wsi_bags,
    )

    model = WSITileAttentionForecaster(
        feature_dim=feature_dim,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    initial = evaluate_wsi_attention_forecasting_epoch(
        model,
        loader,
        top_k=2,
    )

    for _ in range(25):
        train_wsi_attention_forecasting_epoch(
            model,
            loader,
            optimizer,
            top_k=2,
        )

    final = evaluate_wsi_attention_forecasting_epoch(
        model,
        loader,
        top_k=2,
    )

    assert final.loss < initial.loss
