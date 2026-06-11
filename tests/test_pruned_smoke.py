"""Cheap guard for the (unchanged) pruning forward path. Skips if timm/peft are absent."""

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("peft")

from src.models import (  # noqa: E402
    ThunderBackboneAdapter,
    AttentionForecaster,
    GenericLoRAClassifier,
    GenericLoRAWithForecasterPruning,
)


def test_pruned_forward_shape():
    backbone = timm.create_model("vit_tiny_patch16_224", num_classes=0, pretrained=False)
    adapter = ThunderBackboneAdapter(backbone)
    forecaster = AttentionForecaster(embed_dim=adapter.embed_dim, hidden=64, n_heads=4, n_layers=1)
    for p in forecaster.parameters():
        p.requires_grad_(False)
    model = GenericLoRAWithForecasterPruning(
        backbone=backbone, adapter=adapter, n_classes=3,
        forecaster=forecaster, prune_layer=2, keep_ratio=0.2,
    ).eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3)


def test_phase1_checkpoint_warm_starts_phase3():
    """Phase 1's adapted_state_dict() (LoRA backbone + head) must load into Phase 3's
    GenericLoRAWithForecasterPruning, since both use the same LoRA config and head
    architecture (paper's Stage1 -> Stage3 warm start)."""
    backbone1 = timm.create_model("vit_tiny_patch16_224", num_classes=0, pretrained=False)
    adapter1 = ThunderBackboneAdapter(backbone1)
    phase1 = GenericLoRAClassifier(backbone=backbone1, adapter=adapter1, n_classes=3)

    backbone3 = timm.create_model("vit_tiny_patch16_224", num_classes=0, pretrained=False)
    adapter3 = ThunderBackboneAdapter(backbone3)
    forecaster = AttentionForecaster(embed_dim=adapter3.embed_dim, hidden=64, n_heads=4, n_layers=1)
    for p in forecaster.parameters():
        p.requires_grad_(False)
    phase3 = GenericLoRAWithForecasterPruning(
        backbone=backbone3, adapter=adapter3, n_classes=3,
        forecaster=forecaster, prune_layer=2, keep_ratio=0.2,
    )

    sd = phase1.adapted_state_dict()
    phase3.backbone.load_state_dict(sd["backbone"])
    phase3.head.load_state_dict(sd["head"])
