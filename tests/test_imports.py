"""Regression guard: the Phase-1/2/3 restructure did not orphan any core import."""


def test_core_imports():
    import src  # noqa: F401
    import src.losses  # noqa: F401
    import src.utils  # noqa: F401
    import src.models  # noqa: F401
    from src.models import (  # noqa: F401
        ThunderBackboneAdapter,
        FrozenBackbone,
        AttentionForecaster,
        GenericLoRAClassifier,
        load_lora_adapted_backbone,
        load_lora_adapted_weights,
        GenericLoRAWithForecasterPruning,
    )
    from src.collection import collect_and_save_dataset  # noqa: F401
    from src.data.h5_dataset import H5ForecastDataset  # noqa: F401
    from src.evaluation import evaluate  # noqa: F401


def test_old_modules_removed():
    """classifier.py (old full-LoRA Stage 1) and last_layer_classifier.py (the
    last-block-only Phase 1 it was replaced with) were both removed.

    Phase 1 is now GenericLoRAClassifier (full-backbone LoRA + head,
    src/models/lora_classifier.py), matching the paper's Stage 1.
    """
    import importlib

    for missing in ("src.models.classifier", "src.models.last_layer_classifier"):
        try:
            importlib.import_module(missing)
            raised = False
        except ModuleNotFoundError:
            raised = True
        assert raised, f"{missing} should have been removed"
