"""WSI-level training utilities."""

from src.training.wsi.abmil import (
    ABMILClassificationBatchOutput,
    ABMILClassificationEpochOutput,
    evaluate_abmil_classification_epoch,
    run_abmil_classification_batch,
    train_abmil_classification_epoch,
)
from src.training.wsi.attention_forecasting import (
    WSIAttentionForecastingBatchOutput,
    WSIAttentionForecastingEpochOutput,
    evaluate_wsi_attention_forecasting_epoch,
    run_wsi_attention_forecasting_batch,
    train_wsi_attention_forecasting_epoch,
)

__all__ = [
    "ABMILClassificationBatchOutput",
    "ABMILClassificationEpochOutput",
    "WSIAttentionForecastingBatchOutput",
    "WSIAttentionForecastingEpochOutput",
    "evaluate_abmil_classification_epoch",
    "evaluate_wsi_attention_forecasting_epoch",
    "run_abmil_classification_batch",
    "run_wsi_attention_forecasting_batch",
    "train_abmil_classification_epoch",
    "train_wsi_attention_forecasting_epoch",
]
