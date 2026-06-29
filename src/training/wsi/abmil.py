"""Training utilities for WSI-level ABMIL classifiers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.wsi import PaddedWSIBatch
from src.models.wsi import ABMILOutput


@dataclass(frozen=True)
class ABMILClassificationBatchOutput:
    """Output of one ABMIL classification batch pass."""

    loss: torch.Tensor
    logits: torch.Tensor
    attention: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class ABMILClassificationEpochOutput:
    """Aggregated output of one ABMIL classification epoch."""

    loss: float
    metrics: dict[str, float]
    n_batches: int
    n_bags: int


def _labels_to_long_tensor(
    labels: tuple[int | float | torch.Tensor | None, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not labels:
        raise ValueError("batch.labels are required for ABMIL classification.")

    parsed: list[int] = []

    for index, label in enumerate(labels):
        if label is None:
            raise ValueError(f"label at batch index {index} is missing.")

        if isinstance(label, bool):
            raise TypeError("boolean labels are not supported for classification.")

        if isinstance(label, int):
            parsed.append(label)
            continue

        if isinstance(label, float):
            if not label.is_integer():
                raise TypeError(
                    "float labels must represent integer class ids; "
                    f"got {label} at batch index {index}."
                )
            parsed.append(int(label))
            continue

        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                raise ValueError(
                    "tensor labels must contain exactly one element for classification; "
                    f"got shape {tuple(label.shape)} at batch index {index}."
                )
            value = label.detach().cpu().item()
            if isinstance(value, bool):
                raise TypeError("boolean tensor labels are not supported.")
            if isinstance(value, float) and not float(value).is_integer():
                raise TypeError(
                    "tensor labels must represent integer class ids; "
                    f"got {value} at batch index {index}."
                )
            parsed.append(int(value))
            continue

        raise TypeError(
            "labels must be int, integer-valued float, scalar tensor, or None; "
            f"got {type(label).__name__} at batch index {index}."
        )

    label_tensor = torch.tensor(parsed, dtype=torch.long, device=device)
    if (label_tensor < 0).any():
        raise ValueError("classification labels must be non-negative.")

    return label_tensor


def run_abmil_classification_batch(
    model: nn.Module,
    batch: PaddedWSIBatch,
) -> ABMILClassificationBatchOutput:
    """Run ABMIL forward, cross-entropy loss, and accuracy for one batch."""

    if not isinstance(batch, PaddedWSIBatch):
        raise TypeError(
            "batch must be a PaddedWSIBatch; "
            f"got {type(batch).__name__}."
        )

    output: ABMILOutput = model(batch.tile_features, mask=batch.mask)
    labels = _labels_to_long_tensor(batch.labels, device=output.logits.device)

    if output.logits.ndim != 2:
        raise ValueError(
            "ABMIL batched logits must have shape [batch, n_classes]; "
            f"got {tuple(output.logits.shape)}."
        )
    if output.logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "logits batch size must match labels length; "
            f"got {output.logits.shape[0]} and {labels.shape[0]}."
        )

    loss = F.cross_entropy(output.logits, labels)

    with torch.no_grad():
        predictions = output.logits.detach().argmax(dim=1)
        accuracy = (predictions == labels).to(torch.float32).mean()
        metrics = {"accuracy": accuracy}

    return ABMILClassificationBatchOutput(
        loss=loss,
        logits=output.logits,
        attention=output.attention,
        metrics=metrics,
    )


def train_abmil_classification_epoch(
    model: nn.Module,
    loader: Iterable[PaddedWSIBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str | None = None,
    grad_clip_norm: float | None = None,
) -> ABMILClassificationEpochOutput:
    """Train an ABMIL classifier for one epoch."""

    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when provided.")

    model.train()

    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    n_batches = 0
    n_bags = 0

    for batch in loader:
        if not isinstance(batch, PaddedWSIBatch):
            raise TypeError(
                "loader must yield PaddedWSIBatch objects; "
                f"got {type(batch).__name__}."
            )

        batch = batch.to(device=device) if device is not None else batch

        optimizer.zero_grad(set_to_none=True)
        output = run_abmil_classification_batch(model, batch)
        output.loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_weight = batch.n_bags
        total_loss += float(output.loss.detach().cpu()) * batch_weight
        for name, value in output.metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value.cpu()) * batch_weight

        n_batches += 1
        n_bags += batch_weight

    if n_batches == 0 or n_bags == 0:
        raise ValueError("loader yielded no batches.")

    return ABMILClassificationEpochOutput(
        loss=total_loss / n_bags,
        metrics={name: value / n_bags for name, value in metric_sums.items()},
        n_batches=n_batches,
        n_bags=n_bags,
    )


@torch.no_grad()
def evaluate_abmil_classification_epoch(
    model: nn.Module,
    loader: Iterable[PaddedWSIBatch],
    *,
    device: torch.device | str | None = None,
) -> ABMILClassificationEpochOutput:
    """Evaluate an ABMIL classifier for one epoch."""

    was_training = model.training
    model.eval()

    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    n_batches = 0
    n_bags = 0

    for batch in loader:
        if not isinstance(batch, PaddedWSIBatch):
            raise TypeError(
                "loader must yield PaddedWSIBatch objects; "
                f"got {type(batch).__name__}."
            )

        batch = batch.to(device=device) if device is not None else batch
        output = run_abmil_classification_batch(model, batch)

        batch_weight = batch.n_bags
        total_loss += float(output.loss.cpu()) * batch_weight
        for name, value in output.metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value.cpu()) * batch_weight

        n_batches += 1
        n_bags += batch_weight

    if was_training:
        model.train()

    if n_batches == 0 or n_bags == 0:
        raise ValueError("loader yielded no batches.")

    return ABMILClassificationEpochOutput(
        loss=total_loss / n_bags,
        metrics={name: value / n_bags for name, value in metric_sums.items()},
        n_batches=n_batches,
        n_bags=n_bags,
    )
