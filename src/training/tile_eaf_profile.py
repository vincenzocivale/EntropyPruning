"""Portable summaries for Tile-EAF timing profiles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_PHASES = ("data_wait", "teacher", "forecaster", "backward", "metrics")


def summarize_tile_eaf_timing(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one train/validation timing payload for JSON reporting.

    ``elapsed_seconds`` is wall-clock time for the whole epoch.  The phase
    timings deliberately do not have to add up to it: data collation, host to
    device transfer, logging, and Python bookkeeping are reported as
    ``unattributed_seconds`` instead of being silently lost.
    """
    elapsed = max(float(metrics.get("elapsed_seconds", 0.0)), 0.0)
    tiles = max(float(metrics.get("tiles", 0.0)), 0.0)
    raw_phases = metrics.get("phase_seconds", {})
    phases = {
        name: max(float(raw_phases.get(name, 0.0)), 0.0)
        for name in _PHASES
    }
    phase_total = sum(phases.values())
    bottleneck = max(phases, key=phases.__getitem__) if phases else None
    return {
        "elapsed_seconds": elapsed,
        "tiles": tiles,
        "tiles_per_second": tiles / elapsed if elapsed else 0.0,
        "phase_seconds": phases,
        "phase_share": {
            name: seconds / elapsed if elapsed else 0.0
            for name, seconds in phases.items()
        },
        "unattributed_seconds": max(elapsed - phase_total, 0.0),
        "unattributed_share": max(elapsed - phase_total, 0.0) / elapsed if elapsed else 0.0,
        "bottleneck": bottleneck,
    }


def build_tile_eaf_profile(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a comparable, self-describing profile report from epoch history."""
    epochs = []
    for row in history:
        epochs.append(
            {
                "epoch": int(row["epoch"]),
                "train": summarize_tile_eaf_timing(row["train"]),
                "val": summarize_tile_eaf_timing(row["val"]),
            }
        )

    aggregate: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        elapsed = sum(float(row[split]["elapsed_seconds"]) for row in epochs)
        tiles = sum(float(row[split]["tiles"]) for row in epochs)
        phases = {
            name: sum(float(row[split]["phase_seconds"][name]) for row in epochs)
            for name in _PHASES
        }
        phase_total = sum(phases.values())
        bottleneck = max(phases, key=phases.__getitem__)
        aggregate[split] = {
            "elapsed_seconds": elapsed,
            "tiles": tiles,
            "tiles_per_second": tiles / elapsed if elapsed else 0.0,
            "phase_seconds": phases,
            "phase_share": {
                name: seconds / elapsed if elapsed else 0.0
                for name, seconds in phases.items()
            },
            "unattributed_seconds": max(elapsed - phase_total, 0.0),
            "bottleneck": bottleneck,
        }

    train = aggregate["train"]
    dominant = train["bottleneck"]
    share = float(train["phase_share"][dominant])
    if share >= 0.5 and dominant == "data_wait":
        recommendation = "I/O-bound: tune DataLoader/OpenSlide before model changes."
    elif share >= 0.5 and dominant == "teacher":
        recommendation = "Teacher-bound: benchmark batch size and source-layer ablations."
    elif share >= 0.5 and dominant in {"forecaster", "backward"}:
        recommendation = "Optimization-bound: benchmark batch size and torch.compile."
    else:
        recommendation = "Mixed workload: compare end-to-end throughput across one tuning axis at a time."

    return {
        "format": "tile_eaf_profile_v2",
        "profile_overhead": "CUDA synchronization is enabled; do not treat this as production throughput.",
        "epochs": epochs,
        "aggregate": aggregate,
        "recommendation": recommendation,
    }
