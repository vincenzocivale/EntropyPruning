#!/usr/bin/env python
"""Plot WSI pruning evaluation curves from CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


FORECASTER_REQUIRED_COLUMNS = {
    "keep_ratio",
    "mean_spearmanr",
    "mean_topk_overlap",
    "mean_ndcg_at_k",
    "mean_attention_mass_retained",
    "mean_relative_attention_mass_retained",
}

AGREEMENT_REQUIRED_COLUMNS = {
    "keep_ratio",
    "full_accuracy",
    "pruned_accuracy",
    "prediction_agreement",
    "mean_prob_kl_full_to_pruned",
    "mean_attention_mass_retained",
    "mean_relative_attention_mass_retained",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WSI pruning curves from evaluation CSV files."
    )

    parser.add_argument("--forecaster-pruning-csv", type=Path, required=True)
    parser.add_argument("--abmil-agreement-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.forecaster_pruning_csv.exists():
        raise FileNotFoundError(
            f"forecaster pruning CSV not found: {args.forecaster_pruning_csv}"
        )
    if not args.abmil_agreement_csv.exists():
        raise FileNotFoundError(
            f"ABMIL agreement CSV not found: {args.abmil_agreement_csv}"
        )

    output_files = [
        args.output_dir / "attention_mass_retained.png",
        args.output_dir / "relative_attention_mass_retained.png",
        args.output_dir / "ranking_quality.png",
        args.output_dir / "prediction_agreement.png",
        args.output_dir / "pruned_accuracy.png",
        args.output_dir / "kl_full_to_pruned.png",
        args.output_dir / "summary.json",
    ]

    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing[:5])
        )


def _read_numeric_csv(path: Path, *, required_columns: set[str]) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV {path} is missing required columns: "
                + ", ".join(sorted(missing))
            )

        rows = []
        for row_index, raw_row in enumerate(reader, start=2):
            parsed: dict[str, float] = {}
            for key, value in raw_row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"CSV {path} has non-numeric value at line {row_index}, "
                        f"column {key!r}: {value!r}"
                    ) from exc
            rows.append(parsed)

    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")

    rows.sort(key=lambda row: row["keep_ratio"])
    return rows


def _series(rows: list[dict[str, float]], column: str) -> tuple[list[float], list[float]]:
    return (
        [row["keep_ratio"] for row in rows],
        [row[column] for row in rows],
    )


def _plot_lines(
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    lines: list[tuple[str, list[dict[str, float]], str]],
) -> None:
    plt.figure(figsize=(7, 5))

    for label, rows, column in lines:
        x_values, y_values = _series(rows, column)
        plt.plot(x_values, y_values, marker="o", label=label)

    plt.xlabel("Keep ratio")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _best_by(
    rows: list[dict[str, float]],
    *,
    column: str,
    maximize: bool = True,
) -> dict[str, float]:
    return dict(
        max(rows, key=lambda row: row[column])
        if maximize
        else min(rows, key=lambda row: row[column])
    )


def _write_summary(
    path: Path,
    *,
    forecaster_csv: Path,
    agreement_csv: Path,
    forecaster_rows: list[dict[str, float]],
    agreement_rows: list[dict[str, float]],
) -> None:
    summary: dict[str, Any] = {
        "forecaster_pruning_csv": str(forecaster_csv),
        "abmil_agreement_csv": str(agreement_csv),
        "keep_ratios": [row["keep_ratio"] for row in agreement_rows],
        "best": {
            "max_attention_mass_retained": _best_by(
                forecaster_rows,
                column="mean_attention_mass_retained",
                maximize=True,
            ),
            "max_relative_attention_mass_retained": _best_by(
                forecaster_rows,
                column="mean_relative_attention_mass_retained",
                maximize=True,
            ),
            "max_prediction_agreement": _best_by(
                agreement_rows,
                column="prediction_agreement",
                maximize=True,
            ),
            "max_pruned_accuracy": _best_by(
                agreement_rows,
                column="pruned_accuracy",
                maximize=True,
            ),
            "min_kl_full_to_pruned": _best_by(
                agreement_rows,
                column="mean_prob_kl_full_to_pruned",
                maximize=False,
            ),
        },
        "forecaster_rows": forecaster_rows,
        "agreement_rows": agreement_rows,
    }

    path.write_text(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecaster_rows = _read_numeric_csv(
        args.forecaster_pruning_csv,
        required_columns=FORECASTER_REQUIRED_COLUMNS,
    )
    agreement_rows = _read_numeric_csv(
        args.abmil_agreement_csv,
        required_columns=AGREEMENT_REQUIRED_COLUMNS,
    )

    _plot_lines(
        output_path=args.output_dir / "attention_mass_retained.png",
        title="WSI pruning: retained teacher attention mass",
        ylabel="Attention mass retained",
        lines=[
            ("Forecaster pruning", forecaster_rows, "mean_attention_mass_retained"),
            ("ABMIL agreement eval", agreement_rows, "mean_attention_mass_retained"),
        ],
    )

    _plot_lines(
        output_path=args.output_dir / "relative_attention_mass_retained.png",
        title="WSI pruning: retained mass relative to oracle top-k",
        ylabel="Relative retained attention mass",
        lines=[
            (
                "Forecaster pruning",
                forecaster_rows,
                "mean_relative_attention_mass_retained",
            ),
            (
                "ABMIL agreement eval",
                agreement_rows,
                "mean_relative_attention_mass_retained",
            ),
        ],
    )

    _plot_lines(
        output_path=args.output_dir / "ranking_quality.png",
        title="WSI pruning: ranking quality",
        ylabel="Metric value",
        lines=[
            ("Spearman", forecaster_rows, "mean_spearmanr"),
            ("Top-k overlap", forecaster_rows, "mean_topk_overlap"),
            ("NDCG@k", forecaster_rows, "mean_ndcg_at_k"),
        ],
    )

    _plot_lines(
        output_path=args.output_dir / "prediction_agreement.png",
        title="WSI pruning: ABMIL full-vs-pruned prediction agreement",
        ylabel="Prediction agreement",
        lines=[
            ("Prediction agreement", agreement_rows, "prediction_agreement"),
        ],
    )

    _plot_lines(
        output_path=args.output_dir / "pruned_accuracy.png",
        title="WSI pruning: full vs pruned ABMIL accuracy",
        ylabel="Accuracy",
        lines=[
            ("Full ABMIL accuracy", agreement_rows, "full_accuracy"),
            ("Pruned ABMIL accuracy", agreement_rows, "pruned_accuracy"),
        ],
    )

    _plot_lines(
        output_path=args.output_dir / "kl_full_to_pruned.png",
        title="WSI pruning: KL(full probabilities || pruned probabilities)",
        ylabel="KL divergence",
        lines=[
            ("KL full to pruned", agreement_rows, "mean_prob_kl_full_to_pruned"),
        ],
    )

    _write_summary(
        args.output_dir / "summary.json",
        forecaster_csv=args.forecaster_pruning_csv,
        agreement_csv=args.abmil_agreement_csv,
        forecaster_rows=forecaster_rows,
        agreement_rows=agreement_rows,
    )

    print(
        json.dumps(
            {
                "event": "done",
                "output_dir": str(args.output_dir),
                "n_keep_ratios": len(agreement_rows),
            },
            indent=2,
        ),
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
