import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_forecaster_csv(path: Path) -> None:
    _write_csv(
        path,
        [
            "keep_ratio",
            "n_slides",
            "n_tiles_total",
            "n_tiles_kept_total",
            "mean_n_tiles",
            "mean_n_tiles_kept",
            "mean_effective_keep_ratio",
            "mean_spearmanr",
            "mean_topk_overlap",
            "mean_ndcg_at_k",
            "mean_attention_mass_retained",
            "mean_oracle_attention_mass_at_k",
            "mean_relative_attention_mass_retained",
        ],
        [
            {
                "keep_ratio": 0.25,
                "n_slides": 5,
                "n_tiles_total": 100,
                "n_tiles_kept_total": 25,
                "mean_n_tiles": 20,
                "mean_n_tiles_kept": 5,
                "mean_effective_keep_ratio": 0.25,
                "mean_spearmanr": 0.4,
                "mean_topk_overlap": 0.5,
                "mean_ndcg_at_k": 0.6,
                "mean_attention_mass_retained": 0.55,
                "mean_oracle_attention_mass_at_k": 0.8,
                "mean_relative_attention_mass_retained": 0.6875,
            },
            {
                "keep_ratio": 0.5,
                "n_slides": 5,
                "n_tiles_total": 100,
                "n_tiles_kept_total": 50,
                "mean_n_tiles": 20,
                "mean_n_tiles_kept": 10,
                "mean_effective_keep_ratio": 0.5,
                "mean_spearmanr": 0.45,
                "mean_topk_overlap": 0.6,
                "mean_ndcg_at_k": 0.7,
                "mean_attention_mass_retained": 0.75,
                "mean_oracle_attention_mass_at_k": 0.9,
                "mean_relative_attention_mass_retained": 0.833333,
            },
        ],
    )


def _write_agreement_csv(path: Path) -> None:
    _write_csv(
        path,
        [
            "keep_ratio",
            "n_slides",
            "n_tiles_total",
            "n_tiles_kept_total",
            "mean_n_tiles",
            "mean_n_tiles_kept",
            "mean_effective_keep_ratio",
            "full_accuracy",
            "pruned_accuracy",
            "prediction_agreement",
            "mean_logit_cosine_similarity",
            "mean_prob_kl_full_to_pruned",
            "mean_attention_mass_retained",
            "mean_oracle_attention_mass_at_k",
            "mean_relative_attention_mass_retained",
        ],
        [
            {
                "keep_ratio": 0.25,
                "n_slides": 5,
                "n_tiles_total": 100,
                "n_tiles_kept_total": 25,
                "mean_n_tiles": 20,
                "mean_n_tiles_kept": 5,
                "mean_effective_keep_ratio": 0.25,
                "full_accuracy": 0.8,
                "pruned_accuracy": 0.6,
                "prediction_agreement": 0.7,
                "mean_logit_cosine_similarity": 0.75,
                "mean_prob_kl_full_to_pruned": 0.2,
                "mean_attention_mass_retained": 0.55,
                "mean_oracle_attention_mass_at_k": 0.8,
                "mean_relative_attention_mass_retained": 0.6875,
            },
            {
                "keep_ratio": 0.5,
                "n_slides": 5,
                "n_tiles_total": 100,
                "n_tiles_kept_total": 50,
                "mean_n_tiles": 20,
                "mean_n_tiles_kept": 10,
                "mean_effective_keep_ratio": 0.5,
                "full_accuracy": 0.8,
                "pruned_accuracy": 0.75,
                "prediction_agreement": 0.9,
                "mean_logit_cosine_similarity": 0.9,
                "mean_prob_kl_full_to_pruned": 0.05,
                "mean_attention_mass_retained": 0.75,
                "mean_oracle_attention_mass_at_k": 0.9,
                "mean_relative_attention_mass_retained": 0.833333,
            },
        ],
    )


def test_plot_wsi_pruning_curves_cli_writes_pngs_and_summary(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    forecaster_csv = tmp_path / "forecaster.csv"
    agreement_csv = tmp_path / "agreement.csv"
    output_dir = tmp_path / "report"

    _write_forecaster_csv(forecaster_csv)
    _write_agreement_csv(agreement_csv)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_wsi_pruning_curves.py",
            "--forecaster-pruning-csv",
            str(forecaster_csv),
            "--abmil-agreement-csv",
            str(agreement_csv),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"event": "done"' in result.stdout

    expected_files = [
        "attention_mass_retained.png",
        "relative_attention_mass_retained.png",
        "ranking_quality.png",
        "prediction_agreement.png",
        "pruned_accuracy.png",
        "kl_full_to_pruned.png",
        "summary.json",
    ]

    for filename in expected_files:
        path = output_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["forecaster_pruning_csv"] == str(forecaster_csv)
    assert summary["abmil_agreement_csv"] == str(agreement_csv)
    assert summary["keep_ratios"] == [0.25, 0.5]
    assert summary["best"]["max_prediction_agreement"]["keep_ratio"] == 0.5
    assert summary["best"]["min_kl_full_to_pruned"]["keep_ratio"] == 0.5


def test_plot_wsi_pruning_curves_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    forecaster_csv = tmp_path / "forecaster.csv"
    agreement_csv = tmp_path / "agreement.csv"
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text("{}")

    _write_forecaster_csv(forecaster_csv)
    _write_agreement_csv(agreement_csv)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_wsi_pruning_curves.py",
            "--forecaster-pruning-csv",
            str(forecaster_csv),
            "--abmil-agreement-csv",
            str(agreement_csv),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output files already exist" in result.stderr


def test_plot_wsi_pruning_curves_cli_overwrites_when_requested(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    forecaster_csv = tmp_path / "forecaster.csv"
    agreement_csv = tmp_path / "agreement.csv"
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text("{}")

    _write_forecaster_csv(forecaster_csv)
    _write_agreement_csv(agreement_csv)

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_wsi_pruning_curves.py",
            "--forecaster-pruning-csv",
            str(forecaster_csv),
            "--abmil-agreement-csv",
            str(agreement_csv),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["keep_ratios"] == [0.25, 0.5]
