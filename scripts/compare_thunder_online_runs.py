#!/usr/bin/env python3
"""Compare external-only and THUNDER-trained EAF result files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _group_summary(rows: list[dict], group: str) -> dict:
    selected = [row for row in rows if group == "all" or row["group"] == group]
    if not selected:
        return {"n_datasets": 0}
    return {
        "n_datasets": len(selected),
        "median_delta_kl": statistics.median(row["delta_kl"] for row in selected),
        "median_delta_rho": statistics.median(row["delta_rho"] for row in selected),
        "median_delta_recall_0.10": statistics.median(
            row["delta_recall_0.10"] for row in selected
        ),
        "kl_win_fraction": sum(row["delta_kl"] < 0 for row in selected) / len(selected),
        "rho_win_fraction": sum(row["delta_rho"] > 0 for row in selected) / len(selected),
    }


def compare(baseline: dict, candidate: dict, candidate_name: str) -> dict:
    baseline_metrics = baseline["test"]["per_dataset"]
    candidate_metrics = candidate["test"]["per_dataset"]
    train_datasets = set(candidate["data_plan"].get("train_datasets", []))
    common = sorted(set(baseline_metrics) & set(candidate_metrics))
    if not common:
        raise ValueError(f"No common test datasets for {candidate_name}")
    rows = []
    for dataset in common:
        base = baseline_metrics[dataset]
        cand = candidate_metrics[dataset]
        rows.append({
            "candidate": candidate_name,
            "dataset": dataset,
            "group": "seen" if dataset in train_datasets else "held_out",
            "baseline_kl": base["kl"],
            "candidate_kl": cand["kl"],
            "delta_kl": cand["kl"] - base["kl"],
            "baseline_rho": base["rho"],
            "candidate_rho": cand["rho"],
            "delta_rho": cand["rho"] - base["rho"],
            "baseline_recall_0.10": base["recall_0.10"],
            "candidate_recall_0.10": cand["recall_0.10"],
            "delta_recall_0.10": cand["recall_0.10"] - base["recall_0.10"],
        })
    return {
        "candidate": candidate_name,
        "rows": rows,
        "summary": {
            group: _group_summary(rows, group)
            for group in ("all", "seen", "held_out")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-only", required=True)
    parser.add_argument(
        "--candidate", action="append", nargs=2, metavar=("NAME", "RESULTS_JSON"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load(args.external_only)
    comparisons = [
        compare(baseline, _load(path), name) for name, path in args.candidate
    ]
    payload = {
        "external_only": args.external_only,
        "comparisons": comparisons,
    }
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2))
    rows = [row for comparison in comparisons for row in comparison["rows"]]
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({c["candidate"]: c["summary"] for c in comparisons}, indent=2))


if __name__ == "__main__":
    main()
