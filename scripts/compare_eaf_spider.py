"""Compare per-dataset vs universal EAF forecasters via a spider (radar) plot.

Data sources
------------
--per-dataset-csv   CSV produced by train_per_dataset_eaf.py
                    columns: dataset, val_rho, test_rho

--universal-json    One or more JSON files produced by train_forecaster_unsupervised.py
                    key: per_dataset → {dataset: {rho_forecaster, rho_token_norm, delta}}
                    Multiple files can be passed to compare e.g. h256 vs h512.

Output
------
results/ablations/eaf_rho_{model_name}.png   — spider plot
results/ablations/eaf_rho_{model_name}.csv   — numeric table (all series × datasets)

Usage examples
--------------
# Per-dataset only (no universal JSON yet)
python scripts/compare_eaf_spider.py \\
    --model-name uni \\
    --per-dataset-csv logs/per_dataset_eaf_rho.csv

# Full comparison (per-dataset + universal)
python scripts/compare_eaf_spider.py \\
    --model-name uni \\
    --per-dataset-csv logs/per_dataset_eaf_rho.csv \\
    --universal-json checkpoints/unsupervised/uni_forecaster/results_forecaster_uni_src02_attn23_universal.json

# Multiple universal models side-by-side
python scripts/compare_eaf_spider.py \\
    --model-name uni \\
    --per-dataset-csv logs/per_dataset_eaf_rho.csv \\
    --universal-json \\
        checkpoints/unsupervised/uni_forecaster/results_forecaster_uni_src02_attn23_universal.json \\
        checkpoints/unsupervised/uni_forecaster_h512/results_forecaster_uni_src02_attn23_universal.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE = [
    ("#E65100", "-",  2.0),   # universal models: dark orange, solid
    ("#2E7D32", "-",  2.0),   # dark green
    ("#6A1B9A", "-",  2.0),   # purple
    ("#00838F", "-",  2.0),   # teal
]
_PER_DATASET_STYLE = ("#1565C0", "-",  2.2)   # dark blue, solid
_BASELINE_STYLE    = ("#757575", "--", 1.4)   # grey, dashed


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_per_dataset_csv(path, metric="test_rho"):
    """Return {dataset: rho} from train_per_dataset_eaf.py output CSV."""
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                data[row["dataset"]] = float(row[metric])
            except (KeyError, ValueError):
                pass
    return data


def _load_universal_json(path):
    """Return (forecaster_rho, baseline_rho, label) from train_forecaster_unsupervised.py JSON."""
    with open(path) as f:
        obj = json.load(f)
    per_ds = obj.get("per_dataset", {})
    forecaster = {k: v["rho_forecaster"] for k, v in per_ds.items()}
    baseline   = {k: v["rho_token_norm"]  for k, v in per_ds.items()}
    layers = obj.get("layers_source", obj.get("layer_source", "?"))
    if isinstance(layers, list):
        src_tag = "+".join(map(str, layers))
    else:
        src_tag = str(layers)
    label = f"universal (src={src_tag})"
    return forecaster, baseline, label


# ---------------------------------------------------------------------------
# Spider plot
# ---------------------------------------------------------------------------

def _make_spider(series, datasets, title, out_png):
    """
    series : list of (label, {dataset: rho}, color, linestyle, linewidth, fill)
    datasets: ordered list of axis labels
    """
    N = len(datasets)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles_plot = np.append(angles, angles[0])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    all_vals = []
    legend_handles = []

    for label, rho_dict, color, ls, lw, do_fill in series:
        values = [rho_dict.get(d, np.nan) for d in datasets]
        values_plot = list(values) + [values[0]]
        vals_arr = np.array(values_plot, dtype=float)
        has_nan = np.isnan(vals_arr)

        # plot segments, skipping NaN gaps
        if has_nan.any():
            # draw point-by-point, skip gaps
            for i in range(len(vals_arr) - 1):
                if not np.isnan(vals_arr[i]) and not np.isnan(vals_arr[i + 1]):
                    ax.plot(angles_plot[i:i+2], vals_arr[i:i+2],
                            color=color, linestyle=ls, linewidth=lw)
        else:
            ax.plot(angles_plot, vals_arr, color=color, linestyle=ls, linewidth=lw)
            if do_fill:
                ax.fill(angles_plot, vals_arr, alpha=0.12, color=color)

        all_vals.extend([v for v in values if not np.isnan(v)])
        legend_handles.append(
            mpatches.Patch(color=color, label=label, alpha=0.8)
        )

    # radial limits
    lo = max(0.0, min(all_vals) - 0.05) if all_vals else 0.0
    ax.set_ylim(lo, 1.0)
    ax.set_yticks(np.round(np.linspace(lo, 1.0, 5), 2))
    ax.set_rlabel_position(20)

    ax.set_xticks(angles)
    ax.set_xticklabels(datasets, size=9)

    ax.set_title(title, size=13, pad=24, fontweight="bold")
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.38, 1.18), fontsize=10)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Spider plot  → {out_png}")


# ---------------------------------------------------------------------------
# Summary table (stdout + CSV)
# ---------------------------------------------------------------------------

def _save_table(series, datasets, out_csv):
    """Write a CSV with columns: dataset, <series labels...>"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["dataset"] + [s[0] for s in series]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for ds in datasets:
            row = [ds] + [
                round(s[1].get(ds, float("nan")), 6) for s in series
            ]
            writer.writerow(row)
    print(f"Numeric table → {out_csv}")


def _print_table(series, datasets):
    col_w = max(len(s[0]) for s in series) + 2
    header = f"{'dataset':>25}" + "".join(f"{s[0]:>{col_w}}" for s in series)
    print("\n" + header)
    print("-" * len(header))
    for ds in datasets:
        row = f"{ds:>25}" + "".join(
            f"{s[1].get(ds, float('nan')):>{col_w}.4f}" for s in series
        )
        print(row)

    # per-series averages (ignoring NaN)
    print("-" * len(header))
    avgs = []
    for s in series:
        vals = [v for v in s[1].values() if not np.isnan(v)]
        avgs.append(np.mean(vals) if vals else float("nan"))
    avg_row = f"{'avg':>25}" + "".join(f"{a:>{col_w}.4f}" for a in avgs)
    print(avg_row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Spider plot comparing per-dataset vs universal EAF Spearman rho"
    )
    ap.add_argument("--model-name",       type=str, required=True,
                    help="Encoder name used in filenames and plot title.")
    ap.add_argument("--per-dataset-csv",  type=str, default=None,
                    help="CSV from train_per_dataset_eaf.py "
                         "(columns: dataset, val_rho, test_rho).")
    ap.add_argument("--universal-json",   type=str, nargs="+", default=None,
                    help="One or more JSON files from train_forecaster_unsupervised.py. "
                         "Each adds one line to the spider plot.")
    ap.add_argument("--metric",           type=str, default="test_rho",
                    choices=["test_rho", "val_rho"],
                    help="Which column to use from --per-dataset-csv (default: test_rho).")
    ap.add_argument("--datasets",         type=str, nargs="+", default=None,
                    help="Explicit dataset order. Default: union of all sources, sorted.")
    ap.add_argument("--no-baseline",      action="store_true",
                    help="Omit the token-norm baseline line from the plot.")
    ap.add_argument("--out-dir",          type=str, default="results/ablations",
                    help="Output directory (default: results/ablations).")
    args = ap.parse_args()

    if args.per_dataset_csv is None and args.universal_json is None:
        ap.error("Provide at least one of --per-dataset-csv or --universal-json.")

    # ------------------------------------------------------------------
    # Load all data sources
    # ------------------------------------------------------------------
    series = []         # (label, {ds: rho}, color, linestyle, linewidth, fill)
    baseline_rho = {}   # token-norm baseline from first universal JSON

    # Per-dataset EAF
    if args.per_dataset_csv:
        pd_rho = _load_per_dataset_csv(args.per_dataset_csv, args.metric)
        if not pd_rho:
            print(f"[warn] No data loaded from {args.per_dataset_csv}")
        else:
            c, ls, lw = _PER_DATASET_STYLE
            series.append((f"per-dataset EAF ({args.metric})", pd_rho, c, ls, lw, True))

    # Universal EAF (one entry per JSON)
    if args.universal_json:
        for i, jpath in enumerate(args.universal_json):
            jpath = Path(jpath)
            if not jpath.exists():
                print(f"[warn] JSON not found: {jpath} — skip")
                continue
            fc_rho, bl_rho, label = _load_universal_json(jpath)
            c, ls, lw = _PALETTE[i % len(_PALETTE)]
            series.append((label, fc_rho, c, ls, lw, True))
            if not baseline_rho:
                baseline_rho = bl_rho   # use first JSON's baseline

    if not series:
        print("No data loaded. Exiting.")
        return

    # Token-norm baseline (optional, from universal JSON)
    if baseline_rho and not args.no_baseline:
        c, ls, lw = _BASELINE_STYLE
        series.append(("token-norm baseline", baseline_rho, c, ls, lw, False))

    # ------------------------------------------------------------------
    # Dataset list
    # ------------------------------------------------------------------
    if args.datasets:
        datasets = args.datasets
    else:
        all_ds = set()
        for _, rho_dict, *_ in series:
            all_ds.update(rho_dict.keys())
        datasets = sorted(all_ds)

    # ------------------------------------------------------------------
    # Build averages for title
    # ------------------------------------------------------------------
    avg_parts = []
    for label, rho_dict, *_ in series:
        vals = [rho_dict.get(d) for d in datasets if rho_dict.get(d) is not None]
        if vals:
            avg_parts.append(f"{label}={np.mean(vals):.3f}")
    title = (
        f"EAF Spearman ρ — {args.model_name}\n"
        + "  ".join(avg_parts)
    )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    out_dir  = Path(args.out_dir)
    stem     = f"eaf_rho_{args.model_name}"
    out_png  = out_dir / f"{stem}.png"
    out_csv  = out_dir / f"{stem}.csv"

    # ------------------------------------------------------------------
    # Plot + table
    # ------------------------------------------------------------------
    _print_table(series, datasets)
    _make_spider(series, datasets, title, out_png)
    _save_table(series, datasets, out_csv)


if __name__ == "__main__":
    main()
